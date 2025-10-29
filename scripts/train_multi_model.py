from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Your utilities
from evaluation.lib import save_shap_summary,evaluate_classification_metrics, get_sorted_feature_importance

import joblib
import warnings

# ------------------------------
# Helpers
# ------------------------------

SCALER_REGISTRY = {
    "passthrough": "passthrough",
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}

# Map estimator import path to a short pipeline step name
ESTIMATOR_ALIAS = {
    "xgboost.XGBClassifier": "xgb",
    "sklearn.ensemble.RandomForestClassifier": "rf",
    "sklearn.linear_model.LogisticRegression": "lr",
    "catboost.CatBoostClassifier": "catboost",
}

def dynamic_import(path: str):
    """
    Import a class given a "module.ClassName" string.
    """
    mod_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiple ML models from config and compare outputs.")
    p.add_argument("--data", required=True, help="Path to input CSV with a label column.")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config describing models.")
    p.add_argument("--outdir", default="./model_outputs_multi", help="Output directory.")
    p.add_argument("--basename", default="experiment", help="Base name for outputs.")
    return p.parse_args()

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in [".yaml", ".yml"]:
        import yaml  # local import so PyYAML is optional if using JSON
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config format. Use .yaml/.yml or .json")

def ensure_outdir(p: str | Path) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

def drop_by_prefix(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    if not prefixes:
        return df
    cols_to_drop = [c for c in df.columns if any(c.startswith(pref) for pref in prefixes)]
    return df.drop(columns=cols_to_drop, errors="ignore")

def build_preprocessor(X: pd.DataFrame, numeric_transformer: Any) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols

def build_pipeline(preprocessor: ColumnTransformer, estimator_path: str, fixed_params: Dict[str, Any]) -> Tuple[Pipeline, str]:
    Estimator = dynamic_import(estimator_path)
    alias = ESTIMATOR_ALIAS.get(estimator_path)
    if alias is None:
        # create a safe alias if not listed
        alias = estimator_path.split(".")[-1].lower()
    est = Estimator(**(fixed_params or {}))
    pipe = Pipeline([("preprocessor", preprocessor), (alias, est)])
    return pipe, alias

def expand_param_grid(grid: Dict[str, Any], numeric_transformers: List[str | Dict[str, Any]] | None, alias: str):
    """
    Turn config entries into a valid param_grid for GridSearchCV.
    - numeric_transformers: list of names matching SCALER_REGISTRY keys (e.g., 'passthrough', 'StandardScaler', ...)
    - grid keys must already be prefixed with '{alias}__' for estimator params.
    """
    param_grid = grid.copy() if grid else {}

    # normalize numeric_transformers
    scalers = []
    if numeric_transformers:
        for item in numeric_transformers:
            if isinstance(item, str):
                reg = SCALER_REGISTRY.get(item)
                if reg is None:
                    raise ValueError(f"Unknown numeric transformer '{item}'")
                scalers.append(reg if reg == "passthrough" else reg())
            else:
                raise ValueError("numeric_transformers entries must be strings referencing SCALER_REGISTRY")
    else:
        scalers = ["passthrough"]

    param_grid["preprocessor__num"] = scalers
    return param_grid

def safe_predict_proba_or_score(clf: Pipeline, X):
    step_names = list(clf.named_steps.keys())
    # last step is the estimator
    est = clf.named_steps[step_names[-1]]
    if hasattr(est, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
    else:
        # scale decision_function into [0,1] if available
        if hasattr(est, "decision_function"):
            scores = clf.decision_function(X)
            smin, smax = np.min(scores), np.max(scores)
            prob = (scores - smin) / (smax - smin + 1e-12)
        else:
            prob = np.zeros(len(clf.predict(X)), dtype=float)
    return prob

def generic_feature_importance(
    fitted_pipeline: Pipeline,
    X_val,
    y_val,
    feature_names: List[str],
    estimator_alias: str,
) -> pd.DataFrame:
    """
    Try model-native importance, then linear coef_, else permutation importance.
    Returns a df with ['feature','importance','method'] sorted descending.
    """
    est = fitted_pipeline.named_steps[estimator_alias]

    # 1) Tree-based .feature_importances_
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": vals})
            .sort_values("importance", ascending=False)
            .assign(method="model_native")
        )

    # 2) Linear coef_
    if hasattr(est, "coef_"):
        coef = est.coef_.ravel() if np.ndim(est.coef_) > 1 else est.coef_
        vals = np.abs(coef)
        return (
            pd.DataFrame({"feature": feature_names, "importance": vals})
            .sort_values("importance", ascending=False)
            .assign(method="abs_coef")
        )

    # 3) Permutation importance on validation set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(
            fitted_pipeline, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
        )
    vals = r.importances_mean
    return (
        pd.DataFrame({"feature": feature_names, "importance": vals})
        .sort_values("importance", ascending=False)
        .assign(method="permutation")
    )

# def compute_shap(
#     fitted_pipeline: Pipeline,
#     X_test: pd.DataFrame,
#     feature_names: List[str],
#     estimator_alias: str,
#     out_base: Path,
#     kernel_sample_size: int = 200,
#     background_size: int = 100,
# ):
#     """
#     Save SHAP summary for any model.
#     Prefer TreeExplainer -> LinearExplainer -> KernelExplainer (sampled).
#     Produces:
#       - {base}_shap_summary.png  (classic beeswarm, not interaction plot)
#     """
#     try:
#         import shap
#         import matplotlib.pyplot as plt
#         import numpy as np
#         import pandas as pd
#     except Exception as e:
#         print(f"SHAP unavailable: {e}")
#         return

#     # Transform X_test into model space
#     Xt = fitted_pipeline.named_steps["preprocessor"].transform(X_test)
#     Xt_df = pd.DataFrame(Xt, columns=feature_names)

#     # Pick estimator
#     est = fitted_pipeline.named_steps[estimator_alias]

#     # --- build explainer & compute values (may return Explanation or ndarray/list) ---
#     explanation_or_values = None
#     try:
#         # Tree-based (XGB, RF, etc.)
#         expl = shap.TreeExplainer(est)
#         explanation_or_values = expl(Xt_df)
#     except Exception:
#         try:
#             # Linear models
#             expl = shap.LinearExplainer(est, Xt_df, feature_perturbation="interventional")
#             explanation_or_values = expl(Xt_df)
#         except Exception:
#             try:
#                 # Kernel fallback (sample to keep runtime reasonable)
#                 n_bg = min(background_size, len(Xt_df))
#                 n_eval = min(kernel_sample_size, len(Xt_df))
#                 background = shap.sample(Xt_df, n_bg, random_state=0)
#                 eval_set = shap.sample(Xt_df, n_eval, random_state=1)
#                 predict_fn = est.predict_proba if hasattr(est, "predict_proba") else est.predict
#                 expl = shap.KernelExplainer(predict_fn, background)
#                 explanation_or_values = expl(eval_set)
#                 Xt_df = eval_set  # align plotting data with what we explained
#             except Exception as e:
#                 print(f"SHAP explainability skipped: {e}")
#                 return

#     # --- ensure we have 2-D main-effect SHAP values for beeswarm ---
#     # Handle shap.Explanation, ndarray, or list outputs
#     try:
#         # If it's a shap.Explanation
#         values = getattr(explanation_or_values, "values", None)
#         if values is None:
#             # Could be ndarray or list
#             values = np.array(explanation_or_values)

#         # Some backends can return a list (e.g., multiclass); take class 1 if so
#         if isinstance(values, list):
#             values = np.array(values[-1])

#         # If interaction values (n x f x f), collapse to main effects
#         if values.ndim == 3:
#             values = values.sum(axis=2)

#         # If multiclass 3-D in the form (n_classes, n_samples, n_features), take positive class
#         if values.ndim == 3 and values.shape[0] <= 10:  # heuristic for class-first layout
#             values = values[-1]  # take last class (typically the "positive" class)

#     except Exception as e:
#         print(f"Failed to coerce SHAP values to 2-D: {e}")
#         return

#     # --- plot classic beeswarm (dot) ---
#     shap.summary_plot(values, Xt_df, plot_type="dot", show=False)
#     plt.tight_layout()
#     plt.savefig(f"{out_base}_shap_summary.png", dpi=200)
#     plt.close()
def compute_shap(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    feature_names: List[str],
    estimator_alias: str,
    out_base: Path,
    kernel_sample_size: int = 200,
    background_size: int = 100,
):
    """
    Save SHAP summary + dependence + PDP/ICE for any model.
    Creates:
      - {base}_shap_summary.png      (beeswarm; main effects only)
      - {base}_shap_dependence_num__exposure_furosemide.png
      - {base}_pdp_ice_exposure_furosemide.png
    """
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.inspection import PartialDependenceDisplay
    except Exception as e:
        print(f"SHAP/PDP unavailable: {e}")
        return

    # Transform X_test into model space
    Xt = fitted_pipeline.named_steps["preprocessor"].transform(X_test)
    Xt_df = pd.DataFrame(Xt, columns=feature_names)

    est = fitted_pipeline.named_steps[estimator_alias]

    # ---------- Compute SHAP (prefer Tree -> Linear -> Kernel) ----------
    explanation_or_values = None
    try:
        expl = shap.TreeExplainer(est)
        explanation_or_values = expl(Xt_df)
    except Exception:
        try:
            expl = shap.LinearExplainer(est, Xt_df, feature_perturbation="interventional")
            explanation_or_values = expl(Xt_df)
        except Exception:
            try:
                n_bg = min(background_size, len(Xt_df))
                n_eval = min(kernel_sample_size, len(Xt_df))
                background = shap.sample(Xt_df, n_bg, random_state=0)
                eval_set = shap.sample(Xt_df, n_eval, random_state=1)
                predict_fn = est.predict_proba if hasattr(est, "predict_proba") else est.predict
                expl = shap.KernelExplainer(predict_fn, background)
                explanation_or_values = expl(eval_set)
                Xt_df = eval_set  # align what we plot with what we explained
            except Exception as e:
                print(f"SHAP explainability skipped: {e}")
                return

    # ---------- Coerce to 2-D main effects for classic beeswarm ----------
    try:
        values = getattr(explanation_or_values, "values", None)
        if values is None:
            values = np.array(explanation_or_values)
        if isinstance(values, list):
            values = np.array(values[-1])  # pick positive class if list
        if values.ndim == 3:
            # collapse interaction tensor or (n_classes, n, f)
            if values.shape[0] <= 10:  # likely class-first layout
                values = values[-1]
            else:
                values = values.sum(axis=2)
    except Exception as e:
        print(f"Failed to coerce SHAP values to 2-D: {e}")
        return

    
    # ---------- 1) Beeswarm summary ----------
    shap.summary_plot(values, Xt_df, plot_type="dot", show=False)
    save_shap_summary(
    shap_values=values,
    feature_matrix=Xt_df,   # after your pipeline's preprocessor
    feature_names=feature_names,
    out_path=f"{out_base}_shap_summary.png",
    max_display=20,
    figsize=(12, 10),
    dpi=200,
    left=0.35,   # increase if your labels are even longer
    right=0.95
)
    # plt.tight_layout()
    # plt.savefig(f"{out_base}_shap_summary.png", dpi=200)
    # plt.close()

    # ---------- 2) SHAP dependence for furosemide ----------
    # feat_t = "num__exposure_furosemide"
    # if feat_t in Xt_df.columns:
    #     color_by = "num__age_at_outcome" if "num__age_at_outcome" in Xt_df.columns else "auto"
    #     shap.dependence_plot(feat_t, values, Xt_df, interaction_index=color_by, show=False)
    #     plt.tight_layout()
    #     plt.savefig(f"{out_base}_shap_dependence_{feat_t}.png", dpi=200)
    #     plt.close()
    # else:
    #     print("Dependence plot skipped: 'num__exposure_furosemide' not found in transformed features.")

    # ---------- 3) PDP/ICE for furosemide (on original feature space) ----------
    # Map transformed name -> raw column name for numeric columns from ColumnTransformer
    # ('num__feature' -> 'feature'); PDP works on the *raw* X columns through the Pipeline.
    raw_feat = "exposure_furosemide"
    try:
        if raw_feat in X_test.columns:
            PartialDependenceDisplay.from_estimator(
                fitted_pipeline,
                X_test,
                features=[raw_feat],   # original column name
                kind="both"            # PDP + ICE
            )
            plt.tight_layout()
            plt.savefig(f"{out_base}_pdp_ice_{raw_feat}.png", dpi=200)
            plt.close()
        else:
            print(f"PDP/ICE skipped: raw feature '{raw_feat}' not found in X_test columns.")
    except Exception as e:
        print(f"PDP/ICE failed: {e}")



def main():
    args = parse_args()
    cfg = load_config(args.config)

    outdir = ensure_outdir(args.outdir)
    base = Path(outdir) / args.basename

    # ------------------------------
    # Load data
    # ------------------------------
    df = pd.read_csv(args.data, index_col=0)
    g = cfg.get("globals", {})
    label_col = g.get("label_col", "label")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in data.")

    df = drop_by_prefix(df, g.get("drop_prefixes", []))

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Split once, reuse for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=g.get("test_size", 0.2),
        random_state=g.get("random_state", 42),
        stratify=y
    )

    # For consistent feature name extraction after fit
    # We'll build preprocessor with a placeholder scaler (overridden by grid)
    pre_placeholder, num_cols, cat_cols = build_preprocessor(X_train, "passthrough")

    # Storage for cross-model comparisons
    metrics_rows = []
    importance_tables = []  # list of (model_name, df)
    predictions_tables = []  # list of (model_name, df)

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models specified in config under 'models'.")

    for m in models_cfg:
        name = m["name"]
        try:
            estimator_path = m["estimator"]
            fixed_params = m.get("fixed_params", {})
            numeric_transformers = m.get("numeric_transformers", ["passthrough"])

            # Build a fresh preprocessor (scaler will be replaced by grid search)
            preprocessor, _, _ = build_preprocessor(X_train, "passthrough")

            # Build pipeline
            pipe, alias = build_pipeline(preprocessor, estimator_path, fixed_params)

            # Assemble param_grid (includes numeric scalers)
            grid = m.get("param_grid", {})
            param_grid = expand_param_grid(grid, numeric_transformers, alias)

            scoring = g.get("scoring", "accuracy")
            cv = g.get("cv", 5)
            n_jobs = g.get("n_jobs", -1)

            print(f"\n=== Training {name} ({estimator_path}) ===")
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                verbose=1,
                n_jobs=n_jobs,
            )
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            print(f"Best params for {name}: {best_params}")

            # Rebuild best pipeline explicitly (optional; GridSearchCV.best_estimator_ is okay too)
            best_pipe = gs.best_estimator_
            best_pipe.fit(X_train, y_train)

            # Outputs base path per model
            model_base = Path(outdir) / f"{args.basename}_{name}"

            # Predictions & metrics
            y_pred = best_pipe.predict(X_test)
            y_prob = safe_predict_proba_or_score(best_pipe, X_test)

            result = evaluate_classification_metrics(
                y_all=df[label_col],
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                analysis_id=1,
                model_name=name,
                split_desc=f"{int((1-g.get('test_size',0.2))*100)}/{int(g.get('test_size',0.2)*100)} split"
            )
            metrics_dict = dict(result)
            metrics_rows.append(metrics_dict)

            

            # Feature names after transform
            feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()

            # Prefer your existing importance util; if it fails, use generic fallback
            importance_df = None
            try:
                importance_df = get_sorted_feature_importance(best_pipe, feature_names)
                importance_df = importance_df.rename(columns={importance_df.columns[0]: "feature", importance_df.columns[1]: "importance"})
                importance_df["method"] = "library_util"
            except Exception as e:
                print(f"get_sorted_feature_importance failed for {name}: {e}")
                importance_df = generic_feature_importance(best_pipe, X_test, y_test, list(feature_names), alias)

            # Normalize for cross-model comparability
            if "importance" in importance_df.columns:
                total = importance_df["importance"].abs().sum()
                if total > 0:
                    importance_df["importance_normalized"] = importance_df["importance"].abs() / total
                else:
                    importance_df["importance_normalized"] = 0.0

            importance_tables.append((name, importance_df))

            # Save pipeline
            joblib.dump(best_pipe, f"{model_base}_pipeline.joblib")

            # SHAP explainability
            try:
                compute_shap(
                    fitted_pipeline=best_pipe,
                    X_test=X_test,
                    feature_names=list(feature_names),
                    estimator_alias=alias,
                    out_base=model_base,
                    kernel_sample_size=g.get("shap_kernel_sample_size", 200),
                    background_size=g.get("shap_background_size", 100)
                )
            except Exception as e:
                print(f"SHAP failed for {name}: {e}")
        except Exception as e:
            print(f"❌ Skipping model '{name}' due to error: {e}")
            continue
    # ------------------------------
    # Cross-model comparison exports
    # ------------------------------

    # 1) Combined metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f"{base}_ALL_metrics.csv", index=False)

    # 2) Side-by-side top features (normalized) per model
    #    Build a wide table: one column per model’s normalized importance
    wide = None
    for name, imp in importance_tables:
        sub = imp[["feature", "importance_normalized"]].copy()
        sub = sub.rename(columns={"importance_normalized": f"{name}_importance"})
        wide = sub if wide is None else wide.merge(sub, on="feature", how="outer")
    if wide is not None:
        # also include mean rank across models
        for col in [c for c in wide.columns if c.endswith("_importance")]:
            # convert NaN to 0 before ranking
            wide[col] = wide[col].fillna(0.0)
        rank_cols = []
        for col in [c for c in wide.columns if c.endswith("_importance")]:
            rcol = col.replace("_importance", "_rank")
            wide[rcol] = wide[col].rank(ascending=False, method="average")
            rank_cols.append(rcol)
        wide["mean_rank"] = wide[rank_cols].mean(axis=1)
        wide.sort_values(["mean_rank"], ascending=True, inplace=True)
        wide.to_csv(f"{base}_ALL_feature_importance_comparison.csv", index=False)

    print(f"\nSaved all outputs to: {outdir}")

if __name__ == "__main__":
    main()
