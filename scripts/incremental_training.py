from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Your utilities
from evaluation.lib import (
    save_shap_summary,
    evaluate_classification_metrics,
    get_sorted_feature_importance,
)

# ------------------------------
# Registry / aliases
# ------------------------------
SCALER_REGISTRY = {
    "passthrough": "passthrough",
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}

ESTIMATOR_ALIAS = {
    "xgboost.XGBClassifier": "xgb",
    "sklearn.ensemble.RandomForestClassifier": "rf",
    "sklearn.linear_model.LogisticRegression": "lr",
    "catboost.CatBoostClassifier": "catboost",
}


def dynamic_import(path: str):
    mod_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in [".yaml", ".yml"]:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
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
    alias = ESTIMATOR_ALIAS.get(estimator_path, estimator_path.split(".")[-1].lower())
    est = Estimator(**(fixed_params or {}))
    pipe = Pipeline([("preprocessor", preprocessor), (alias, est)])
    return pipe, alias


def expand_param_grid(grid: Dict[str, Any], numeric_transformers: List[str] | None, alias: str):
    param_grid = grid.copy() if grid else {}

    scalers = []
    if numeric_transformers:
        for name in numeric_transformers:
            reg = SCALER_REGISTRY.get(name)
            if reg is None:
                raise ValueError(f"Unknown numeric transformer '{name}'")
            scalers.append(reg if reg == "passthrough" else reg())
    else:
        scalers = ["passthrough"]

    param_grid["preprocessor__num"] = scalers
    return param_grid


def safe_predict_proba_or_score(clf: Pipeline, X):
    step_names = list(clf.named_steps.keys())
    est = clf.named_steps[step_names[-1]]
    if hasattr(est, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        scores = clf.decision_function(X)
        smin, smax = np.min(scores), np.max(scores)
        return (scores - smin) / (smax - smin + 1e-12)
    return np.zeros(len(clf.predict(X)), dtype=float)


def compute_shap(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    feature_names: List[str],
    estimator_alias: str,
    out_base: Path,
    kernel_sample_size: int = 200,
    background_size: int = 100,
):
    # (Your exact compute_shap, unchanged except it writes importance CSV too)
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.inspection import PartialDependenceDisplay
    except Exception as e:
        print(f"SHAP/PDP unavailable: {e}")
        return

    Xt = fitted_pipeline.named_steps["preprocessor"].transform(X_test)
    Xt_df = pd.DataFrame(Xt, columns=feature_names)
    est = fitted_pipeline.named_steps[estimator_alias]

    explanation_or_values = None
    try:
        expl = shap.TreeExplainer(est)
        explanation_or_values = expl(Xt_df)
    except Exception:
        try:
            print("Using LinearExplainer for SHAP values...")
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
                Xt_df = eval_set
            except Exception as e:
                print(f"SHAP explainability skipped: {e}")
                return

    try:
        values = getattr(explanation_or_values, "values", None)
        if values is None:
            values = np.array(explanation_or_values)
        if isinstance(values, list):
            values = np.array(values[-1])
        if values.ndim == 3:
            if values.shape[0] <= 10:
                values = values[-1]
            else:
                values = values.sum(axis=2)
    except Exception as e:
        print(f"Failed to coerce SHAP values to 2-D: {e}")
        return

    # save mean abs shap table
    try:
        assert values.shape[1] == len(feature_names)
        mean_abs = np.mean(np.abs(values), axis=0)
        shap_importance = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_importance["rank"] = np.arange(1, len(shap_importance) + 1)
        shap_importance.to_csv(Path(f"{out_base}_shap_importance.csv"), index=False)
    except Exception as e:
        print(f"Failed to save SHAP importance table: {e}")

    # beeswarm + your custom saver
    import shap
    shap.summary_plot(values, Xt_df, plot_type="dot", show=False)
    save_shap_summary(
        shap_values=values,
        feature_matrix=Xt_df,
        feature_names=feature_names,
        out_path=f"{out_base}_shap_summary.png",
        max_display=20,
        figsize=(12, 10),
        dpi=200,
        left=0.35,
        right=0.95,
    )


def read_ranking_features(xlsx_path: str | Path) -> List[str]:
    ranking = pd.read_excel(xlsx_path)
    if "feature" in ranking.columns:
        feats = ranking["feature"].astype(str).tolist()
    else:
        feats = ranking.iloc[:, 0].astype(str).tolist()
    feats = [f for f in feats if f and f != "nan"]
    return feats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental top-k feature training with SHAP + metrics.")
    p.add_argument("--data", nargs="+", required=True, help="One or more CSVs (each contains label col).")
    p.add_argument("--ranking_xlsx", required=True, help="XLSX with ordered feature names (column 'feature' preferred).")
    p.add_argument("--config", required=True, help="YAML/JSON config (same structure as your trainer).")
    p.add_argument("--outdir", default="./model_outputs_incremental", help="Output directory.")
    p.add_argument("--basename", default="incremental", help="Base name prefix for outputs.")
    p.add_argument("--min_k", type=int, default=15, help="Start with top-k features.")
    p.add_argument("--max_k", type=int, default=None, help="Optional stop at max_k.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    outdir = ensure_outdir(args.outdir)

    g = cfg.get("globals", {})
    label_col = g.get("label_col", "label")

    ranking_feats = read_ranking_features(args.ranking_xlsx)
    if args.max_k is None:
        max_k = len(ranking_feats)
    else:
        max_k = min(args.max_k, len(ranking_feats))

    metrics_rows: List[Dict[str, Any]] = []

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models specified in config under 'models'.")

    for data_path in args.data:
        data_path = Path(data_path)
        ds_name = data_path.stem

        df = pd.read_csv(data_path, index_col=0)  # keep your convention
        df = drop_by_prefix(df, g.get("drop_prefixes", []))

        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in {data_path}")

        y = df[label_col]
        X_full = df.drop(columns=[label_col])

        # keep only ranking features that exist in THIS dataset
        feats_in_data = [f for f in ranking_feats if f in X_full.columns]
        if len(feats_in_data) < args.min_k:
            raise ValueError(
                f"{ds_name}: only {len(feats_in_data)} ranked features exist in data; need at least {args.min_k}."
            )

        # fixed split for fair comparison across k (per dataset)
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_full,
            y,
            test_size=g.get("test_size", 0.2),
            random_state=g.get("random_state", 42),
            stratify=y,
        )

        for k in range(args.min_k, max_k + 1):
            topk = feats_in_data[:k]
            X_train = X_train_full[topk].copy()
            print(X_train.head())
            X_test = X_test_full[topk].copy()

            for m in models_cfg:
                name = m["name"]
                estimator_path = m["estimator"]
                fixed_params = dict(m.get("fixed_params", {}) or {})
                numeric_transformers = m.get("numeric_transformers", ["passthrough"])
                grid = m.get("param_grid", {})

                # scale_pos_weight adjustment (your logic)
                if (estimator_path in ("xgboost.XGBClassifier", "catboost.CatBoostClassifier")) and ("scale_pos_weight" in fixed_params):
                    neg = (y_train == 0).sum()
                    pos = (y_train == 1).sum()
                    fixed_params["scale_pos_weight"] = neg / pos

                preprocessor, _, _ = build_preprocessor(X_train, "passthrough")
                pipe, alias = build_pipeline(preprocessor, estimator_path, fixed_params)
                param_grid = expand_param_grid(grid, numeric_transformers, alias)

                scoring = g.get("scoring", "accuracy")
                cv = g.get("cv", 5)
                n_jobs = g.get("n_jobs", -1)

                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=skf,
                    verbose=0,
                    n_jobs=n_jobs,
                )

                print(f"[{ds_name}] k={k} | Training {name} ...")
                gs.fit(X_train, y_train)
                best_pipe = gs.best_estimator_
                best_pipe.fit(X_train, y_train)

                # outputs base path per run
                run_base = Path(outdir) / f"{args.basename}_{ds_name}_{name}_top{k}"

                # metrics
                y_pred = best_pipe.predict(X_test)
                y_prob = safe_predict_proba_or_score(best_pipe, X_test)

                result = evaluate_classification_metrics(
                    y_all=y,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    analysis_id=1,
                    model_name=name,
                    split_desc=f"{int((1-g.get('test_size',0.2))*100)}/{int(g.get('test_size',0.2)*100)} split",
                )
                metrics_dict = dict(result)
                metrics_dict.update({
                    "dataset": ds_name,
                    "k": k,
                    "topk_features": ";".join(topk),
                })
                metrics_rows.append(metrics_dict)

                # save pipeline
                joblib.dump(best_pipe, f"{run_base}_pipeline.joblib")

                # SHAP
                try:
                    feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
                    compute_shap(
                        fitted_pipeline=best_pipe,
                        X_test=X_test,
                        feature_names=list(feature_names),
                        estimator_alias=alias,
                        out_base=run_base,
                        kernel_sample_size=g.get("shap_kernel_sample_size", 200),
                        background_size=g.get("shap_background_size", 100),
                    )
                except Exception as e:
                    print(f"SHAP failed for {ds_name} {name} k={k}: {e}")

    # Save metrics to ONE Excel file
    metrics_df = pd.DataFrame(metrics_rows)
    out_xlsx = Path(outdir) / f"{args.basename}_ALL_metrics_incremental.xlsx"
    out_csv = Path(outdir) / f"{args.basename}_ALL_metrics_incremental.csv"

    metrics_df.to_csv(out_csv, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        metrics_df.to_excel(w, sheet_name="metrics", index=False)

    print(f"\nSaved metrics CSV: {out_csv}")
    print(f"Saved metrics XLSX: {out_xlsx}")
    print(f"Saved per-run pipelines + SHAP artifacts under: {outdir}")


if __name__ == "__main__":
    main()
