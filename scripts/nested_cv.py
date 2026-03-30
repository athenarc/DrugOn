from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import  clone

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

import warnings


# ------------------------------
# Helpers / Registries
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested CV training + model selection (inner) + unbiased evaluation (outer)."
    )
    p.add_argument("--data", required=True, help="Path to input CSV with a label column.")
    p.add_argument("--apply_preprocess", action="store_true", help="Whether to apply custom preprocessing steps.")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config describing models.")
    p.add_argument("--outdir", default="./model_outputs_nestedcv", help="Output directory.")
    p.add_argument("--basename", default="experiment", help="Base name for outputs.")
    return p.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in [".yaml", ".yml"]:
        import yaml
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
        alias = estimator_path.split(".")[-1].lower()
    est = Estimator(**(fixed_params or {}))
    pipe = Pipeline([("preprocessor", preprocessor), (alias, est)])
    return pipe, alias


def expand_param_grid(
    grid: Dict[str, Any],
    numeric_transformers: List[str] | None,
    alias: str
) -> Dict[str, Any]:
    """
    Builds a param_grid for GridSearchCV.
    - estimator params in config must already be prefixed like '{alias}__param'
    - numeric_transformers controls preprocessor__num choices
    """
    param_grid = grid.copy() if grid else {}

    scalers: List[Any] = []
    if numeric_transformers:
        for item in numeric_transformers:
            reg = SCALER_REGISTRY.get(item)
            if reg is None:
                raise ValueError(f"Unknown numeric transformer '{item}'")
            scalers.append(reg if reg == "passthrough" else reg())
    else:
        scalers = ["passthrough"]

    param_grid["preprocessor__num"] = scalers
    return param_grid


def safe_predict_proba_or_score(clf: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Returns a (n_samples,) float array usable for AUC:
    - prefer predict_proba[:, 1]
    - else decision_function rescaled to [0,1]
    - else zeros
    """
    step_names = list(clf.named_steps.keys())
    est = clf.named_steps[step_names[-1]]

    if hasattr(est, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
        return prob

    if hasattr(est, "decision_function"):
        scores = clf.decision_function(X)
        smin, smax = float(np.min(scores)), float(np.max(scores))
        prob = (scores - smin) / (smax - smin + 1e-12)
        return prob

    # fallback
    return np.zeros(len(clf.predict(X)), dtype=float)


# ------------------------------
# Nested CV training/evaluation
# ------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    outdir = ensure_outdir(args.outdir)
    base = Path(outdir) / args.basename

    # Load data
    df = pd.read_csv(args.data)
    apply_preprocess = args.apply_preprocess

    g = cfg.get("globals", {})
    label_col = g.get("label_col", "label")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in data.")

    df = drop_by_prefix(df, g.get("drop_prefixes", []))
    # df.drop(columns=['person_id','race','gender'], inplace=True)
    print(df.shape)
    zero_cols = df.columns[(df == 0).all()]
    df = df.drop(columns=zero_cols)
    print(df.shape)

    # If your original script always drops gender, keep it safe/optional here:

    y = df[label_col].astype(int)  # assumes binary labels 0/1
    X = df.drop(columns=[label_col])


    # CV configuration
    outer_splits = int(g.get("outer_cv", g.get("cv", 5)))
    inner_splits = int(g.get("inner_cv", 5))
    random_state = int(g.get("random_state", 42))
    scoring = g.get("scoring", "roc_auc")  # GridSearchCV metric (selection criterion)
    n_jobs = int(g.get("n_jobs", -1))

    cv_outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    cv_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    models_cfg = cfg.get("models", [])
    if not models_cfg:
        raise ValueError("No models specified in config under 'models'.")

    rows: List[Dict[str, Any]] = []

    for m in models_cfg:
        name = m["name"]
        estimator_path = m["estimator"]
        fixed_params = m.get("fixed_params", {}) or {}

        numeric_transformers = m.get("numeric_transformers", ["passthrough"])

        # Build a preprocessor placeholder (scaler swapped by grid)
        preprocessor, _, _ = build_preprocessor(X, "passthrough")
        pipe, alias = build_pipeline(preprocessor, estimator_path, fixed_params)

        # Build param grid
        grid = m.get("param_grid", {}) or {}
        param_grid = expand_param_grid(grid, numeric_transformers, alias)

        # Collect outer-fold metrics
        outer_auc: List[float] = []
        outer_bal_acc: List[float] = []
        outer_recall: List[float] = []
        outer_acc: List[float] = []

        print(f"\n=== Nested CV: {name} ({estimator_path}) ===")

        for fold_idx, (train_ix, test_ix) in enumerate(cv_outer.split(X, y), start=1):
            print(f"\n--- Outer Fold {fold_idx:02d}/{outer_splits} ---")
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            print(X_train.shape)

            if apply_preprocess:
                train_df = X_train.copy()
                train_df["label"] = y_train.values

                feature_cols = [c for c in train_df.columns if c != "label"]

                train_df = (
                    train_df.groupby(feature_cols, dropna=False, group_keys=False)
                    .apply(lambda g: g[g["label"] == g["label"].mode().iloc[0]])
                )

                y_train = train_df["label"].astype(int)
                X_train = train_df.drop(columns=["label"])
            
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            print(X_train.shape)

            if (estimator_path in ["xgboost.XGBClassifier", "catboost.CatBoostClassifier"]) and ("scale_pos_weight" in fixed_params):
                neg = n_neg
                pos = n_pos
                spw = (neg / pos) if pos > 0 else 1.0

            # Inner: hyperparameter tuning only on X_train
            pipe_fold = clone(pipe)

            # only set if the estimator supports it (avoids crashing for models that don't)
            estimator_params = pipe_fold.named_steps[alias].get_params()
            if "scale_pos_weight" in estimator_params:
                pipe_fold.set_params(**{f"{alias}__scale_pos_weight": spw})
            print("Running inner CV")
            gs = GridSearchCV(
                estimator=pipe_fold,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv_inner,
                refit=True,
                n_jobs=n_jobs,
                verbose=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            # Outer: evaluate once on held-out outer test
            y_prob = safe_predict_proba_or_score(best_model, X_test)
            y_pred = best_model.predict(X_test)

            # AUC (handle edge cases where a fold has one class)
            try:
                auc = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                auc = float("nan")

            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            rec = float(recall_score(y_test, y_pred, pos_label=1))

            outer_auc.append(auc)
            outer_bal_acc.append(bal_acc)
            outer_recall.append(rec)
            outer_acc.append(acc)

            print(
                f"Fold {fold_idx:02d}/{outer_splits} | "
                f"AUC={auc:.4f} | BalAcc={bal_acc:.4f} | Recall={rec:.4f} | Acc={acc:.4f} | "
                f"best={gs.best_params_}"
            )

        # Aggregate (mean metrics requested)
        auc_mean = float(np.nanmean(outer_auc))
        auc_std = float(np.nanstd(outer_auc))
        acc_mean = float(np.mean(outer_acc))
        acc_std = float(np.std(outer_acc))
        bal_acc_mean = float(np.mean(outer_bal_acc))
        bal_acc_std = float(np.std(outer_bal_acc))
        recall_mean = float(np.mean(outer_recall))
        recall_std = float(np.std(outer_recall))

        rows.append(
            {
                "model_name": name,
                "estimator": estimator_path,
                # "n_pos": n_pos,
                # "n_neg": n_neg,
                "outer_folds": outer_splits,
                "inner_folds": inner_splits,
                "selection_scoring": scoring,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "balanced_accuracy_mean": bal_acc_mean,
                "balanced_accuracy_std": bal_acc_std,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
            }
        )

    # Save final CSV with mean metrics per model
    out_csv = f"{base}_NESTEDCV_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved nested-CV summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()
