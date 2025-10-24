from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from evaluation.lib import evaluate_classification_metrics,get_sorted_feature_importance

import joblib



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost pipeline and export metrics/explainability.")
    p.add_argument("--data", required=True, help="Path to input CSV with a 'label' column.")
    p.add_argument("--outdir", default="./model_outputs", help="Directory to save outputs.")
    p.add_argument("--basename", default="model", help="Base name for output files.")
    p.add_argument("--drop-prefixes", nargs="*", default=[], help="Column prefixes to drop (e.g., measurement).")
    p.add_argument("--label-col", default="label", help="Target column name.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--scoring", default="accuracy", help="Scoring metric for GridSearchCV (e.g., accuracy, roc_auc).")
    p.add_argument("--cv", type=int, default=5, help="CV folds.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV.")
    return p.parse_args()


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def drop_by_prefix(df: pd.DataFrame, prefixes):
    if not prefixes:
        return df
    cols_to_drop = [c for c in df.columns if any(c.startswith(pref) for pref in prefixes)]
    return df.drop(columns=cols_to_drop, errors="ignore")


def main():
    args = parse_args()
    outdir = ensure_outdir(args.outdir)
    base = Path(outdir) / args.basename

    df = pd.read_csv(args.data,index_col=0)
    if args.label_col not in df.columns:
        raise KeyError(f"Label column '{args.label_col}' not found in data.")

    df = drop_by_prefix(df, args.drop_prefixes)

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    numerical_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    xgb_model = XGBClassifier(use_label_encoder=False, random_state=args.random_state, eval_metric="logloss")
    pipeline = Pipeline([("preprocessor", preprocessor), ("xgb", xgb_model)])

    param_grid = {
        "preprocessor__num": [StandardScaler(), MinMaxScaler(), RobustScaler(), "passthrough"],
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 5, 7],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=args.cv,
        verbose=1,
        n_jobs=args.n_jobs,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    chosen_scaler = best_params.get("preprocessor__num", "passthrough")
    preprocessor_best = ColumnTransformer(
        transformers=[
            ("num", chosen_scaler, numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    xgb_best = XGBClassifier(
        use_label_encoder=False,
        random_state=args.random_state,
        eval_metric="logloss",
        n_estimators=best_params["xgb__n_estimators"],
        max_depth=best_params["xgb__max_depth"],
        learning_rate=best_params["xgb__learning_rate"],
    )
    best_pipeline = Pipeline([("preprocessor", preprocessor_best), ("xgb", xgb_best)])
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline.named_steps["xgb"], "predict_proba"):
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        try:
            scores = best_pipeline.decision_function(X_test)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        except Exception:
            y_prob = np.zeros_like(y_pred, dtype=float)

    result = evaluate_classification_metrics(
            y_all=df[args.label_col],
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            analysis_id=1,
            model_name="XGBoost",
            split_desc=f"{int((1-args.test_size)*100)}/{int(args.test_size*100)} split"
        )
    metrics_dict = dict(result)


    with open(f"{base}_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    pd.DataFrame([metrics_dict]).to_csv(f"{base}_metrics.csv", index=False)

    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance_df = get_sorted_feature_importance(best_pipeline, feature_names)
  
    importance_df.to_csv(f"{base}_feature_importance.csv", index=False)

    joblib.dump(best_pipeline, f"{base}_pipeline.joblib")

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob}).reset_index(drop=True)
    pred_df.to_csv(f"{base}_predictions.csv", index=False)

    try:
        import shap
        import matplotlib.pyplot as plt

        def sanitize_feature_names(df):
            df = df.copy()
            new_columns = {
                col: col.replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt").replace(" ", "_").replace(",", "_")
                for col in df.columns
            }
            df.columns = [new_columns[col] for col in df.columns]
            return df

        X_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        X_transformed_df = sanitize_feature_names(X_transformed_df)

        model = best_pipeline.named_steps["xgb"]
        explainer = shap.Explainer(model)
        shap_values = explainer(X_transformed_df)

        shap.summary_plot(shap_values, X_transformed_df, show=False)
        plt.tight_layout()
        plt.savefig(f"{base}_shap_summary.png", dpi=200)
        plt.close()

        np.save(f"{base}_shap_values.npy", shap_values.values)
    except Exception as e:
        print(f"SHAP explainability skipped due to error: {e}")

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()