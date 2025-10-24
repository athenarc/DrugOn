# ADE-Causal Pipeline – Scripts

This repository contains helper scripts for:

- **Building reference datasets** from OMOP-like DuckDB databases.
- **Training ML models** (XGBoost pipelines) with metrics & explainability.

---

## Environment Setup

We recommend using a dedicated conda environment.

```bash
# Create environment with Python 3.10.18
conda create -n ade-causal python=3.10.18 
conda activate ade-causal

# Install dependencies
pip install -r requirements.txt

```

## Build Reference Dataset

Script: *scripts/build_reference_set.py* 

Generates a labeled dataset (positives/negatives) for causal/ML modeling.

### Modes
From inside the scripts directory run the following commands based on the mode you want:


**Threshold mode**: DAG run where creatinine is not a found concept_id and we define it by a measurement threshold (e.g., creatinine > 1.2).

```bash

python build_reference_set.py \
  --db-path ../data/input/database-1M_filtered.duckdb \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --threshold 1.2 \
  --time-window-days 365 \
  --top-exposures 956874 1177480 950637 1115008 19010482 \
  --extra-conditions 316139 46271022 201820 320128 3006906 197320 132797 81902 4220631 \
  --outdir ./outputs \
  --basename creatinine_gt_1_2 \
  --max-per-class 25000 \
  --save-format csv

```

**Known outcome mode**: define outcomes by an explicit concept ID, optionally discovering top exposures/measurements/procedures.

```bash

python build_reference_set.py \
  --db-path ../data/input/database-1M_filtered.duckdb \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --outcome-id 439777 \
  --time-window-days 365 \
  --discover-top-concepts \
  --discover-n-exposures 10 \
  --discover-n-procedures 5 \
  --outdir ./outputs \
  --basename outcome_439777 \
  --max-per-class 25000 \
  --save-format csv

```

## Train Model and output metrics and explainability
Script: *scripts/train_model.py*
Trains an XGBoost model with preprocessing, hyperparameter search, and SHAP explainability.

**Usage**
```bash
python train_model.py \
  --data ./outputs/creatinine_gt_1_2.csv \
  --drop-prefixes measurement \
  --outdir ./model_outputs \
  --basename xgb_creatinine \
  --test-size 0.2 \
  --scoring roc_auc \
  --n-jobs -1
```

## Train multiple models with train_multi_model.py
```bash
python train_multi_model.py \
  --data ./outputs/creatinine_gt_1_2.csv \
  --config models.yml \
  --outdir ./model_outputs_multis \
  --basename omop_experiment
```
## What it does

- Loads dataset (`--data` must include a `label` column).
- Drops any feature columns with prefixes specified in `--drop-prefixes`.
- Splits into train/test (default 80/20).
- Builds preprocessing pipeline:
  - **Numeric**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or passthrough (tuned)
  - **Categorical**: `OneHotEncoder`
- Runs `GridSearchCV` over:
  - `n_estimators ∈ {100, 200}`
  - `max_depth ∈ {3, 5, 7}`
  - `learning_rate ∈ {0.01, 0.1, 0.2}`
  - scaler choice
- Retrains with best parameters.
- Saves outputs.

---

## Outputs

Saved under `--outdir` with prefix `--basename`:

- **Metrics**
  - `<basename>_metrics.json`
  - `<basename>_metrics.csv`
- **Feature importance**
  - `<basename>_feature_importance.csv`
- **Model artifact**
  - `<basename>_pipeline.joblib`
- **Predictions on test set**
  - `<basename>_predictions.csv`
- **Explainability**
  - `<basename>_shap_summary.png`
  - `<basename>_shap_values.npy` (optional)
