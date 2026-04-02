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

## Anastasia Graph
```bash
python build_reference_set.py \
  --db-path ../data/input/mimiciv_omop.db \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --threshold 1.2 \
  --time-window-days 365 \
  --top-exposures 956874 1774470 1776684 1178663 1177480 1115008 1124300 45892419 902722 1341927 1308216 1367500 19010482 950637 19081224 45775324 19080985 \
  --extra-conditions 316139 4064161 4078925 201620 197320 46271022 132797 81902 320128 434610 3006906 201820 \
  --outdir ./outputs \
  --basename mimiciv_creatinine_anastasia \
  --save-format csv
```

## Antonia Graph

```bash
  python build_reference_set.py \
  --db-path ../data/input/mimiciv_omop.db \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --threshold 1.2 \
  --time-window-days 365 \
  --top-exposures 956874 1774470 1776684 1178663 1177480 1115008 1124300 45892419 902722 1341927 1308216 1367500 19010482 950637 19081224 45775324 19080985 40235485 1317640 1317967 43526465 44785829 45774751 1308842 1707687 997276 1551803 1705674 1775741 19086888 1555120 975125 1506270 1550557 1518254 1755112 1337107 19011773 1344143 3024561 974166 920458 1347384 40226742 1351557 1346686  \
  --extra-conditions 316139 4064161  4078925 201620 197320 46271022 132797 81902 320128 434610 3006906 201820 4177206 196455 40393236 4236160 441408 196523 437663\
  --outdir ./outputs \
  --basename mimiciv_creatinine_antonia \
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
## Aspirin Pair
```bash
  python build_reference_set.py \
  --db-path ../data/input/mimiciv_omop.db \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --outcome-id 192671 \
  --time-window-days 365 \
  --top-exposures   1112807 373503 1310149 19024063 19084670 1322184 1331270 19029393 1381253 1397141 46275677 1118084 797617 19056874 1124300 1331270 715259 715939 19011355 755695 751412 1177480 1344992 1185922 1115008 19003570 722031 1146810 40163718 739138 19041220 19042155 902726 743670 40241186 44507700 1150345 1301065 1301025 1308473 1367571 19026343 43013024 40241331 1307515 1302398 1347450 43013024 1322199 19098548 19017067 1518254 975125 1506270 903963 1557272 1512480 1325363 1786055 19049105 734275 751347 700253 \
  --extra-conditions 4329041 437663 4329847 312327 4270024 381316 321318 4133004 440417 313217 439393 4236898 434007 434316 40321716 432296 432863 134057 316866 201820 4030518  194984 4131611 4265600 4340781 4096031 192669 318800 36713504 4053830 4112603 30753 23245 4155764 4027663 4074815 194992 4342655 3655249 37017137 3655249 4053591 4131614 4131598 4183009 441394 4130994 196444 192680 313504 195562 4164337 4160603 4134562 4053834 4342752 444397 4130514 \
  --outdir ./outputs \
  --basename mimiciv_creatinine_antonia \
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
