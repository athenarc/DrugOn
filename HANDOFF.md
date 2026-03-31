# DrugOn Repository Handoff

## Purpose

This repository builds labeled datasets from OMOP-style clinical databases and trains ML models to detect or study adverse drug event patterns.

There are two main workstreams:

1. Dataset construction from OMOP tables in DuckDB/Postgres
2. Model training, comparison, explainability, and evaluation on the generated datasets

The production code lives mainly in:

- `features/`
- `scripts/`
- `evaluation/`

The `notebooks/` folder is exploratory and contains research/prototyping work around graph discovery, causal analysis, MIMIC experiments, and manual feature analysis.

## Repository Map

### `features/`

#### `features/sqltranslate.py`

Shared SQL portability helpers.

- `qualify_tables(...)`
  Prefixes bare OMOP table names with the chosen schema.
- `translate_sql(...)`
  Keeps DuckDB SQL as-is or transpiles DuckDB-style SQL to PostgreSQL via `sqlglot`.
- `fetch_df(...)`
  Executes SQL against DuckDB or PostgreSQL and returns a pandas DataFrame.
- `persist_df_duckdb(...)` / `persist_df_postgres(...)`
  Saves intermediate cohort tables used later by feature builders.

This file is the database abstraction layer for the rest of the repo.

#### `features/builder.py`

Core cohort and feature-engineering logic.

Main responsibilities:

- Discover candidate concept IDs associated with an outcome
- Build SQL for positive patients
- Build SQL for negative patients
- Prevent label leakage for threshold-based creatinine outcomes
- Map OMOP concept IDs to human-readable names
- Rename generated columns into interpretable feature names

Important functions:

- `get_top_concept_ids(...)`
  Finds frequent drugs, measurements, and procedures in the lookback window before an outcome.

- `build_feature_query_from_concept_ids(...)`
  Builds the positive-cohort feature query.

  Output includes:
  - demographics: age, gender, race
  - dense burden features:
    - `visit_count`
    - `distinct_drug_count`
    - `distinct_condition_count`
    - `days_observed_before_index`
  - exposure features as binary indicators
  - procedure features as binary indicators
  - extra condition features as binary indicators
  - measurement features as latest pre-index numeric values

- `build_negative_patient_query_random_window(...)`
  Builds the negative-cohort feature query.

  Design:
  - negative patients are anchored at a deterministic pseudo-random `index_date`
  - the same burden features are built for negatives
  - this keeps positive and negative feature windows structurally similar

- `map_all_feature_ids(...)`
  Extracts concept IDs from generated feature columns and retrieves concept names from the OMOP `concept` table.

- `rename_columns_using_concept_names(...)`
  Renames columns like `exposure_956874` into names like `exposure_<concept_name>`.

- `replace_demographic_ids_with_names(...)`
  Replaces `gender_concept_id` and `race_concept_id` with readable categorical columns.

There are also older commented-out versions of the feature-query builders kept in the file. The active implementations are the newer ones that add burden features and tighten joins around the index date.

### `evaluation/`

#### `evaluation/lib.py`

Model evaluation and plotting utilities.

Main functions:

- `evaluate_classification_metrics(...)`
  Computes AUC, PR AUC, accuracy, balanced accuracy, precision, recall, F1, and type I/type II error summaries.

- `get_sorted_feature_importance(...)`
  Reads model-native feature importances from a trained XGBoost pipeline.

- `save_shap_summary(...)`
  Saves SHAP beeswarm plots without clipping labels.

- `select_threshold_max_balanced_accuracy(...)`
  Scans thresholds and selects the one maximizing balanced accuracy.

- `plot_pdp_all_features(...)` and `plot_auc_k(...)`
  Convenience plotting helpers for interpretation and incremental feature experiments.

This file is reused by multiple training scripts.

### `scripts/`

#### `scripts/build_reference_set.py`

Main dataset-building entry point.

This is the script that turns OMOP data into a training table with a `label` column.

It supports two modes:

1. Threshold mode
   The outcome is defined as creatinine above a threshold.

2. Known outcome mode
   The outcome is an explicit `condition_concept_id`.

Key behavior in threshold mode:

- Positives:
  first high-creatinine measurement per patient
- Negatives:
  patients who never crossed the threshold, anchored at first low-creatinine measurement
- Positive and negative cohorts are persisted as:
  - `result_schema.outcome_patients`
  - `result_schema.negative_outcome_patients`
- Features are then built for each cohort separately and combined

Key behavior in known outcome mode:

- Positives are patients with the target condition
- Negatives are patients without that condition, assigned random anchor dates inside observation periods

Final output behavior:

- positive and negative feature tables are concatenated
- `gender`, `race`, and `person_id` are dropped from the final exported dataset
- columns that are entirely zero are dropped
- output is written as CSV

Helper flow inside the script:

- `rename_and_replace(...)`
  Concept-ID-to-name enrichment
- `drop_irrelevant_cols(...)`
  Drops `condition_start_date` and `condition_concept_id`
- `run_threshold_mode(...)`
  Threshold-based cohort creation plus features
- `run_known_outcome_mode(...)`
  Known-outcome feature construction

#### `scripts/train_model.py`

Single-model training script.

Pipeline:

- load CSV with `label`
- optionally drop columns by prefix
- split train/test
- infer numeric vs categorical columns
- build sklearn preprocessing pipeline
- run `GridSearchCV`
- fit the best XGBoost pipeline
- export metrics, predictions, feature importance, trained pipeline, and SHAP outputs

This is the simplest end-to-end training entry point.

#### `scripts/train_multi_model.py`

Main configurable experiment runner.

This is the most flexible training script in the repo.

It:

- loads model definitions from `scripts/models.yml`
- trains each configured estimator under one common train/test split
- supports multiple estimators through dynamic import
- applies per-model hyperparameter search
- exports combined comparison tables across models
- computes SHAP and PDP/ICE artifacts where possible

Important implementation details:

- numeric preprocessing is configurable through the config
- categorical columns are one-hot encoded
- XGBoost and CatBoost can auto-adjust `scale_pos_weight` from class counts
- if model-native feature importance is unavailable, it falls back to coefficients or permutation importance

Outputs include:

- per-model pipelines
- per-model SHAP artifacts
- one combined metrics file
- one combined feature-importance comparison file

#### `scripts/incremental_training.py`

Top-k feature ablation/accumulation experiment runner.

Purpose:

- read one or more datasets
- read a ranked feature list from Excel
- train models repeatedly with top-`k` features
- save metrics across increasing feature counts

This is useful when trying to answer:

- how many top features are enough?
- where does performance plateau?
- which feature ranking transfers across datasets?

Outputs:

- per-run pipelines and SHAP artifacts
- one aggregated CSV/XLSX of metrics over all datasets and all `k`

#### `scripts/nested_cv.py`

Nested cross-validation evaluation script.

Purpose:

- perform unbiased outer-fold evaluation
- perform hyperparameter selection only within inner folds
- summarize robust model performance

This is the script to use when you want less optimistic performance estimates than a single train/test split.

Optional behavior:

- `--apply_preprocess` triggers a duplicate-handling/grouping step on the training fold before fitting

Final output:

- one CSV with mean and standard deviation of AUC, balanced accuracy, and recall across outer folds

#### `scripts/models.yml`

Central experiment configuration.

Current active config:

- one XGBoost model enabled
- RandomForest, LogisticRegression, and CatBoost templates are present but commented out

Global settings include:

- `label_col`
- `test_size`
- `random_state`
- `scoring`
- `cv`
- `n_jobs`
- SHAP sampling settings

### `notebooks/`

This folder is exploratory rather than productionized.

Observed themes:

- `drugon_graph.ipynb`
  Large notebook for causal graph work, LiNGAM experiments, bootstrap stability, effect tables, causal forest work, and some replicated ML experiments.

- `graph_effects.ipynb`
  Graph/effect exploration built around causal discovery tooling.

- `mimic.ipynb`
  MIMIC-oriented data analysis and feature experiments.

The CSV files in `notebooks/` look like exported intermediate results for graph/effect analysis.

### `data/`

- `data/input/`
  Source OMOP databases in DuckDB/SQLite-like form
- `data/output/`
  Saved labeled datasets already generated by the pipeline

The databases are not code, but they are central runtime inputs.

## How The Pipeline Runs

### 1. Build a labeled dataset

Primary script:

`scripts/build_reference_set.py`

Typical flow:

1. Connect to DuckDB or PostgreSQL
2. Define outcome cohort
3. Define negative cohort
4. Optionally discover top concepts
5. Build SQL feature queries for positive and negative cohorts
6. Map concept IDs to readable names
7. Combine cohorts into one labeled dataset
8. Export CSV

### 2. Train model(s)

Options:

- `scripts/train_model.py`
  single XGBoost training run
- `scripts/train_multi_model.py`
  config-driven multi-model experiment
- `scripts/nested_cv.py`
  robust evaluation via nested CV
- `scripts/incremental_training.py`
  top-k feature growth experiments

### 3. Review outputs

Generated artifacts usually include:

- metrics CSV/JSON/XLSX
- feature importance tables
- SHAP summary plots
- SHAP importance tables
- prediction files
- serialized sklearn pipelines via `joblib`

## Typical Commands

### Build threshold-based creatinine dataset

```bash
python scripts/build_reference_set.py \
  --db-path data/input/mimiciv_omop.db \
  --schema main \
  --result-schema result \
  --dialect duckdb \
  --threshold 1.2 \
  --time-window-days 365 \
  --top-exposures 956874 1774470 1776684 \
  --extra-conditions 316139 4064161 4078925 \
  --outdir data/output \
  --basename creatinine_gt_1_2
```

### Train a single XGBoost model

```bash
python scripts/train_model.py \
  --data data/output/creatinine_gt_1_2.csv \
  --outdir model_outputs \
  --basename xgb_creatinine \
  --scoring roc_auc
```

### Train configured models

```bash
python scripts/train_multi_model.py \
  --data data/output/creatinine_gt_1_2.csv \
  --config scripts/models.yml \
  --outdir model_outputs_multi \
  --basename omop_experiment
```

### Run nested CV

```bash
python scripts/nested_cv.py \
  --data data/output/creatinine_gt_1_2.csv \
  --config scripts/models.yml \
  --outdir model_outputs_nestedcv \
  --basename omop_experiment
```

## Important Design Choices

### Outcome anchoring

The code treats feature generation as an index-date problem.

- positives are anchored at outcome dates
- negatives are anchored at deterministic random dates or low-creatinine dates

That design is important because it tries to make the comparison clinically and temporally fair.

### Leakage prevention

There is explicit logic to remove creatinine measurement concept `3016723` from discovered measurement features in threshold mode.

Without this, the label-defining measurement could leak directly into the feature set.

### Dense burden features

The newer feature builders add four dense summary variables:

- `visit_count`
- `distinct_drug_count`
- `distinct_condition_count`
- `days_observed_before_index`

These provide general patient complexity/context beyond sparse one-hot clinical concepts.

### Human-readable feature names

The exported datasets are made interpretable by renaming concept-based columns to OMOP concept names. That is one reason the downstream CSVs are much easier to inspect manually than raw concept-ID outputs.

## Known Weak Spots And Things To Watch

### 1. `features/builder.py` is doing too much

It currently mixes:

- SQL generation
- cohort logic
- concept discovery
- concept-name mapping
- output cleanup

This file is the main dependency hotspot and would benefit from splitting into smaller modules.

### 2. Commented-out legacy code is still in active files

Several large older implementations remain commented out. They are useful for history, but they also make the file harder for a new person to parse quickly.

### 3. Readme vs actual code can drift

The README already documents the pipeline at a high level, but the actual code has evolved beyond it, especially around:

- burden features
- negative anchor construction
- multi-model training
- incremental top-k experiments

The new collaborator should trust the script code first, then use README/HANDOFF as orientation.

### 4. Feature cleaning is partly hard-coded

`build_reference_set.py` currently drops:

- `gender`
- `race`
- `person_id`

before export.

That may be intentional for modeling, but it is a meaningful design choice and should not be changed casually.

### 5. Notebook logic is valuable but not packaged

There is important research logic in notebooks, especially around causal graph analysis, but it has not been converted into reusable modules yet.

## Suggested Starting Point For The Next Person

If someone needs to resume the work quickly, this is the best order:

1. Read `scripts/build_reference_set.py`
2. Read `features/builder.py`
3. Read `scripts/train_multi_model.py`
4. Read `evaluation/lib.py`
5. Review `scripts/models.yml`
6. Inspect an existing dataset in `data/output/`
7. Inspect an existing model-output folder if available
8. Use notebooks only after understanding the scripted pipeline

## What Is Already In The Repository

- source OMOP databases under `data/input/`
- multiple generated creatinine datasets under `data/output/`
- exploratory notebooks for graph and causal analysis
- both `requirements.txt` and `req.txt`

`requirements.txt` appears to reflect the main Python environment for the current repository. `req.txt` looks broader and includes additional causal-analysis dependencies used more heavily in notebooks.

## Practical Resume Checklist

When resuming work, the next collaborator should:

1. Create the environment and install dependencies
2. Confirm which database file is the intended source of truth
3. Re-run `build_reference_set.py` for the target cohort definition
4. Re-run `train_multi_model.py` or `nested_cv.py` depending on whether the goal is experimentation or unbiased evaluation
5. Compare exported feature-importance and SHAP outputs
6. Only then continue notebook-based causal or graph analysis

## Short Summary

This repo is a clinical ML pipeline around OMOP data.

- `features/` builds cohorts and SQL-derived features
- `scripts/` runs dataset creation and training experiments
- `evaluation/` computes metrics and explainability outputs
- `notebooks/` hold exploratory causal and graph-analysis work

If the next person understands `build_reference_set.py`, `builder.py`, and `train_multi_model.py`, they will understand most of the operational workflow of the project.
