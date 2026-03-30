import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import duckdb

# Allow running from anywhere by adding project root (one level up from this file) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.builder import (
    get_top_concept_ids,
    build_feature_query_from_concept_ids,
    map_all_feature_ids,
    rename_columns_using_concept_names,
    build_negative_patient_query_random_window,
    replace_demographic_ids_with_names,
)

from features.sqltranslate import *


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and save an exposure/outcome dataset from DuckDB.")
    p.add_argument("--db-path", required=True, help="Path to DuckDB database file.")
    p.add_argument("--schema", default="main", help="Schema where OMOP tables are located.")
    p.add_argument("--result-schema", default="main", help="Schema where result tables are located.")
    p.add_argument("--dialect", choices=["duckdb", "postgres", "postgresql"], default="duckdb", help="SQL dialect of the database.")
    p.add_argument("--pg-url", help="SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db")


    # Mode selection
    p.add_argument("--outcome-id", type=int, default=None, help="Known outcome concept_id. If omitted, uses known outcome mode.")
    p.add_argument("--threshold", type=float, default=1.2, help="Creatinine threshold for threshold mode (concept_id=3016723).")

    # Feature window and concept lists
    p.add_argument("--time-window-days", type=int, default=365, help="Lookback window in days for features.")
    p.add_argument("--top-exposures", type=int, nargs="*", default=None, help="Exposure concept IDs (optional).")
    p.add_argument("--top-measurements", type=int, nargs="*", default=None, help="Measurement concept IDs (optional).")
    p.add_argument("--top-procedures", type=int, nargs="*", default=None, help="Procedure concept IDs (optional).")
    p.add_argument("--extra-conditions", type=int, nargs="*", default=[], help="Additional condition concept IDs to include as features.")

    # Automatic discovery of top concepts when outcome-id is known
    p.add_argument("--discover-top-concepts", action="store_true", help="If set with --outcome-id, use get_top_concept_ids().")
    p.add_argument("--discover-lookback-days", type=int, default=365, help="Lookback days for get_top_concept_ids().")
    p.add_argument("--discover-n-exposures",  type=int, default=5)
    p.add_argument("--discover-n-measurements", type=int, default=5)
    p.add_argument("--discover-n-procedures", type=int, default=5)

    # Output and sampling
    p.add_argument("--outdir", default="./outputs", help="Directory to save outputs.")
    p.add_argument("--basename", default="dataset", help="Base filename (extensions will be added).")
    p.add_argument("--save-format", choices=["csv"], default="csv", help="Which file formats to write.")
    p.add_argument("--max-per-class", type=int, default=25000, help="Maximum number of rows to keep per class label (1=pos, 0=neg).")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling.")

    return p.parse_args()


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def rename_and_replace(df: pd.DataFrame, con, dialect, source_schema="main") -> pd.DataFrame:
    cmap = map_all_feature_ids(con=con, df=df,dialect=dialect, source_schema=source_schema)
    df = rename_columns_using_concept_names(df, cmap)
    df = replace_demographic_ids_with_names(df, cmap)
    return df


def drop_irrelevant_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["condition_start_date", "condition_concept_id"], errors="ignore")


def sample_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Deduplicate within each class, then sample up to max_per_class from each label, then concat and deduplicate again."""
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column before sampling.")
    df = df.copy()

    # Deduplicate within each class first
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    # Sample if necessary
    if len(pos) > max_per_class:
        pos = pos.sample(n=max_per_class, random_state=seed)
    if len(neg) > max_per_class:
        neg = neg.sample(n=max_per_class, random_state=seed)

    # Combine and deduplicate again to ensure no duplicates overall
    combined = pd.concat([pos, neg], ignore_index=True)

    return combined.reset_index(drop=True)


def run_threshold_mode(
    con,
    threshold: float,
    time_window_days: int,
    top_exposures,
    top_measurements,
    top_procedures,
    extra_conditions,
    source_schema: str = 'main',
    result_schema: str = 'main',
    dialect: str = 'duckdb',
    discover: bool = False,
    discover_lookback: int = 365,
    n_exp: int = 5,
    n_meas: int = 5,
    n_proc: int = 5,
):
    """
    Build positives/negatives using creatinine threshold (concept_id=3016723).

    Positives  = patients whose creatinine EVER exceeded the threshold;
                 we take their *first* high-creatinine measurement as index.

    Negatives  = patients whose creatinine NEVER exceeded the threshold at any time;
                 we take their *first* low-creatinine measurement as index.

    This ensures:
      - no patient appears in both positive and negative cohorts
      - cleaner separation of label=1 vs label=0
    """

    # -------------------------
    # 1) POSITIVE OUTCOME COHORT
    # -------------------------
    # First high-creatinine measurement per person
    
    sql_pos = f"""
        WITH high_creat AS (
            SELECT
                person_id,
                measurement_date
            FROM {source_schema}.measurement
            WHERE measurement_concept_id = 3016723
              AND value_as_number IS NOT NULL
              AND value_as_number > {threshold}
        )
        SELECT
            person_id,
            MIN(measurement_date) AS condition_start_date
        FROM high_creat
        GROUP BY person_id
    """
    sql = qualify_tables(sql_pos, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)

    df_outcomes = (
        fetch_df(con, sql, dialect=dialect)
        .sort_values(["person_id", "condition_start_date"])
        .drop_duplicates(subset="person_id", keep="first")
    )
    df_outcomes["condition_concept_id"] = 9999999

    # -------------------------
    # 2) NEGATIVE OUTCOME COHORT (CLEAN)
    # -------------------------
    # Patients who NEVER have a high creatinine.
    # Among their low-creatinine measurements, take the first one as index.
    sql_neg = f"""
        WITH high_creat AS (
            SELECT DISTINCT
                person_id
            FROM {source_schema}.measurement
            WHERE measurement_concept_id = 3016723
              AND value_as_number IS NOT NULL
              AND value_as_number > {threshold}
        ),
        low_creat AS (
            SELECT
                person_id,
                measurement_date
            FROM {source_schema}.measurement
            WHERE measurement_concept_id = 3016723
              AND value_as_number IS NOT NULL
              AND value_as_number < {threshold}
        )
        SELECT
            lc.person_id,
            MIN(lc.measurement_date) AS condition_start_date
        FROM low_creat lc
        LEFT JOIN high_creat hc
          ON lc.person_id = hc.person_id
        WHERE hc.person_id IS NULL              -- ensure NEVER high creatinine
        GROUP BY lc.person_id
    """
    sql = qualify_tables(sql_neg, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)

    df_neg_outcomes = (
        fetch_df(con, sql, dialect=dialect)
        .sort_values(["person_id", "condition_start_date"])
        .drop_duplicates(subset="person_id", keep="first")
    )
    df_neg_outcomes["condition_concept_id"] = 9999999

    print("Positive cohort shape (outcome_patients):", df_outcomes.shape)
    print("Negative cohort shape (negative_outcome_patients):", df_neg_outcomes.shape)

    # -------------------------
    # 3) Persist cohorts for downstream feature builders
    # -------------------------
    if dialect == "duckdb":
        print("Registering outcome_patients and negative_outcome_patients tables in DuckDB")
        persist_df_duckdb(con, df_outcomes,     result_schema, "outcome_patients")
        persist_df_duckdb(con, df_neg_outcomes, result_schema, "negative_outcome_patients")
    else:
        print("Registering outcome_patients and negative_outcome_patients tables in Postgres")
        persist_df_postgres(con, df_outcomes,     result_schema, "outcome_patients")
        persist_df_postgres(con, df_neg_outcomes, result_schema, "negative_outcome_patients")

    def _merge_lists(user_list, discovered_list):
        """Merge user-specified and discovered concept IDs, preserving order and removing duplicates."""
        if discovered_list is None:
            return user_list
        if user_list is None:
            user_list = []
        merged = list(dict.fromkeys(list(user_list) + list(discovered_list)))
        return merged

    if discover:
        print("Discovering top concept IDs using get_top_concept_ids()...")
        # Assumes get_top_concept_ids can handle outcome_id="predefined" and
        # uses outcome_patients / negative_outcome_patients under the hood.
        disc_exp, top_measurements, top_procedures = get_top_concept_ids(
            con=con,
            outcome_id="predefined",
            time_window_days=discover_lookback,
            num_exposures=n_exp,
            num_measurements=n_meas,
            num_procedures=n_proc,
            source_schema=source_schema,
            result_schema=result_schema,
            dialect=dialect,
        )

        top_exposures    = _merge_lists(top_exposures,    disc_exp)
        # top_measurements = _merge_lists(top_measurements, disc_meas)
        # top_procedures   = _merge_lists(top_procedures,   disc_proc)
    # -------------------------
    # 4) Build FEATURES for POSITIVES
    # -------------------------
    print(top_exposures, top_measurements, top_procedures)
    print("Building features for positives and negatives...")
    pos_query = build_feature_query_from_concept_ids(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id="predefined",         # uses result_schema.outcome_patients
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect,
    )
    print("Fetching features for positive cohort...")
    print(pos_query)
    df_pos = fetch_df(con, pos_query, dialect=dialect)
    df_pos = rename_and_replace(df_pos, con, dialect=dialect, source_schema=source_schema)
    # If you still want the furosemide re-balancing logic, keep your custom block here.
    # Otherwise, you can keep all positives as-is and just set label=1.
    df_pos = drop_irrelevant_cols(df_pos)

    df_pos["label"] = 1

    # -------------------------
    # 5) Build FEATURES for NEGATIVES
    # -------------------------
    print("Building features for negatives...")
    neg_query = build_negative_patient_query_random_window(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id="predefined",          # uses result_schema.negative_outcome_patients
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect,
        seed=25
    )
    print("Fetching features for negative cohort...")
    df_neg = fetch_df(con, neg_query, dialect=dialect)
    df_neg = rename_and_replace(df_neg, con, dialect=dialect, source_schema=source_schema)
    df_neg = df_neg.fillna(0)
    df_neg = drop_irrelevant_cols(df_neg)
    df_neg["label"] = 0
    print("Finished building features.")

    return df_pos, df_neg

def run_known_outcome_mode(con: duckdb.DuckDBPyConnection, outcome_id: int, time_window_days: int,
                           top_exposures, top_measurements, top_procedures, extra_conditions,
                           discover: bool, discover_lookback: int, n_exp: int, n_meas: int, n_proc: int,source_schema='main',result_schema='main',dialect='duckdb'):
    """Build positives/negatives when outcome concept_id is known."""
    # If requested or if no lists provided, discover top concept ids
    if discover or (top_exposures is None and top_measurements is None and top_procedures is None):
        top_exposures, top_measurements, top_procedures = get_top_concept_ids(
            con, outcome_id, discover_lookback, n_exp, n_meas, n_proc,source_schema, dialect
        )
    print(top_exposures, top_measurements, top_procedures)
    # Positive cohort features
    query = build_feature_query_from_concept_ids(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id=outcome_id,
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect,
    )
    df_pos = fetch_df(con, query,dialect)
    df_pos = rename_and_replace(df_pos, con,dialect=dialect,source_schema=source_schema)
    df_pos = drop_irrelevant_cols(df_pos)
    df_pos["label"] = 1

    # Negative cohort features from random windows
    neg_query = build_negative_patient_query_random_window(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id=outcome_id,
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect,
        seed=42
    )
    df_neg = fetch_df(con, neg_query,dialect)
    df_neg = rename_and_replace(df_neg, con,dialect,source_schema=source_schema)
    df_neg = df_neg.fillna(0)
    df_neg = drop_irrelevant_cols(df_neg)
    df_neg["label"] = 0

    return df_pos, df_neg


def main():
    args = parse_args()
    pd.set_option("display.max_columns", None)
    outdir = ensure_outdir(args.outdir)

    if args.dialect == "duckdb":
        con = duckdb.connect(database=args.db_path)
    else:
        if not args.pg_url:
            raise SystemExit("--pg-url is required when --dialect postgresql")
        con = connect_postgres(args.pg_url)

    if args.outcome_id is None:
        df_pos, df_neg = run_threshold_mode(
            con=con,
            threshold=args.threshold,
            time_window_days=args.time_window_days,
            top_exposures=args.top_exposures,
            top_measurements=args.top_measurements,
            top_procedures=args.top_procedures,
            extra_conditions=args.extra_conditions,
            source_schema=args.schema,
            result_schema=args.result_schema,
            dialect=args.dialect,
            discover=args.discover_top_concepts,
            discover_lookback=args.discover_lookback_days,
            n_exp=args.discover_n_exposures,
            n_meas=args.discover_n_measurements,
            n_proc=args.discover_n_procedures,
        )
    else:
        df_pos, df_neg = run_known_outcome_mode(
            con=con,
            outcome_id=args.outcome_id,
            time_window_days=args.time_window_days,
            top_exposures=args.top_exposures,
            top_measurements=args.top_measurements,
            top_procedures=args.top_procedures,
            extra_conditions=args.extra_conditions,
            discover=args.discover_top_concepts,
            discover_lookback=args.discover_lookback_days,
            n_exp=args.discover_n_exposures,
            n_meas=args.discover_n_measurements,
            n_proc=args.discover_n_procedures,
            source_schema=args.schema,
            result_schema=args.result_schema,
            dialect=args.dialect,
        )

    combined = pd.concat([df_pos, df_neg], ignore_index=True)
    combined.drop(columns=['gender','race','person_id'], inplace=True)
    zero_cols = combined.columns[(combined == 0).all()]
    combined = combined.drop(columns=zero_cols)
    feature_cols = [c for c in combined.columns if c != "label"]
    # df_majority = (
    #     combined.groupby(feature_cols, dropna=False, group_keys=False)
    #   .apply(lambda g: g[g["label"] == g["label"].mode()[0]]))

    # df_filtered_2 = df_filtered_2.drop(columns=['person_id'])


    # combined = combined.drop_duplicates()

    # combined = sample_per_class(combined, max_per_class=args.max_per_class, seed=args.random_seed)

    base = Path(outdir) / args.basename
    if args.save_format in ("csv"):
        csv_path = f"{base}.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
    print(f"Final rows: {len(combined)} (pos={n_pos}, neg={n_neg})")
    with pd.option_context("display.max_columns", 12, "display.width", 200):
        print("\nPreview:\n", combined.head())


if __name__ == "__main__":
    main()
