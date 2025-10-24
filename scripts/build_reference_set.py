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
    p.add_argument("--discover-n-exposures", type=int, default=10)
    p.add_argument("--discover-n-measurements", type=int, default=10)
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
    return df.drop(columns=["person_id", "condition_start_date", "condition_concept_id"], errors="ignore")


def sample_per_class(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Deduplicate within each class, then sample up to max_per_class from each label, then concat and deduplicate again."""
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column before sampling.")
    df = df.copy()

    # Deduplicate within each class first
    pos = df[df["label"] == 1].drop_duplicates()
    neg = df[df["label"] == 0].drop_duplicates()

    # Sample if necessary
    if len(pos) > max_per_class:
        pos = pos.sample(n=max_per_class, random_state=seed)
    if len(neg) > max_per_class:
        neg = neg.sample(n=max_per_class, random_state=seed)

    # Combine and deduplicate again to ensure no duplicates overall
    combined = pd.concat([pos, neg], ignore_index=True).drop_duplicates()

    return combined.reset_index(drop=True)


def run_threshold_mode(con, threshold: float, time_window_days: int,
                       top_exposures, top_measurements, top_procedures, extra_conditions,source_schema='main',result_schema='main',dialect='duckdb'):
    """Build positives/negatives using creatinine threshold (concept_id=3016723)."""
    # Positive outcome cohort
    df_outcomes = f"""
        SELECT person_id, measurement_date AS condition_start_date
        FROM {source_schema}.measurement
        WHERE measurement_concept_id = 3016723
          AND value_as_number IS NOT NULL
          AND value_as_number > {threshold}
        """
    sql = qualify_tables(df_outcomes, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    df_outcomes = fetch_df(con, sql, dialect=dialect).sort_values(["person_id", "condition_start_date"]).drop_duplicates(subset="person_id", keep="first")
    df_outcomes["condition_concept_id"] = 9999999

    # Negative outcome cohort
    df_neg_outcomes = f"""
        SELECT person_id, measurement_date AS condition_start_date
        FROM {source_schema}.measurement
        WHERE measurement_concept_id = 3016723
          AND value_as_number IS NOT NULL
          AND value_as_number < {threshold}
        """
    sql = qualify_tables(df_neg_outcomes, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    df_neg_outcomes = fetch_df(con,sql,dialect=dialect).sort_values(["person_id", "condition_start_date"]).drop_duplicates(subset="person_id", keep="first")
    df_neg_outcomes["condition_concept_id"] = 9999999
    print(df_neg_outcomes.shape)


    if dialect == "duckdb":
        print("Registering outcome_patients and negative_outcome_patients tables in DuckDB")
        persist_df_duckdb(con, df_outcomes,       result_schema, "outcome_patients")
        persist_df_duckdb(con, df_neg_outcomes,   result_schema, "negative_outcome_patients")
    else:  
        persist_df_postgres(con, df_outcomes,     result_schema, "outcome_patients")
        persist_df_postgres(con, df_neg_outcomes, result_schema, "negative_outcome_patients")


    # Features for positives
    query = build_feature_query_from_concept_ids(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id="predefined",
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect
    )
    
    df_pos = fetch_df(con, query,dialect=dialect)
    df_pos = rename_and_replace(df_pos, con,dialect=dialect,source_schema=source_schema)
    # df_pos = drop_irrelevant_cols(df_pos)
    df_1 = df_pos[df_pos.exposure_furosemide == 1]
    df_0 = df_pos[df_pos.exposure_furosemide == 0]
    df_0 = df_0.drop(columns=['person_id','condition_start_date','condition_concept_id']).drop_duplicates()
    df_1 = df_1.drop(columns=['person_id','condition_start_date','condition_concept_id'])
    df_pos = pd.concat([df_0,df_1])
    df_pos['label']=1
    # df_pos = df_pos.sample(n=1648, random_state=123)
    df_pos.reset_index(inplace=True,drop=True)

    # Features for negatives
    neg_query = build_negative_patient_query_random_window(
        top_exposures=top_exposures,
        top_measurements=top_measurements,
        top_procedures=top_procedures,
        extra_conditions=list(extra_conditions) if extra_conditions else None,
        outcome_id="predefined",
        time_window_days=time_window_days,
        source_schema=source_schema,
        result_schema=result_schema,
        dialect=dialect
    )
    df_neg = fetch_df(con, neg_query,dialect=dialect)
    df_neg = rename_and_replace(df_neg, con,dialect=dialect, source_schema=source_schema)
    df_neg = df_neg.fillna(0)
    print(df_neg.shape)
    df_neg = drop_irrelevant_cols(df_neg)
    df_neg["label"] = 0
    

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
        dialect=dialect
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
        dialect=dialect
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
            dialect=args.dialect
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
            dialect=args.dialect
        )

    combined = pd.concat([df_pos, df_neg], ignore_index=True)
    # combined = combined.drop_duplicates()

    #combined = sample_per_class(combined, max_per_class=args.max_per_class, seed=args.random_seed)

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
