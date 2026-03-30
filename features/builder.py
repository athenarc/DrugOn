from typing import List,Union,Optional
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from features.sqltranslate import *

def build_exposure_outcome_dataset(
    duckdb_conn,
    exposures: List[int],
    outcomes: List[int],
    include_demographics: bool = True,
    include_drug_info: bool = True,
    include_condition_info: bool = True,
    include_visits: bool = True,
    include_measurements: bool = True,
    include_drug_history: bool = True,
    include_observations: bool = True,
    include_procedures: bool = True
) -> str:
    """
    Build an enriched dataset where each row is a (person, drug exposure, condition outcome) pair,
    including optional demographics, visit, lab, and drug history info.
    Measurements are now joined using both person_id and visit_occurrence_id.
    """

    exposure_ids = ", ".join(str(e) for e in exposures)
    outcome_ids = ", ".join(str(o) for o in outcomes)

    select_fields = [
        "de.person_id",
        "de.drug_exposure_id"
    ]

    if include_drug_info:
        select_fields += [
            "de.drug_concept_id",
            "de.drug_exposure_start_date",
            "de.drug_exposure_end_date",
            "de.days_supply",
            "de.visit_occurrence_id",
            "de.visit_detail_id"
        ]

    if include_condition_info:
        select_fields += [
            "co.condition_concept_id",
            "co.condition_start_date",
            "co.condition_end_date"
        ]

    joins = [
        f"""
        JOIN condition_occurrence co
        ON de.person_id = co.person_id
        AND co.condition_concept_id IN ({outcome_ids})
        """
    ]

    if include_demographics:
        joins.append("JOIN person p ON de.person_id = p.person_id")
        select_fields += [
            "p.gender_concept_id",
            "p.race_concept_id",
            "p.year_of_birth"
        ]

    if include_visits:
        joins.append("""
            LEFT JOIN visit_occurrence v
            ON de.person_id = v.person_id AND de.visit_occurrence_id = v.visit_occurrence_id
        """)
        select_fields += [
            "v.visit_concept_id",
            "v.visit_start_date",
            "v.visit_end_date"
        ]

    if include_measurements:
        joins.append("""
            LEFT JOIN measurement m
            ON de.person_id = m.person_id AND de.visit_occurrence_id = m.visit_occurrence_id
        """)
        select_fields += [
            "m.measurement_concept_id",
            "m.measurement_date",
            "m.value_as_number"
        ]

    if include_drug_history:
        joins.append("LEFT JOIN drug_era d ON de.person_id = d.person_id")
        select_fields += [
            "d.drug_era_start_date",
            "d.drug_era_end_date",
            "d.drug_concept_id AS prior_drug_concept_id",
            "d.drug_exposure_count"
        ]

    if include_observations:
        joins.append("""
            LEFT JOIN observation o
            ON de.person_id = o.person_id AND de.visit_occurrence_id = o.visit_occurrence_id
        """)
        select_fields += [
            "o.observation_concept_id",
            "o.observation_date",
        ]

    if include_procedures:
        joins.append("""
            LEFT JOIN procedure_occurrence pr
            ON de.person_id = pr.person_id AND de.visit_occurrence_id = pr.visit_occurrence_id
        """)
        select_fields += [
            "pr.procedure_concept_id",
            "pr.procedure_date"
        ]

    query = f"""
        SELECT {', '.join(select_fields)}
        FROM drug_exposure de
        {' '.join(joins)}
        WHERE de.drug_concept_id IN ({exposure_ids})
    """

    return query



def engineer_ade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw exposure-outcome dataset and adds engineered features useful for ADE detection.
    """

    # Copy to avoid modifying in place
    df = df.copy()

    # Age at exposure
    if 'year_of_birth' in df.columns and 'drug_exposure_start_date' in df.columns:
        df['drug_exposure_start_date'] = pd.to_datetime(df['drug_exposure_start_date'], errors='coerce')
        df['age_at_exposure'] = df['drug_exposure_start_date'].dt.year - df['year_of_birth']

    # Exposure duration
    if 'drug_exposure_start_date' in df.columns and 'drug_exposure_end_date' in df.columns:
        df['drug_exposure_end_date'] = pd.to_datetime(df['drug_exposure_end_date'], errors='coerce')
        df['exposure_duration_days'] = (df['drug_exposure_end_date'] - df['drug_exposure_start_date']).dt.days

    # Days to outcome
    if 'condition_start_date' in df.columns and 'drug_exposure_start_date' in df.columns:
        df['condition_start_date'] = pd.to_datetime(df['condition_start_date'], errors='coerce')
        df['days_to_outcome'] = (df['condition_start_date'] - df['drug_exposure_start_date']).dt.days

    # Outcome occurred after exposure
    if 'days_to_outcome' in df.columns:
        df['outcome_after_exposure'] = df['days_to_outcome'] >= 0

    # Visit length (if available)
    if 'visit_start_date' in df.columns and 'visit_end_date' in df.columns:
        df['visit_start_date'] = pd.to_datetime(df['visit_start_date'], errors='coerce')
        df['visit_end_date'] = pd.to_datetime(df['visit_end_date'], errors='coerce')
        df['visit_length'] = (df['visit_end_date'] - df['visit_start_date']).dt.days

    # Flag chronic drug use
    if 'drug_exposure_count' in df.columns:
        df['chronic_use_flag'] = df['drug_exposure_count'] > 3

    # Flag outcome concept match (binary label placeholder)
    if 'condition_concept_id' in df.columns:
        df['had_outcome'] = df['condition_concept_id'].notnull()

    return df


def map_concept_ids_to_names(df: pd.DataFrame, duckdb_conn,dialect
) -> pd.DataFrame:
    """
    Takes a DataFrame with OMOP concept_id columns and maps them to human-readable concept_name using the concept table.
    Includes mapping for demographics (gender, race) and standard clinical domains.
    """
    # List of concept_id columns to map
    concept_columns = [
        'drug_concept_id',
        'condition_concept_id',
        'measurement_concept_id',
        'visit_concept_id',
        'prior_drug_concept_id',
        'gender_concept_id',
        'race_concept_id',
        'observation_concept_id',
        'procedure_concept_id'
    ]

    df = df.copy()

    for col in concept_columns:
        if col in df.columns:
            unique_ids = df[col].dropna().unique().tolist()
            if not unique_ids:
                continue
            ids_str = ", ".join(map(str, unique_ids))

            query = f"""
                SELECT concept_id, concept_name
                FROM concept
                WHERE concept_id IN ({ids_str})
            """
            concept_df = fetch_df(duckdb_conn, query, dialect=dialect)
            concept_df.rename(columns={
                'concept_id': col,
                'concept_name': f"{col}_name"
            }, inplace=True)

            df = df.merge(concept_df, how='left', on=col)

    return df

def get_top_concept_ids(
    con,
    outcome_id: Union[int, str],
    time_window_days: int = 365,
    num_exposures: int = 10,
    num_measurements: int = 10,
    num_procedures: int = 5,
    source_schema: str = 'main',
    result_schema: str = 'main',
    dialect: str = 'duckdb',
):
    """
    Discover top exposures/measurements/procedures associated with an outcome.

    - If outcome_id is an INT  -> use condition_occurrence (known-outcome mode).
    - If outcome_id == 'predefined' -> use result_schema.outcome_patients
      (threshold mode: cohort already built in run_threshold_mode).

    In threshold mode we also DROP creatinine (3016723) from discovered
    measurements to avoid label leakage.
    """

    # -------------------------
    # 0) Build outcome_patients CTE depending on mode
    # -------------------------
    if outcome_id == "predefined":
        # Threshold mode: outcome_patients already persisted
        outcome_cte = f"""
        outcome_patients AS (
            SELECT person_id, condition_start_date
            FROM {result_schema}.outcome_patients
        )
        """
    else:
        # Known outcome mode: pull directly from condition_occurrence
        outcome_cte = f"""
        outcome_patients AS (
            SELECT person_id, condition_start_date
            FROM {source_schema}.condition_occurrence
            WHERE condition_concept_id = {outcome_id}
        )
        """

    # -------------------------
    # 1) Top EXPOSURES
    # -------------------------
    print("Getting top exposures...")
    top_exposures_sql = f"""
        WITH
        {outcome_cte}
        SELECT d.drug_concept_id, COUNT(*) AS freq
        FROM {source_schema}.drug_era d
        JOIN outcome_patients o ON d.person_id = o.person_id
        WHERE d.drug_era_end_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
          AND d.drug_era_start_date <  o.condition_start_date
        GROUP BY d.drug_concept_id
        ORDER BY freq DESC
        LIMIT {num_exposures}
    """
    sql = qualify_tables(top_exposures_sql, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    top_exposures = fetch_df(con, sql, dialect=dialect)["drug_concept_id"].tolist()

    # -------------------------
    # 2) Top MEASUREMENTS
    # -------------------------
    print("Getting top measurements...")
    top_measurements_sql = f"""
        WITH
        {outcome_cte}
        SELECT m.measurement_concept_id, COUNT(*) AS freq
        FROM {source_schema}.measurement m
        JOIN outcome_patients o ON m.person_id = o.person_id
        WHERE m.measurement_date BETWEEN (o.condition_start_date - INTERVAL {time_window_days} DAY)
                                     AND o.condition_start_date
        GROUP BY m.measurement_concept_id
        ORDER BY freq DESC
        LIMIT {num_measurements}
    """
    sql = qualify_tables(top_measurements_sql, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    top_measurements = fetch_df(con, sql, dialect=dialect)["measurement_concept_id"].tolist()

    # In threshold mode, drop creatinine from discovered measurements to avoid leakage
    CREATININE_CONCEPT_ID = 3016723
    if outcome_id == "predefined":
        top_measurements = [cid for cid in top_measurements if cid != CREATININE_CONCEPT_ID]

    # -------------------------
    # 3) Top PROCEDURES
    # -------------------------
    print("Getting top procedures...")
    top_procedures_sql = f"""
        WITH
        {outcome_cte}
        SELECT p.procedure_concept_id, COUNT(*) AS freq
        FROM {source_schema}.procedure_occurrence p
        JOIN outcome_patients o ON p.person_id = o.person_id
        WHERE p.procedure_date BETWEEN (o.condition_start_date - INTERVAL {time_window_days} DAY)
                                   AND o.condition_start_date
        GROUP BY p.procedure_concept_id
        ORDER BY freq DESC
        LIMIT {num_procedures}
    """
    sql = qualify_tables(top_procedures_sql, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    top_procedures = fetch_df(con, sql, dialect=dialect)["procedure_concept_id"].tolist()
    print("Discovered top concept IDs.")

    return top_exposures, top_measurements, top_procedures

# def build_feature_query_from_concept_ids(
#     top_exposures: Optional[List[int]] = None,
#     top_measurements: Optional[List[int]] = None,
#     top_procedures: Optional[List[int]] = None,
#     extra_conditions: Optional[List[int]] = None,
#     outcome_id: Union[int, str] = 'predefined',
#     time_window_days: int = 365,
#     source_schema: str = 'main',
#     result_schema: str = 'main',
#     dialect: str = 'duckdb',
# ) -> str:
#     """
#     Generate a SQL query to extract patient-level features for a specific outcome.
#     Optional: exposures, measurements, procedures, and extra conditions.
#     """

#     # ---- 1) Optional: remove creatinine from features in threshold mode ----
#     # Adjust 3016723 to your actual creatinine concept_id if different.
#     CREATININE_CONCEPT_ID = 3016723
#     if outcome_id == 'predefined' and top_measurements:
#         top_measurements = [
#             cid for cid in top_measurements
#             if cid != CREATININE_CONCEPT_ID
#         ]

#     ctes = []

#     # Outcome CTE
#     if outcome_id == 'predefined':
#         outcome_cte = f"""
#         outcome_patients AS (
#             SELECT person_id, condition_start_date, condition_concept_id
#             FROM {result_schema}.outcome_patients
#         )
#         """
#     else:
#         outcome_cte = f"""
#         outcome_patients AS (
#             SELECT person_id, condition_start_date, condition_concept_id
#             FROM {source_schema}.condition_occurrence
#             WHERE condition_concept_id = {outcome_id}
#         )
#         """
#     ctes.append(outcome_cte)

#     feature_selects = []

#     # Exposure CTE and selects
#     if top_exposures:
#         ctes.append(f"""
#         exposure_features AS (
#             SELECT o.person_id, o.condition_start_date, d.drug_concept_id
#             FROM outcome_patients o
#             JOIN {source_schema}.drug_era d ON d.person_id = o.person_id
#             WHERE d.drug_era_end_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
#               AND d.drug_era_start_date <  o.condition_start_date              -- STRICT <
#               AND d.drug_concept_id IN ({','.join(map(str, top_exposures))})
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN ef.drug_concept_id = {cid} THEN 1 ELSE 0 END) AS exposure_{cid}"
#             for cid in top_exposures
#         ])
    
#     # Measurement CTE and selects
#     if top_measurements:
#         ctes.append(f"""
#             measurement_features AS (
#                 SELECT
#                     o.person_id,
#                     o.condition_start_date,
#                     m.measurement_concept_id,
#                     m.value_as_number,
#                     ROW_NUMBER() OVER (
#                         PARTITION BY o.person_id, m.measurement_concept_id
#                         ORDER BY m.measurement_date DESC   -- latest measurement before event
#                     ) AS rn
#                 FROM outcome_patients o
#                 JOIN {source_schema}.measurement m
#                 ON m.person_id = o.person_id
#                 WHERE m.measurement_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
#                 AND m.measurement_date <  o.condition_start_date
#                 AND m.measurement_concept_id IN ({','.join(map(str, top_measurements))})
#             )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN mf.measurement_concept_id = {cid} AND mf.rn = 1 THEN mf.value_as_number ELSE NULL END) AS measurement_{cid}"
#             for cid in top_measurements
#         ])

#     # Procedure CTE and selects
#     if top_procedures:
#         ctes.append(f"""
#         procedure_features AS (
#             SELECT o.person_id, o.condition_start_date, p.procedure_concept_id
#             FROM {source_schema}.procedure_occurrence p
#             JOIN outcome_patients o ON p.person_id = o.person_id
#             WHERE p.procedure_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
#               AND p.procedure_date <  o.condition_start_date               -- STRICT <
#               AND p.procedure_concept_id IN ({','.join(map(str, top_procedures))})
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN pf.procedure_concept_id = {cid} THEN 1 ELSE 0 END) AS procedure_{cid}"
#             for cid in top_procedures
#         ])

#     # Extra condition CTE and selects
#     if extra_conditions:
#         ctes.append(f"""
#         condition_features AS (
#             SELECT o.person_id, o.condition_start_date, c.condition_concept_id
#             FROM {source_schema}.condition_occurrence c
#             JOIN outcome_patients o ON c.person_id = o.person_id
#             WHERE c.condition_start_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
#               AND c.condition_start_date <  o.condition_start_date          -- STRICT <
#               AND c.condition_concept_id IN ({','.join(map(str, extra_conditions))})
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN cf.condition_concept_id = {cid} THEN 1 ELSE 0 END) AS condition_{cid}"
#             for cid in extra_conditions
#         ])

#     # Final features CTE
#     joins = []
#     if top_exposures:
#         joins.append("LEFT JOIN exposure_features ef ON o.person_id = ef.person_id AND o.condition_start_date = ef.condition_start_date")
#     if top_measurements:
#         joins.append("LEFT JOIN measurement_features mf ON o.person_id = mf.person_id AND o.condition_start_date = mf.condition_start_date")
#     if top_procedures:
#         joins.append("LEFT JOIN procedure_features pf ON o.person_id = pf.person_id AND o.condition_start_date = pf.condition_start_date")
#     if extra_conditions:
#         joins.append("LEFT JOIN condition_features cf ON o.person_id = cf.person_id AND o.condition_start_date = cf.condition_start_date")

#     ctes.append(f"""
#     final_features AS (
#         SELECT
#             o.person_id,
#             o.condition_start_date,
#             o.condition_concept_id,
#             p.gender_concept_id,
#             p.race_concept_id,
#             YEAR(o.condition_start_date) - p.year_of_birth AS age_at_outcome
#             {',' if feature_selects else ''}{', '.join(feature_selects)}
#         FROM outcome_patients o
#         JOIN person p ON o.person_id = p.person_id
#         {' '.join(joins)}
#         GROUP BY o.person_id, o.condition_start_date,
#                  o.condition_concept_id, p.gender_concept_id, p.race_concept_id, p.year_of_birth
#     )
#     """)
#     with_clause = ',\n'.join(ctes)
#     query = f"""
#     WITH
#     {with_clause}
#     SELECT * FROM final_features;
#     """
#     sql = qualify_tables(query, source_schema=source_schema)
#     sql = translate_sql(sql, dialect=dialect)
#     return sql

# the above commented out function is the old version. 
# The new version below adds 4 dense burden features and ensures all joins are on both person_id 
# and condition_start_date to prevent leakage.
def build_feature_query_from_concept_ids(
    top_exposures: Optional[List[int]] = None,
    top_measurements: Optional[List[int]] = None,
    top_procedures: Optional[List[int]] = None,
    extra_conditions: Optional[List[int]] = None,
    outcome_id: Union[int, str] = "predefined",  # "predefined" or an int
    time_window_days: int = 365,
    source_schema: str = "main",
    result_schema: str = "main",
    dialect: str = "duckdb",
) -> str:
    """
    POSITIVES (and known-outcome positives):
    - Anchors on outcome_patients.condition_start_date (index date).
    - Builds binary features for selected exposures/procedures/conditions.
    - Builds continuous features for selected measurements (latest value pre-index).
    - Adds 4 dense burden features:
        visit_count, distinct_drug_count, distinct_condition_count, days_observed_before_index
    """

    # Prevent leakage in threshold mode (predefined outcome built from creatinine)
    CREATININE_CONCEPT_ID = 3016723
    if outcome_id == "predefined" and top_measurements:
        top_measurements = [cid for cid in top_measurements if cid != CREATININE_CONCEPT_ID]

    ctes: List[str] = []

    # Outcome CTE
    if outcome_id == "predefined":
        ctes.append(f"""
        outcome_patients AS (
            SELECT person_id, condition_start_date, condition_concept_id
            FROM {result_schema}.outcome_patients
        )
        """)
    else:
        ctes.append(f"""
        outcome_patients AS (
            SELECT
                person_id,
                MIN(condition_start_date) AS condition_start_date,
                {outcome_id} AS condition_concept_id
            FROM main.condition_occurrence
            WHERE condition_concept_id = {outcome_id}
            GROUP BY person_id
            )
        """)

    # 4 dense burden features
    ctes.append(f"""
    burden_features AS (
        SELECT
            o.person_id,
            o.condition_start_date,
            COUNT(DISTINCT v.visit_occurrence_id) AS visit_count,
            COUNT(DISTINCT d.drug_concept_id) AS distinct_drug_count,
            COUNT(DISTINCT c.condition_concept_id) AS distinct_condition_count
        FROM outcome_patients o
        LEFT JOIN {source_schema}.visit_occurrence v
          ON v.person_id = o.person_id
         AND v.visit_start_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
         AND v.visit_start_date <  o.condition_start_date
        LEFT JOIN {source_schema}.drug_era d
          ON d.person_id = o.person_id
         AND d.drug_era_end_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
         AND d.drug_era_start_date <  o.condition_start_date
        LEFT JOIN {source_schema}.condition_occurrence c
          ON c.person_id = o.person_id
         AND c.condition_start_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
         AND c.condition_start_date <  o.condition_start_date
        GROUP BY o.person_id, o.condition_start_date
    )
    """)

    ctes.append(f"""
    observation_depth AS (
        SELECT
            o.person_id,
            o.condition_start_date,
            MAX(DATE_DIFF('day', op.observation_period_start_date, o.condition_start_date)) AS days_observed_before_index
        FROM outcome_patients o
        LEFT JOIN {source_schema}.observation_period op
          ON op.person_id = o.person_id
         AND o.condition_start_date BETWEEN op.observation_period_start_date AND op.observation_period_end_date
        GROUP BY o.person_id, o.condition_start_date
    )
    """)

    feature_selects: List[str] = []
    joins: List[str] = []

    # Exposure features (binary)
    if top_exposures:
        ctes.append(f"""
        exposure_features AS (
            SELECT o.person_id, o.condition_start_date, d.drug_concept_id
            FROM outcome_patients o
            JOIN {source_schema}.drug_era d
              ON d.person_id = o.person_id
            WHERE d.drug_era_end_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
              AND d.drug_era_start_date <  o.condition_start_date
              AND d.drug_concept_id IN ({",".join(map(str, top_exposures))})
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN ef.drug_concept_id = {cid} THEN 1 ELSE 0 END) AS exposure_{cid}"
            for cid in top_exposures
        ])
        joins.append(
            "LEFT JOIN exposure_features ef "
            "ON o.person_id = ef.person_id AND o.condition_start_date = ef.condition_start_date"
        )

    # Measurement features (continuous: latest value pre-index)
    if top_measurements:
        ctes.append(f"""
        measurement_features AS (
            SELECT
                o.person_id,
                o.condition_start_date,
                m.measurement_concept_id,
                m.value_as_number,
                ROW_NUMBER() OVER (
                    PARTITION BY o.person_id, m.measurement_concept_id
                    ORDER BY m.measurement_date DESC
                ) AS rn
            FROM outcome_patients o
            JOIN {source_schema}.measurement m
              ON m.person_id = o.person_id
            WHERE m.measurement_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
              AND m.measurement_date <  o.condition_start_date
              AND m.measurement_concept_id IN ({",".join(map(str, top_measurements))})
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN mf.measurement_concept_id = {cid} AND mf.rn = 1 "
            f"THEN mf.value_as_number ELSE NULL END) AS measurement_{cid}"
            for cid in top_measurements
        ])
        joins.append(
            "LEFT JOIN measurement_features mf "
            "ON o.person_id = mf.person_id AND o.condition_start_date = mf.condition_start_date"
        )

    # Procedure features (binary)
    if top_procedures:
        ctes.append(f"""
        procedure_features AS (
            SELECT o.person_id, o.condition_start_date, p.procedure_concept_id
            FROM outcome_patients o
            JOIN {source_schema}.procedure_occurrence p
              ON p.person_id = o.person_id
            WHERE p.procedure_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
              AND p.procedure_date <  o.condition_start_date
              AND p.procedure_concept_id IN ({",".join(map(str, top_procedures))})
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN pf.procedure_concept_id = {cid} THEN 1 ELSE 0 END) AS procedure_{cid}"
            for cid in top_procedures
        ])
        joins.append(
            "LEFT JOIN procedure_features pf "
            "ON o.person_id = pf.person_id AND o.condition_start_date = pf.condition_start_date"
        )

    # Extra condition features (binary)
    if extra_conditions:
        ctes.append(f"""
        condition_features AS (
            SELECT o.person_id, o.condition_start_date, c.condition_concept_id
            FROM outcome_patients o
            JOIN {source_schema}.condition_occurrence c
              ON c.person_id = o.person_id
            WHERE c.condition_start_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
              AND c.condition_start_date <  o.condition_start_date
              AND c.condition_concept_id IN ({",".join(map(str, extra_conditions))})
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN cf.condition_concept_id = {cid} THEN 1 ELSE 0 END) AS condition_{cid}"
            for cid in extra_conditions
        ])
        joins.append(
            "LEFT JOIN condition_features cf "
            "ON o.person_id = cf.person_id AND o.condition_start_date = cf.condition_start_date"
        )

    # Final
    ctes.append(f"""
    final_features AS (
        SELECT
            o.person_id,
            o.condition_start_date,
            o.condition_concept_id,
            p.gender_concept_id,
            p.race_concept_id,
            YEAR(o.condition_start_date) - p.year_of_birth AS age_at_outcome,

            COALESCE(b.visit_count, 0) AS visit_count,
            COALESCE(b.distinct_drug_count, 0) AS distinct_drug_count,
            COALESCE(b.distinct_condition_count, 0) AS distinct_condition_count,
            COALESCE(od.days_observed_before_index, 0) AS days_observed_before_index
            {"," if feature_selects else ""}{", ".join(feature_selects)}

        FROM outcome_patients o
        JOIN {source_schema}.person p
          ON p.person_id = o.person_id

        LEFT JOIN burden_features b
          ON o.person_id = b.person_id AND o.condition_start_date = b.condition_start_date
        LEFT JOIN observation_depth od
          ON o.person_id = od.person_id AND o.condition_start_date = od.condition_start_date

        {" ".join(joins)}
        GROUP BY
            o.person_id, o.condition_start_date, o.condition_concept_id,
            p.gender_concept_id, p.race_concept_id, p.year_of_birth,
            b.visit_count, b.distinct_drug_count, b.distinct_condition_count,
            od.days_observed_before_index
    )
    """)

    query = f"WITH {', '.join(ctes)} SELECT * FROM final_features;"
    sql = qualify_tables(query, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    return sql

# Helper function to map concept_ids to concept names from the concept table
def map_concept_ids_to_names2(con, concept_ids: List[int],dialect, source_schema = "main") -> dict:
    placeholders = ', '.join(str(cid) for cid in concept_ids)
    query = f"""
        SELECT concept_id, concept_name
        FROM {source_schema}.concept
        WHERE concept_id IN ({placeholders})
    """
    df = fetch_df(con, query,dialect)
    return dict(zip(df['concept_id'], df['concept_name']))

# Example usage to map exposure and demographic concept IDs
def map_all_feature_ids(con, df,dialect='duckdb', source_schema = "main"):
    # Extract concept_ids from exposure/measurement/procedure column names
    concept_ids = set()
    for col in df.columns:
        if any(prefix in col for prefix in ['exposure_', 'measurement_', 'procedure_','condition_']):
            try:
                cid = int(col.split('_')[-1])
                concept_ids.add(cid)
            except:
                pass

    # Also map gender and race ids
    gender_ids = df['gender_concept_id'].dropna().unique().tolist()
    race_ids = df['race_concept_id'].dropna().unique().tolist()

    concept_ids.update(gender_ids)
    concept_ids.update(race_ids)

    # Fetch names from concept table
    return map_concept_ids_to_names2(con, list(concept_ids),dialect, source_schema)


def rename_columns_using_concept_names(df, concept_map: dict) -> pd.DataFrame:
    new_columns = {}
    for col in df.columns:
        if any(prefix in col for prefix in ['exposure_', 'measurement_', 'procedure_','condition_']):
            try:
                concept_id = int(col.split('_')[-1])
                name = concept_map.get(concept_id, f"concept_{concept_id}")
                prefix = col.split('_')[0]
                new_columns[col] = f"{prefix}_{name.replace(' ', '_')}"
            except:
                new_columns[col] = col
        else:
            new_columns[col] = col
    return df.rename(columns=new_columns)


def replace_demographic_ids_with_names(df: pd.DataFrame, concept_map: dict) -> pd.DataFrame:
    """
    Replaces gender_concept_id and race_concept_id columns with human-readable names using the concept_map.
    Inserts new columns 'gender' and 'race' as second and third columns in the dataframe.
    """
    df = df.copy()
    
    gender = df['gender_concept_id'].map(concept_map) if 'gender_concept_id' in df.columns else None
    race = df['race_concept_id'].map(concept_map) if 'race_concept_id' in df.columns else None

    if gender is not None:
        df = df.drop(columns=['gender_concept_id'])
    if race is not None:
        df = df.drop(columns=['race_concept_id'])

    # Insert into second and third positions
    insert_at = 1
    if gender is not None:
        df.insert(loc=insert_at, column='gender', value=gender)
        insert_at += 1
    if race is not None:
        df.insert(loc=insert_at, column='race', value=race)

    return df


def get_top_concepts_from_predefined_outcomes(
    con,
    num_exposures: int = 10,
    num_measurements: int = 10,
    num_procedures: int = 5,
    time_window_days: int = 365
):
    """
    Extract top concept_ids from the predefined `outcome_patients` table for:
    - drug exposures (from drug_era)
    - measurements
    - procedures
    Returns: (top_exposures, top_measurements, top_procedures)
    """

    top_exposures = con.execute(f"""
        SELECT d.drug_concept_id, COUNT(*) as freq
        FROM drug_era d
        JOIN outcome_patients o ON d.person_id = o.person_id
        WHERE d.drug_era_end_date >= (o.condition_start_date - INTERVAL {time_window_days} DAY)
          AND d.drug_era_start_date < o.condition_start_date
        GROUP BY d.drug_concept_id
        ORDER BY freq DESC
        LIMIT {num_exposures}
    """).fetchdf()['drug_concept_id'].tolist()

    top_measurements = con.execute(f"""
        SELECT m.measurement_concept_id, COUNT(*) as freq
        FROM measurement m
        JOIN outcome_patients o ON m.person_id = o.person_id
        WHERE m.measurement_date BETWEEN (o.condition_start_date - INTERVAL {time_window_days} DAY)
                                     AND o.condition_start_date
        GROUP BY m.measurement_concept_id
        ORDER BY freq DESC
        LIMIT {num_measurements}
    """).fetchdf()['measurement_concept_id'].tolist()

    top_procedures = con.execute(f"""
        SELECT p.procedure_concept_id, COUNT(*) as freq
        FROM procedure_occurrence p
        JOIN outcome_patients o ON p.person_id = o.person_id
        WHERE p.procedure_date BETWEEN (o.condition_start_date - INTERVAL {time_window_days} DAY)
                                   AND o.condition_start_date
        GROUP BY p.procedure_concept_id
        ORDER BY freq DESC
        LIMIT {num_procedures}
    """).fetchdf()['procedure_concept_id'].tolist()

    return top_exposures, top_measurements, top_procedures


# def build_negative_patient_query_random_window(
#     top_exposures: Optional[List[int]] = None,
#     top_measurements: Optional[List[int]] = None,
#     top_procedures: Optional[List[int]] = None,
#     extra_conditions: Optional[List[int]] = None,
#     outcome_id: Union[int, str] = -1,  # 'predefined' or an int
#     time_window_days: int = 365,
#     source_schema: str = 'main',
#     result_schema: str = 'main',
#     dialect: str = 'duckdb',
#     seed: int = 42,
# ) -> str:
#     """
#     Build a SQL query to extract negative patient examples with a
#     deterministic pseudo-random index date per person.

#     The index_date is stable across runs for the same person_id + seed.
#     """

#     # Optionally remove creatinine as a measurement feature in threshold mode
#     CREATININE_CONCEPT_ID = 3016723
#     if outcome_id == 'predefined' and top_measurements:
#         top_measurements = [cid for cid in top_measurements if cid != CREATININE_CONCEPT_ID]

#     feature_ctes: List[str] = []
#     feature_selects: List[str] = []
#     joins: List[str] = []

#     # Exposure features
#     if top_exposures:
#         feature_ctes.append(f"""
#         exposure_features AS (
#             SELECT d.person_id, d.drug_concept_id
#             FROM {source_schema}.drug_era d
#             JOIN random_anchors n ON d.person_id = n.person_id
#             WHERE d.drug_concept_id IN ({','.join(map(str, top_exposures))})
#               AND d.drug_era_end_date >= n.index_date - INTERVAL {time_window_days} DAY
#               AND d.drug_era_start_date <  n.index_date
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN d.drug_concept_id = {cid} THEN 1 ELSE 0 END) AS exposure_{cid}"
#             for cid in top_exposures
#         ])
#         joins.append("LEFT JOIN exposure_features d ON n.person_id = d.person_id")

#     # Measurement features
#     if top_measurements:
#         feature_ctes.append(f"""
#         measurement_features AS (
#             SELECT
#                 m.person_id,
#                 m.measurement_concept_id,
#                 m.value_as_number,
#                 n.index_date,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY m.person_id, m.measurement_concept_id
#                     ORDER BY m.measurement_date DESC
#                 ) AS rn
#             FROM {source_schema}.measurement m
#             JOIN random_anchors n ON m.person_id = n.person_id
#             WHERE m.measurement_concept_id IN ({','.join(map(str, top_measurements))})
#               AND m.measurement_date >= n.index_date - INTERVAL {time_window_days} DAY
#               AND m.measurement_date <  n.index_date
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN m.measurement_concept_id = {cid} AND m.rn = 1 "
#             f"THEN m.value_as_number ELSE NULL END) AS measurement_{cid}"
#             for cid in top_measurements
#         ])
#         joins.append("LEFT JOIN measurement_features m ON n.person_id = m.person_id")

#     # Procedure features
#     if top_procedures:
#         feature_ctes.append(f"""
#         procedure_features AS (
#             SELECT proc.person_id, proc.procedure_concept_id
#             FROM {source_schema}.procedure_occurrence proc
#             JOIN random_anchors n ON proc.person_id = n.person_id
#             WHERE proc.procedure_concept_id IN ({','.join(map(str, top_procedures))})
#               AND proc.procedure_date >= n.index_date - INTERVAL {time_window_days} DAY
#               AND proc.procedure_date <  n.index_date
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN proc.procedure_concept_id = {cid} THEN 1 ELSE 0 END) AS procedure_{cid}"
#             for cid in top_procedures
#         ])
#         joins.append("LEFT JOIN procedure_features proc ON n.person_id = proc.person_id")

#     # Extra condition features
#     if extra_conditions:
#         feature_ctes.append(f"""
#         condition_features AS (
#             SELECT c.person_id, c.condition_concept_id
#             FROM {source_schema}.condition_occurrence c
#             JOIN random_anchors n ON c.person_id = n.person_id
#             WHERE c.condition_concept_id IN ({','.join(map(str, extra_conditions))})
#               AND c.condition_start_date >= n.index_date - INTERVAL {time_window_days} DAY
#               AND c.condition_start_date <  n.index_date
#         )
#         """)
#         feature_selects.extend([
#             f"MAX(CASE WHEN cf.condition_concept_id = {cid} THEN 1 ELSE 0 END) AS condition_{cid}"
#             for cid in extra_conditions
#         ])
#         joins.append("LEFT JOIN condition_features cf ON n.person_id = cf.person_id")

#     # Base select clause
#     select_clause = f"""
#         n.person_id,
#         n.index_date AS condition_start_date,
#         0 AS condition_concept_id,
#         p.gender_concept_id,
#         p.race_concept_id,
#         YEAR(n.index_date) - p.year_of_birth AS age_at_outcome
#         {',' if feature_selects else ''}{', '.join(feature_selects)}
#     """

#     # Deterministic pseudo-random fraction in [0,1) without integer overflow:
#     # hash( CAST(person_id AS VARCHAR) || '_' || CAST(seed AS VARCHAR) )
#     frac_expr = f"""
#       (
#         (ABS(hash(CAST(person_id AS VARCHAR) || '_' || CAST({seed} AS VARCHAR))) % 1000000)
#         / 1000000.0
#       )
#     """

#     # Base cohort and deterministic anchor dates
#     if outcome_id == 'predefined':
#         base_ctes = f"""
#         negative_cohort AS (
#             SELECT person_id,
#                    observation_period_start_date,
#                    observation_period_end_date,
#                    DATE_DIFF('day', observation_period_start_date, observation_period_end_date) AS duration_days
#             FROM {source_schema}.observation_period
#             WHERE person_id IN (SELECT person_id FROM {result_schema}.negative_outcome_patients)
#               AND DATE_DIFF('day', observation_period_start_date, observation_period_end_date) >= {time_window_days}
#         ),
#         random_anchors AS (
#             SELECT *,
#                    observation_period_start_date
#                    + CAST(
#                        FLOOR(
#                          {frac_expr} * GREATEST(duration_days - {time_window_days}, 0)
#                        ) AS INTEGER
#                      ) * INTERVAL 1 DAY
#                    AS index_date
#             FROM negative_cohort
#         )
#         """
#     else:
#         base_ctes = f"""
#         eligible_patients AS (
#             SELECT person_id,
#                    observation_period_start_date,
#                    observation_period_end_date,
#                    DATE_DIFF('day', observation_period_start_date, observation_period_end_date) AS duration_days
#             FROM {source_schema}.observation_period
#             WHERE DATE_DIFF('day', observation_period_start_date, observation_period_end_date) >= {time_window_days}
#         ),
#         negative_cohort AS (
#             SELECT e.*
#             FROM eligible_patients e
#             LEFT JOIN {source_schema}.condition_occurrence c
#               ON e.person_id = c.person_id AND c.condition_concept_id = {outcome_id}
#             WHERE c.person_id IS NULL
#         ),
#         random_anchors AS (
#             SELECT *,
#                    observation_period_start_date
#                    + CAST(
#                        FLOOR(
#                          {frac_expr} * GREATEST(duration_days - {time_window_days}, 0)
#                        ) AS INTEGER
#                      ) * INTERVAL 1 DAY
#                    AS index_date
#             FROM negative_cohort
#         )
#         """

#     # Assemble WITH clause
#     with_clause = base_ctes
#     if feature_ctes:
#         with_clause += ",\n" + ",\n".join(feature_ctes)

#     with_clause += f""",
#     final_features AS (
#         SELECT
#             {select_clause}
#         FROM random_anchors n
#         JOIN {source_schema}.person p ON n.person_id = p.person_id
#         {' '.join(joins)}
#         GROUP BY n.person_id, n.index_date, p.gender_concept_id, p.race_concept_id, p.year_of_birth
#     )
#     """

#     query = f"""
#     WITH
#     {with_clause}
#     SELECT *, 0 AS label FROM final_features;
#     """

#     sql = qualify_tables(query, source_schema=source_schema)
#     sql = translate_sql(sql, dialect=dialect)
#     return sql

# the above commented out function is the old version. 
# The new version below adds 4 dense burden features and ensures all joins are on both person_id 
# and condition_start_date to prevent leakage.
def build_negative_patient_query_random_window(
    top_exposures: Optional[List[int]] = None,
    top_measurements: Optional[List[int]] = None,
    top_procedures: Optional[List[int]] = None,
    extra_conditions: Optional[List[int]] = None,
    outcome_id: Union[int, str] = -1,  # "predefined" or an int
    time_window_days: int = 365,
    source_schema: str = "main",
    result_schema: str = "main",
    dialect: str = "duckdb",
    seed: int = 42,
) -> str:
    """
    NEGATIVES:
    - Uses a deterministic pseudo-random anchor date per person (index_date).
    - In threshold mode (outcome_id='predefined'), negative persons come from result_schema.negative_outcome_patients.
    - Adds the same 4 dense burden features:
        visit_count, distinct_drug_count, distinct_condition_count, days_observed_before_index
    """

    # Prevent leakage in threshold mode (predefined outcome built from creatinine)
    CREATININE_CONCEPT_ID = 3016723
    if outcome_id == "predefined" and top_measurements:
        top_measurements = [cid for cid in top_measurements if cid != CREATININE_CONCEPT_ID]

    # Deterministic pseudo-random fraction in [0,1)
    frac_expr = f"""
      (
        (ABS(hash(CAST(person_id AS VARCHAR) || '_' || CAST({seed} AS VARCHAR))) % 1000000)
        / 1000000.0
      )
    """

    # Base cohort and random anchors
    if outcome_id == "predefined":
        base_ctes = f"""
        negative_cohort AS (
            SELECT
                person_id,
                observation_period_start_date,
                observation_period_end_date,
                DATE_DIFF('day', observation_period_start_date, observation_period_end_date) AS duration_days
            FROM {source_schema}.observation_period
            WHERE person_id IN (SELECT person_id FROM {result_schema}.negative_outcome_patients)
              AND DATE_DIFF('day', observation_period_start_date, observation_period_end_date) >= {time_window_days}
        ),
        random_anchors AS (
            SELECT *,
                   observation_period_start_date
                   + CAST(
                       FLOOR(
                         {frac_expr} * GREATEST(duration_days - {time_window_days}, 0)
                       ) AS INTEGER
                     ) * INTERVAL 1 DAY
                   AS index_date
            FROM negative_cohort
        )
        """
    else:
        base_ctes = f"""
        eligible_patients AS (
            SELECT
                person_id,
                observation_period_start_date,
                observation_period_end_date,
                DATE_DIFF('day', observation_period_start_date, observation_period_end_date) AS duration_days
            FROM {source_schema}.observation_period
            WHERE DATE_DIFF('day', observation_period_start_date, observation_period_end_date) >= {time_window_days}
        ),
        negative_cohort AS (
            SELECT e.*
            FROM eligible_patients e
            LEFT JOIN {source_schema}.condition_occurrence c
              ON e.person_id = c.person_id AND c.condition_concept_id = {int(outcome_id)}
            WHERE c.person_id IS NULL
        ),
        random_anchors AS (
            SELECT *,
                   observation_period_start_date
                   + CAST(
                       FLOOR(
                         {frac_expr} * GREATEST(duration_days - {time_window_days}, 0)
                       ) AS INTEGER
                     ) * INTERVAL 1 DAY
                   AS index_date
            FROM negative_cohort
        )
        """

    feature_ctes: List[str] = []
    feature_selects: List[str] = []
    joins: List[str] = []

    # 4 dense burden features (anchored on random_anchors.index_date)
    feature_ctes.append(f"""
    burden_features AS (
        SELECT
            n.person_id,
            n.index_date,
            COUNT(DISTINCT v.visit_occurrence_id) AS visit_count,
            COUNT(DISTINCT d.drug_concept_id) AS distinct_drug_count,
            COUNT(DISTINCT c.condition_concept_id) AS distinct_condition_count
        FROM random_anchors n
        LEFT JOIN {source_schema}.visit_occurrence v
          ON v.person_id = n.person_id
         AND v.visit_start_date >= (n.index_date - INTERVAL {time_window_days} DAY)
         AND v.visit_start_date <  n.index_date
        LEFT JOIN {source_schema}.drug_era d
          ON d.person_id = n.person_id
         AND d.drug_era_end_date >= (n.index_date - INTERVAL {time_window_days} DAY)
         AND d.drug_era_start_date <  n.index_date
        LEFT JOIN {source_schema}.condition_occurrence c
          ON c.person_id = n.person_id
         AND c.condition_start_date >= (n.index_date - INTERVAL {time_window_days} DAY)
         AND c.condition_start_date <  n.index_date
        GROUP BY n.person_id, n.index_date
    )
    """)

    feature_ctes.append(f"""
    observation_depth AS (
        SELECT
            n.person_id,
            n.index_date,
            MAX(DATE_DIFF('day', op.observation_period_start_date, n.index_date)) AS days_observed_before_index
        FROM random_anchors n
        LEFT JOIN {source_schema}.observation_period op
          ON op.person_id = n.person_id
         AND n.index_date BETWEEN op.observation_period_start_date AND op.observation_period_end_date
        GROUP BY n.person_id, n.index_date
    )
    """)

    # Exposure features (binary)
    if top_exposures:
        feature_ctes.append(f"""
        exposure_features AS (
            SELECT d.person_id, d.drug_concept_id
            FROM {source_schema}.drug_era d
            JOIN random_anchors n
              ON d.person_id = n.person_id
            WHERE d.drug_concept_id IN ({",".join(map(str, top_exposures))})
              AND d.drug_era_end_date >= n.index_date - INTERVAL {time_window_days} DAY
              AND d.drug_era_start_date <  n.index_date
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN d.drug_concept_id = {cid} THEN 1 ELSE 0 END) AS exposure_{cid}"
            for cid in top_exposures
        ])
        joins.append("LEFT JOIN exposure_features d ON n.person_id = d.person_id")

    # Measurement features (continuous: latest value pre-index)
    if top_measurements:
        feature_ctes.append(f"""
        measurement_features AS (
            SELECT
                m.person_id,
                m.measurement_concept_id,
                m.value_as_number,
                n.index_date,
                ROW_NUMBER() OVER (
                    PARTITION BY m.person_id, m.measurement_concept_id
                    ORDER BY m.measurement_date DESC
                ) AS rn
            FROM {source_schema}.measurement m
            JOIN random_anchors n
              ON m.person_id = n.person_id
            WHERE m.measurement_concept_id IN ({",".join(map(str, top_measurements))})
              AND m.measurement_date >= n.index_date - INTERVAL {time_window_days} DAY
              AND m.measurement_date <  n.index_date
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN m.measurement_concept_id = {cid} AND m.rn = 1 "
            f"THEN m.value_as_number ELSE NULL END) AS measurement_{cid}"
            for cid in top_measurements
        ])
        joins.append("LEFT JOIN measurement_features m ON n.person_id = m.person_id")

    # Procedure features (binary)
    if top_procedures:
        feature_ctes.append(f"""
        procedure_features AS (
            SELECT proc.person_id, proc.procedure_concept_id
            FROM {source_schema}.procedure_occurrence proc
            JOIN random_anchors n
              ON proc.person_id = n.person_id
            WHERE proc.procedure_concept_id IN ({",".join(map(str, top_procedures))})
              AND proc.procedure_date >= n.index_date - INTERVAL {time_window_days} DAY
              AND proc.procedure_date <  n.index_date
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN proc.procedure_concept_id = {cid} THEN 1 ELSE 0 END) AS procedure_{cid}"
            for cid in top_procedures
        ])
        joins.append("LEFT JOIN procedure_features proc ON n.person_id = proc.person_id")

    # Extra condition features (binary)
    if extra_conditions:
        feature_ctes.append(f"""
        condition_features AS (
            SELECT c.person_id, c.condition_concept_id
            FROM {source_schema}.condition_occurrence c
            JOIN random_anchors n
              ON c.person_id = n.person_id
            WHERE c.condition_concept_id IN ({",".join(map(str, extra_conditions))})
              AND c.condition_start_date >= n.index_date - INTERVAL {time_window_days} DAY
              AND c.condition_start_date <  n.index_date
        )
        """)
        feature_selects.extend([
            f"MAX(CASE WHEN cf.condition_concept_id = {cid} THEN 1 ELSE 0 END) AS condition_{cid}"
            for cid in extra_conditions
        ])
        joins.append("LEFT JOIN condition_features cf ON n.person_id = cf.person_id")

    select_clause = f"""
        n.person_id,
        n.index_date AS condition_start_date,
        0 AS condition_concept_id,
        p.gender_concept_id,
        p.race_concept_id,
        YEAR(n.index_date) - p.year_of_birth AS age_at_outcome,

        COALESCE(b.visit_count, 0) AS visit_count,
        COALESCE(b.distinct_drug_count, 0) AS distinct_drug_count,
        COALESCE(b.distinct_condition_count, 0) AS distinct_condition_count,
        COALESCE(od.days_observed_before_index, 0) AS days_observed_before_index
        {"," if feature_selects else ""}{", ".join(feature_selects)}
    """

    with_clause = base_ctes
    if feature_ctes:
        with_clause += ",\n" + ",\n".join(feature_ctes)

    with_clause += f""",
    final_features AS (
        SELECT
            {select_clause}
        FROM random_anchors n
        JOIN {source_schema}.person p
          ON n.person_id = p.person_id

        LEFT JOIN burden_features b
          ON n.person_id = b.person_id AND n.index_date = b.index_date
        LEFT JOIN observation_depth od
          ON n.person_id = od.person_id AND n.index_date = od.index_date

        {" ".join(joins)}
        GROUP BY
            n.person_id, n.index_date,
            p.gender_concept_id, p.race_concept_id, p.year_of_birth,
            b.visit_count, b.distinct_drug_count, b.distinct_condition_count,
            od.days_observed_before_index
    )
    """

    query = f"WITH {with_clause} SELECT *, 0 AS label FROM final_features;"
    sql = qualify_tables(query, source_schema=source_schema)
    sql = translate_sql(sql, dialect=dialect)
    return sql