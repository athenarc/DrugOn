# utils/sql_portability.py
import re
import sqlglot
from sqlalchemy import create_engine, text
import pandas as pd

# List the OMOP tables you reference (add more if needed)
OMOP_TABLES = [
    "person", "visit_occurrence", "visit_detail", "drug_exposure", "drug_era",
    "condition_occurrence", "measurement", "observation", "procedure_occurrence",
    "observation_period", "concept"
]

# Regex to match bare table names (not already schema-qualified)
TABLE_REGEX = re.compile(r"(?<!\.)\b({})\b".format("|".join(map(re.escape, OMOP_TABLES))), re.IGNORECASE)

def qualify_tables(sql: str, source_schema: str) -> str:
    """
    Prefix known OMOP table names with the schema if not already qualified.
    e.g., 'FROM drug_era' -> 'FROM main.drug_era'
    """
    def _repl(m):
        name = m.group(1)
        return f"{source_schema}.{name}"
    return TABLE_REGEX.sub(_repl, sql)

def translate_sql(sql: str, dialect: str) -> str:
    """
    If target is PostgreSQL, transpile DuckDB SQL -> Postgres using sqlglot.
    Otherwise return SQL as-is (DuckDB).
    """
    dialect = dialect.lower()
    if dialect in ("postgres", "postgresql"):
        # sqlglot will handle DATE_DIFF, INTERVAL, ABS, etc. for these dialects.
        sql = sqlglot.transpile(sql, read="duckdb", write="postgres")[0]
        sql = re.sub(r'\bYEAR\s*\(\s*([^)]+)\s*\)', r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)
        return sql
    return sql

def connect_postgres(url: str):
    # e.g. url = "postgresql+psycopg://user:pass@host:5432/dbname"
    return create_engine(url).connect()

def fetch_df_pg(pg_conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), pg_conn)

def fetch_df(con_or_pg, sql: str, dialect: str) -> pd.DataFrame:
    if dialect == "duckdb":
        return con_or_pg.execute(sql).fetchdf()
    else:
        return fetch_df_pg(con_or_pg, sql)
    
def create_temp_from_df_pg(pg_conn, df: pd.DataFrame, table_name: str):
    df.to_sql(table_name, pg_conn, if_exists="replace", index=False)  # creates a real table
    pg_conn.execute(f'CREATE TEMP TABLE {table_name}_temp AS TABLE {table_name}')
    pg_conn.execute(f'DROP TABLE {table_name}')
    pg_conn.execute(f'ALTER TABLE {table_name}_temp RENAME TO {table_name}')

def persist_df_duckdb(con, df, schema: str, table: str):
    # ensure schema exists
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    # register the DF temporarily so we can CTAS from it
    con.register("__df_tmp__", df)
    con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM __df_tmp__")

    # optional: cleanup the temp registration
    con.unregister("__df_tmp__")

def persist_df_postgres(pg_conn, df, schema: str, table: str):
    # ensure schema exists (requires privileges)
    pg_conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    # write/replace the table in that schema
    df.to_sql(table, pg_conn, schema=schema, if_exists="replace", index=False)
