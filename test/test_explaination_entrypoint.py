import os
import pytest
import psycopg2
import time
import pandas as pd
import re

from pathlib import Path
from main import topic_explaination


@pytest.fixture(scope="session")
def postgres_conn():
    """Wait until postgres is ready, then yield a live connection."""
    for _ in range(30):
        try:
            conn = psycopg2.connect(
                host="127.0.0.1",
                port=5432,
                user="postgres",
                password="postgres",
                database="postgres"
            )
            conn.autocommit = True
            yield conn
            conn.close()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Postgres did not start")


def test_topic_explanation_with_real_ollama(postgres_conn):
    if not os.environ.get("OLLAMA_API_KEY"):
        pytest.skip("OLLAMA_API_KEY not set")

    topic_terms_table = "topic_terms"
    doc_topic_table = "doc_topic"
    norm_cycling_table = "norm_cycling"
    query_information_table = "query_information"

    explanations_output_table = "topic_explanations"

    sql_topic_terms = Path(__file__).parent / "files" / "topic_terms.sql"
    sql_doc_topic = Path(__file__).parent / "files" / "doc_topic.sql"
    sql_norm_cycling = Path(__file__).parent / "files" / "bike_norm.sql"

    with open(sql_topic_terms, "r") as f:
        sql_topic_terms = f.read()

    with open(sql_doc_topic, "r") as f:
        sql_doc_topic = f.read()

    with open(sql_norm_cycling, "r") as f:
        sql_norm_cycling = f.read()

    cur = postgres_conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {query_information_table};")
    cur.execute(f"DROP TABLE IF EXISTS {topic_terms_table};")
    cur.execute(f"DROP TABLE IF EXISTS {doc_topic_table};")
    cur.execute(f"DROP TABLE IF EXISTS {norm_cycling_table};")
    cur.execute(f"DROP TABLE IF EXISTS {explanations_output_table};")

    cur.execute(sql_topic_terms)
    cur.execute(sql_doc_topic)
    cur.execute(sql_norm_cycling)

    # Manually create query information, TODO: Use real export here
    cur.execute(f"""
        CREATE TABLE public.{query_information_table} (
            query TEXT,
            source TEXT,
            created_at TEXT
        );
    """)

    cur.execute(f"""
        INSERT INTO public.{query_information_table} VALUES
        ('ALL=(bike accident)', 'Web Of Science', '2024-01-01');
    """)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    env = {
        "topic_terms_input_PG_HOST": "127.0.0.1",
        "topic_terms_input_PG_PORT": "5432",
        "topic_terms_input_PG_USER": "postgres",
        "topic_terms_input_PG_PASS": "postgres",
        "topic_terms_input_DB_TABLE": topic_terms_table,

        "query_information_input_PG_HOST": "127.0.0.1",
        "query_information_input_PG_PORT": "5432",
        "query_information_input_PG_USER": "postgres",
        "query_information_input_PG_PASS": "postgres",
        "query_information_input_DB_TABLE": query_information_table,

        "explanations_output_PG_HOST": "127.0.0.1",
        "explanations_output_PG_PORT": "5432",
        "explanations_output_PG_USER": "postgres",
        "explanations_output_PG_PASS": "postgres",
        "explanations_output_DB_TABLE": explanations_output_table,

        "MODEL_NAME": "gpt-oss:120b",
        "API_KEY": os.environ.get("OLLAMA_API_KEY")
    }

    for k, v in env.items():
        os.environ[k] = v

    # ------------------------------------------------------------------
    # Run entrypoint
    # ------------------------------------------------------------------

    topic_explaination()

    # ------------------------------------------------------------------
    # Validate results
    # ------------------------------------------------------------------

    cur.execute(
        f"SELECT * FROM public.{explanations_output_table} ORDER BY topic_id;")
    results = pd.DataFrame(
        cur.fetchall(),
        columns=[desc[0] for desc in cur.description]
    )

    assert len(results) == 5
    assert "topic_id" in results.columns
    assert "description" in results.columns

    # Must not be empty
    assert results["description"].str.len().min() > 10

    for desc in results["description"]:
        assert isinstance(desc, str)
        assert len(desc) > 20
        sentences = re.findall(r"[A-Z][^.?!]*[.?!]", desc)
        assert len(sentences) <= 2
