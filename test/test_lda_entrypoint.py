import os
import pytest
import psycopg2
import time
import pandas as pd

from pathlib import Path
from main import lda_topic_modeling

MINIO_USER = "minioadmin"
MINIO_PWD = "minioadmin"
BUCKET_NAME = "testbucket"

POSTGRES_USER = "postgres"
POSTGRES_PWD = "postgres"

N_TOPICS = 5


@pytest.fixture(scope="session")
def postgres_conn():
    """Wait until postgres is ready, then yield a live connection."""
    for _ in range(30):
        try:
            conn = psycopg2.connect(
                host="127.0.0.1",
                port=5432,
                user=POSTGRES_USER,
                password=POSTGRES_PWD,
                database="postgres",
            )
            conn.autocommit = True
            yield conn
            conn.close()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Postgres did not start")


def test_lda_entrypoint(postgres_conn):
    doc_topic_table_name = "doc_topic"
    topic_terms_table_name = "topic_terms"

    sql_dump_path = Path(__file__).parent / "files" / "norm_docs_dump.sql"
    with open(sql_dump_path, "r") as f:
        sql = f.read()

    cur = postgres_conn.cursor()
    cur.execute("DROP TABLE IF EXISTS norm_docs;")
    cur.execute(sql)

    env = {
        "N_TOPICS": "5",
        "preprocessed_docs_DB_DSN": f"postgresql://{POSTGRES_USER}:{POSTGRES_PWD}@127.0.0.1:5432/postgres",
        "preprocessed_docs_DB_TABLE": "norm_docs",
        "docs_to_topics_DB_DSN": f"postgresql://{POSTGRES_USER}:{POSTGRES_PWD}@127.0.0.1:5432/postgres",
        "docs_to_topics_DB_TABLE": doc_topic_table_name,
        "top_terms_per_topic_DB_DSN": f"postgresql://{POSTGRES_USER}:{POSTGRES_PWD}@127.0.0.1:5432/postgres",
        "top_terms_per_topic_DB_TABLE": topic_terms_table_name,
    }

    for k, v in env.items():
        os.environ[k] = v

    lda_topic_modeling()

    cur = postgres_conn.cursor()

    # 1. doc-topic distribution
    cur.execute(f"SELECT * FROM public.{doc_topic_table_name} ORDER BY 1;")
    doc_topics = pd.DataFrame(
        cur.fetchall(), columns=[desc[0] for desc in cur.description]
    )
    assert len(doc_topics) == 2
    # expect N_TOPICS + 1 for doc_id
    assert doc_topics.shape[1] == N_TOPICS + 1

    # 2. topic-term listing
    cur.execute(
        f"SELECT * FROM public.{
            topic_terms_table_name
        } ORDER BY topic_id, weight DESC;"
    )
    topic_terms = pd.DataFrame(
        cur.fetchall(), columns=[desc[0] for desc in cur.description]
    )
    assert len(topic_terms) > 0
    assert "term" in topic_terms.columns
