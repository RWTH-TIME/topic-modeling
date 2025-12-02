import os
import boto3
import pytest
import psycopg2
import time
import pandas as pd

from pathlib import Path
from botocore.exceptions import ClientError
from main import lda_topic_modeling

MINIO_USER = "minioadmin"
MINIO_PWD = "minioadmin"
BUCKET_NAME = "testbucket"

POSTGRES_USER = "postgres"
POSTGRES_PWD = "postgres"

N_TOPICS = 5


def ensure_bucket(s3, bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=bucket)
        else:
            raise


def download_to_tmp(s3, bucket, key):
    tmp_path = Path("/tmp") / key.replace("/", "_")
    s3.download_file(bucket, key, str(tmp_path))
    return tmp_path


@pytest.fixture
def s3_minio():
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PWD
    )
    ensure_bucket(client, BUCKET_NAME)
    return client


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
                database="postgres"
            )
            conn.autocommit = True
            yield conn
            conn.close()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Postgres did not start")


def test_lda_entrypoint(s3_minio, postgres_conn):
    input_dtm_file_name = "dtm"
    input_vocab_file_name = "vocab"

    doc_topic_table_name = "doc_topic"
    topic_terms_table_name = "topic_terms"

    dtm_path = Path(__file__).parent / "files" / f"{input_dtm_file_name}.pkl"
    dtm_bytes = dtm_path.read_bytes()

    vocab_path = Path(__file__).parent / "files" / \
        f"{input_vocab_file_name}.pkl"
    vocab_bytes = vocab_path.read_bytes()

    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_dtm_file_name}.pkl",
        Body=dtm_bytes
    )
    s3_minio.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{input_vocab_file_name}.pkl",
        Body=vocab_bytes
    )

    env = {
        "N_TOPICS": "5",

        "dtm_S3_HOST": "http://127.0.0.1",
        "dtm_S3_PORT": "9000",
        "dtm_S3_ACCESS_KEY": MINIO_USER,
        "dtm_S3_SECRET_KEY": MINIO_PWD,
        "dtm_BUCKET_NAME": BUCKET_NAME,
        "dtm_FILE_PATH": "",
        "dtm_FILE_NAME": input_dtm_file_name,
        "dtm_FILE_EXT": "pkl",

        "vocab_S3_HOST": "http://127.0.0.1",
        "vocab_S3_PORT": "9000",
        "vocab_S3_ACCESS_KEY": MINIO_USER,
        "vocab_S3_SECRET_KEY": MINIO_PWD,
        "vocab_BUCKET_NAME": BUCKET_NAME,
        "vocab_FILE_PATH": "",
        "vocab_FILE_NAME": input_vocab_file_name,
        "vocab_FILE_EXT": "pkl",

        "docs_to_topics_PG_HOST": "127.0.0.1",
        "docs_to_topics_PG_PORT": "5432",
        "docs_to_topics_PG_USER": POSTGRES_USER,
        "docs_to_topics_PG_PASS": POSTGRES_PWD,
        "docs_to_topics_DB_TABLE": doc_topic_table_name,

        "top_terms_per_topic_PG_HOST": "127.0.0.1",
        "top_terms_per_topic_PG_PORT": "5432",
        "top_terms_per_topic_PG_USER": POSTGRES_USER,
        "top_terms_per_topic_PG_PASS": POSTGRES_PWD,
        "top_terms_per_topic_DB_TABLE": topic_terms_table_name,
    }

    for k, v in env.items():
        os.environ[k] = v

    lda_topic_modeling()

    cur = postgres_conn.cursor()

    # 1. doc-topic distribution
    cur.execute(f"SELECT * FROM {doc_topic_table_name} ORDER BY 1;")
    doc_topics = pd.DataFrame(cur.fetchall(), columns=[
                              desc[0] for desc in cur.description])
    assert len(doc_topics) == 26
    assert doc_topics.shape[1] == N_TOPICS

    # 2. topic-term listing
    cur.execute(
        f"SELECT * FROM {
            topic_terms_table_name} ORDER BY topic_id, weight DESC;")
    topic_terms = pd.DataFrame(cur.fetchall(), columns=[
                               desc[0] for desc in cur.description])
    assert len(topic_terms) > 0
    assert "term" in topic_terms.columns
