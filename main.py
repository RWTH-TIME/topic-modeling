import logging
import pandas as pd

from scystream.sdk.config import get_compute_block
from scystream.sdk.config.config_loader import generate_config_from_compute_block
from pathlib import Path

from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    PostgresSettings,
)
from sqlalchemy import create_engine

from algorithms.lda import LDAModeler
from algorithms.models import PreprocessedDocument
from algorithms.vectorizer import NLPVectorizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessedDocuments(PostgresSettings, InputSettings):
    __identifier__ = "preprocessed_docs"


class DocTopicOutput(PostgresSettings, OutputSettings):
    __identifier__ = "docs_to_topics"


class TopicTermsOutput(PostgresSettings, OutputSettings):
    __identifier__ = "top_terms_per_topic"


class LDATopicModeling(EnvSettings):
    N_TOPICS: int = 5
    MAX_ITER: int = 10
    LEARNING_METHOD: str = "batch"
    N_TOP_WORDS: int = 10

    preprocessed_docs: PreprocessedDocuments

    doc_topic: DocTopicOutput
    topic_term: TopicTermsOutput


def _make_engine(settings: PostgresSettings):
    return create_engine(
        f"postgresql+psycopg2://{settings.PG_USER}:{settings.PG_PASS}"
        f"@{settings.PG_HOST}:{int(settings.PG_PORT)}/"
    )


def write_df_to_postgres(df, settings: PostgresSettings):
    logger.info(f"Writing DataFrame to DB table '{settings.DB_TABLE}'…")
    engine = _make_engine(settings)
    df.to_sql(settings.DB_TABLE, engine, if_exists="replace", index=False)
    logger.info(f"Successfully wrote {len(df)} rows to '{settings.DB_TABLE}'.")


def read_table_from_postgres(settings: PostgresSettings) -> pd.DataFrame:
    engine = _make_engine(settings)
    query = f"SELECT * FROM {settings.DB_TABLE};"
    return pd.read_sql(query, engine)


@entrypoint(LDATopicModeling)
def lda_topic_modeling(settings):
    logger.info("Starting LDA topic modeling pipeline…")

    logger.info("Querying normalized docs from db...")
    normalized_docs = read_table_from_postgres(settings.preprocessed_docs)

    preprocessed_docs = [
        PreprocessedDocument(
            doc_id=row["doc_id"],
            tokens=row["tokens"]
        )
        for _, row in normalized_docs.iterrows()
    ]

    vectorizer = NLPVectorizer(preprocessed_docs)
    vectorizer.analyze_frequencies()
    vocab = vectorizer.build_vocabulary()
    dtm = vectorizer.build_dtm()

    lda = LDAModeler(
        dtm=dtm,
        vocab=vocab,
        n_topics=settings.N_TOPICS,
        max_iter=settings.MAX_ITER,
        learning_method=settings.LEARNING_METHOD,
        random_state=42,
        n_top_words=settings.N_TOP_WORDS
    )
    lda.fit()

    doc_topics = lda.extract_doc_topics()
    topic_terms = lda.extract_topic_terms()

    # TODO: Use Spark Integration here
    write_df_to_postgres(doc_topics, settings.doc_topic)
    write_df_to_postgres(topic_terms, settings.topic_term)


if __name__ == "__main__":
    cb = get_compute_block()
    generate_config_from_compute_block(cb, Path("cbc2.yaml"))
