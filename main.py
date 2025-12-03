import logging
import pickle

from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    FileSettings,
    PostgresSettings,
)
from scystream.sdk.file_handling.s3_manager import S3Operations
from sqlalchemy import create_engine

from algorithms.lda import LDAModeler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DTMFileInput(FileSettings, InputSettings):
    __identifier__ = "dtm"

    FILE_EXT: str = "pkl"


class VocabFileInput(FileSettings, InputSettings):
    __identifier__ = "vocab"

    FILE_EXT: str = "pkl"


class DocTopicOutput(PostgresSettings, OutputSettings):
    __identifier__ = "docs_to_topics"


class TopicTermsOutput(PostgresSettings, OutputSettings):
    __identifier__ = "top_terms_per_topic"


class LDATopicModeling(EnvSettings):
    N_TOPICS: int = 5
    MAX_ITER: int = 10
    LEARNING_METHOD: str = "batch"
    N_TOP_WORDS: int = 10

    vocab: VocabFileInput
    dtm: DTMFileInput

    doc_topic: DocTopicOutput
    topic_term: TopicTermsOutput


def write_df_to_postgres(df, settings: PostgresSettings):
    logger.info(f"Writing DataFrame to DB table '{settings.DB_TABLE}'…")

    engine = create_engine(
        f"postgresql+psycopg2://{settings.PG_USER}:{settings.PG_PASS}"
        f"@{settings.PG_HOST}:{int(settings.PG_PORT)}/"
    )
    df.to_sql(settings.DB_TABLE, engine, if_exists="replace", index=False)
    logger.info(f"Successfully wrote {len(df)} rows to '{settings.DB_TABLE}'.")


@entrypoint(LDATopicModeling)
def lda_topic_modeling(settings):
    logger.info("Starting LDA topic modeling pipeline…")

    logger.info("Downloading vocabulary file...")
    S3Operations.download(settings.vocab, "vocab.pkl")

    logger.info("Loading vocab.pkl from disk...")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    logger.info(f"Loaded vocab with {len(vocab)} terms.")

    logger.info("Downloading DTM file...")
    S3Operations.download(settings.dtm, "dtm.pkl")

    logger.info("Loading dtm.pkl from disk...")
    with open("dtm.pkl", "rb") as f:
        dtm = pickle.load(f)

    logger.info(f"Loaded DTM with shape {dtm.shape}")

    # TODO: Check if dtm and vocab is of correct schema
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


"""
if __name__ == "__main__":
    test = LDATopicModeling(
        vocab=VocabFileInput(
            S3_HOST="http://localhost",
            S3_PORT="9000",
            S3_ACCESS_KEY="minioadmin",
            S3_SECRET_KEY="minioadmin",
            BUCKET_NAME="output-bucket",
            FILE_PATH="output_file_path",
            FILE_NAME="vocab_file_bib",
        ),
        dtm=DTMFileInput(
            S3_HOST="http://localhost",
            S3_PORT="9000",
            S3_ACCESS_KEY="minioadmin",
            S3_SECRET_KEY="minioadmin",
            BUCKET_NAME="output-bucket",
            FILE_PATH="output_file_path",
            FILE_NAME="dtm_file_bib",
        ),
        doc_topic=DocTopicOutput(
            PG_USER="postgres",
            PG_PASS="postgres",
            PG_HOST="localhost",
            PG_PORT="5432",
            DB_TABLE="doc_topic"
        ),
        topic_term=TopicTermsOutput(
            PG_USER="postgres",
            PG_PASS="postgres",
            PG_HOST="localhost",
            PG_PORT="5432",
            DB_TABLE="topic_term"
        )
    )

    lda_topic_modeling(test)
"""
