import logging
import pandas as pd

from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    PostgresSettings,
)
from sqlalchemy import create_engine, text

from algorithms.lda import LDAModeler
from algorithms.models import PreprocessedDocument
from algorithms.vectorizer import NLPVectorizer

from algorithms.explanations import TopicExplainer

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


class TopicTermsInput(PostgresSettings, InputSettings):
    __identifier__ = "topic_terms_input"


class QueryInformationInput(PostgresSettings, InputSettings):
    """
    Our TopicExplaination needs some kind of information about the actual
    query executed,this query information includes the query and the source

    Looking like: query, source
    """
    __identifier__ = "query_information_input"


class ExplanationsOutput(PostgresSettings, OutputSettings):
    __identifier__ = "explanations_output"


class TopicExplanation(EnvSettings):
    MODEL_NAME: str = "gpt-oss:120b"
    OLLAMA_API_KEY: str = ""

    topic_terms: TopicTermsInput
    query_information: QueryInformationInput

    explanations_output: ExplanationsOutput


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
    query = text(f'SELECT * FROM "{settings.DB_TABLE}";')
    return pd.read_sql(query, engine)


def parse_pg_array(val):
    if isinstance(val, str):
        return val.strip("{}").split(",")
    return val


@entrypoint(LDATopicModeling)
def lda_topic_modeling(settings):
    logger.info("Starting LDA topic modeling pipeline…")

    logger.info("Querying normalized docs from db...")
    normalized_docs = read_table_from_postgres(settings.preprocessed_docs)

    preprocessed_docs = [
        PreprocessedDocument(
            doc_id=row["doc_id"],
            tokens=parse_pg_array(row["tokens"])
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
        doc_ids=vectorizer.doc_ids,
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


@entrypoint(TopicExplanation)
def topic_explaination(settings):
    logger.info("Starting topic explaination...")

    logging.info("Querying topic terms from db...")
    topic_terms = read_table_from_postgres(settings.topic_terms)

    logging.info("Querying query information from db...")
    query_information = read_table_from_postgres(settings.query_information)

    metadata = query_information.iloc[0]

    explainer = TopicExplainer(
        model_name=settings.MODEL_NAME,
        api_key=settings.OLLAMA_API_KEY
    )

    explainations = explainer.explain_topics(
        topic_terms=topic_terms,
        search_query=metadata["query"],
        source=metadata["source"],
        created_at=metadata["created_at"]
    )

    write_df_to_postgres(explainations, settings.explanations_output)

    logging.info("Topic explanation block finished.")
