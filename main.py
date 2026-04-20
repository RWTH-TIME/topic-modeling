import logging

from scystream.sdk.core import entrypoint
from scystream.sdk.database_handling.database_manager import (
    PandasDatabaseOperations,
)
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    DatabaseSettings,
)

from algorithms.lda import LDAModeler
from algorithms.models import PreprocessedDocument
from algorithms.vectorizer import NLPVectorizer

from algorithms.explanations import TopicExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PreprocessedDocuments(DatabaseSettings, InputSettings):
    __identifier__ = "preprocessed_docs"


class DocTopicOutput(DatabaseSettings, OutputSettings):
    __identifier__ = "docs_to_topics"


class TopicTermsOutput(DatabaseSettings, OutputSettings):
    __identifier__ = "top_terms_per_topic"


class LDATopicModeling(EnvSettings):
    N_TOPICS: int = 5
    MAX_ITER: int = 10
    LEARNING_METHOD: str = "batch"
    N_TOP_WORDS: int = 10

    preprocessed_docs: PreprocessedDocuments

    doc_topic: DocTopicOutput
    topic_term: TopicTermsOutput


class TopicTermsInput(DatabaseSettings, InputSettings):
    __identifier__ = "topic_terms_input"


class QueryInformationInput(DatabaseSettings, InputSettings):
    """
    Our TopicExplaination needs some kind of information about the actual
    query executed,this query information includes the query and the source

    Looking like: query, source, created_at
    """

    __identifier__ = "query_information_input"


class ExplanationsOutput(DatabaseSettings, OutputSettings):
    __identifier__ = "explanations_output"


class TopicExplanation(EnvSettings):
    MODEL_NAME: str = "gpt-oss:120b"
    OLLAMA_API_KEY: str = ""

    topic_terms: TopicTermsInput
    query_information: QueryInformationInput

    explanations_output: ExplanationsOutput


def parse_pg_array(val):
    if isinstance(val, str):
        return val.strip("{}").split(",")
    return val


@entrypoint(LDATopicModeling)
def lda_topic_modeling(settings):
    logger.info("Starting LDA topic modeling pipeline…")

    logger.info("Querying normalized docs from db...")
    preprocessed_docs_db = PandasDatabaseOperations(
        settings.preprocessed_docs.DB_DSN, settings.preprocessed_docs.DB_SCHEMA
    )
    normalized_docs = preprocessed_docs_db.read(
        table=settings.preprocessed_docs.DB_TABLE
    )

    preprocessed_docs = [
        PreprocessedDocument(
            doc_id=row["doc_id"], tokens=parse_pg_array(row["tokens"])
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
        n_top_words=settings.N_TOP_WORDS,
    )
    lda.fit()

    doc_topics = lda.extract_doc_topics()
    topic_terms = lda.extract_topic_terms()

    # TODO: Use Spark Integration here
    logging.info("Writing dataframes to db...")
    doc_topic_db = PandasDatabaseOperations(
        settings.doc_topic.DB_DSN, settings.doc_topic.DB_SCHEMA
    )
    topic_terms_db = PandasDatabaseOperations(
        settings.topic_term.DB_DSN, settings.topic_term.DB_SCHEMA
    )

    doc_topic_db.write(
        table=settings.doc_topic.DB_TABLE, data=doc_topics, mode="overwrite"
    )
    topic_terms_db.write(
        table=settings.topic_term.DB_TABLE, data=topic_terms, mode="overwrite"
    )


@entrypoint(TopicExplanation)
def topic_explanation(settings):
    logger.info("Starting topic explaination...")

    logging.info("Querying topic terms from db...")
    topic_terms_db = PandasDatabaseOperations(
        settings.topic_terms.DB_DSN, settings.topic_terms.DB_SCHEMA
    )
    topic_terms = topic_terms_db.read(table=settings.topic_terms.DB_TABLE)

    logging.info("Querying query information from db...")
    query_info_db = PandasDatabaseOperations(
        settings.query_information.DB_DSN, settings.query_information.DB_SCHEMA
    )
    query_information = query_info_db.read(
        table=settings.query_information.DB_TABLE
    )

    metadata = query_information.iloc[0]

    explainer = TopicExplainer(
        model_name=settings.MODEL_NAME, api_key=settings.OLLAMA_API_KEY
    )

    explanations = explainer.explain_topics(
        topic_terms=topic_terms,
        search_query=metadata["query"],
        source=metadata["source"],
        created_at=metadata["created_at"],
    )

    explainations_output_db = PandasDatabaseOperations(
        settings.explanations_output.DB_DSN,
        settings.explanations_output.DB_SCHEMA,
    )
    explainations_output_db.write(
        table=settings.explanations_output.DB_TABLE,
        data=explanations,
        mode="overwrite",
    )

    logging.info("Topic explanation block finished.")
