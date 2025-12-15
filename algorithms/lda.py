import logging

import numpy as np
import pandas as pd

from typing import List

from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)


class LDAModeler:
    def __init__(
        self,
        dtm: np.ndarray = None,
        vocab: dict = None,
        doc_ids: List[str] = [],
        n_topics: int = 10,
        max_iter: int = 10,
        learning_method: str = "batch",
        random_state: int = 42,
        n_top_words: int = 10,
    ):
        self.dtm: np.ndarray = dtm
        self.vocab: dict = vocab
        self.doc_ids = doc_ids

        if len(self.doc_ids) != self.dtm.shape[0]:
            raise ValueError(
                "doc_ids length must match number of DTM rows"
            )

        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state
        self.n_top_words = n_top_words

        logger.debug(
            f"Initialized LDAModeler: topics={n_topics}, iter={max_iter}, "
            f"learning='{learning_method}', n_top_words={n_top_words}"
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state
        )

        self.doc_topic_dist = None

    def fit(self):
        logger.info(f"Running LDA fit(), DTM shape {self.dtm.shape}")
        self.lda.fit(self.dtm)
        logger.info("LDA model fitted successfully")

    def extract_doc_topics(self) -> pd.DataFrame:
        """
        Generates document-topic distribution DF
        """
        logger.info("Extracting doc-topics...")
        self.doc_topic_dist = self.lda.transform(self.dtm)

        df = pd.DataFrame(
            self.doc_topic_dist,
            columns=[f"topic_{i}" for i in range(self.n_topics)],
        )
        df.insert(0, "doc_id", self.doc_ids)

        logger.debug(
            f"Extracted doc-topic distribution DataFrame shape={df.shape}")
        return df

    def extract_topic_terms(self):
        """
        Generate topic and top-terms DataFrame
        """
        logger.info("Extracting top terms per topic...")

        # NOTE:
        # The order of `terms` is guaranteed to match the DTM column order.
        # This is because the vocabulary is built in NLPVectorizer using:
        #   sorted_terms = sorted(all_terms)
        #   vocab = {term: i for i, term in enumerate(sorted_terms)}
        # The same vocab indices are then used to construct the DTM columns.
        # Since Python dicts preserve insertion order (>=3.7),
        # list(self.vocab.keys())[i] correctly maps to DTM column i,
        # and thus to lda.components_[topic_idx][i].
        terms = list(self.vocab.keys())
        topic_rows = []

        for topic_idx, topic in enumerate(self.lda.components_):
            sorted_idx = np.argsort(topic)[::-1]
            top_indices = sorted_idx[: self.n_top_words]

            for i in top_indices:
                topic_rows.append({
                    "topic_id": topic_idx,
                    "term": terms[int(i)],
                    "weight": topic[i],
                })

        df = pd.DataFrame(topic_rows)
        logger.info(f"Generated topic_terms DataFrame rows={df.shape[0]}")
        return df
