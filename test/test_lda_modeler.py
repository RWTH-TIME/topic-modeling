import pandas as pd
from algorithms.lda import LDAModeler

import numpy as np
import pytest


@pytest.fixture
def small_vocab():
    # term -> index
    return {"apple": 1, "banana": 1, "carrot": 2, "date": 3}


@pytest.fixture
def small_dtm():
    # 2 documents, 4 terms
    return np.array([
        [0, 0, 2, 0],
        [1, 1, 0, 3],
        [0, 1, 1, 1],
    ])


def test_lda_fit(small_dtm, small_vocab):
    lda = LDAModeler(dtm=small_dtm, vocab=small_vocab, n_topics=2)
    lda.fit()

    assert lda.lda.components_.shape == (2, 4)


def test_extract_doc_topics(small_dtm, small_vocab):
    lda = LDAModeler(dtm=small_dtm, vocab=small_vocab, n_topics=2)
    lda.fit()
    df = lda.extract_doc_topics()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["topic_0", "topic_1"]


def test_extract_topic_terms(small_dtm, small_vocab):
    lda = LDAModeler(dtm=small_dtm, vocab=small_vocab,
                     n_topics=2, n_top_words=2)
    lda.fit()
    df = lda.extract_topic_terms()

    # 2 topics x 2 top-words each = 4 rows
    assert df.shape[0] == 4
    assert set(df.columns) == {"topic_id", "term", "weight"}

    # All terms must come from vocab
    assert set(df["term"]).issubset(set(small_vocab.keys()))
