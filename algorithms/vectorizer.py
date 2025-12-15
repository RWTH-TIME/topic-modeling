from typing import List
import numpy as np
from collections import Counter

from algorithms.models import PreprocessedDocument


class NLPVectorizer:
    def __init__(self, preprocessed_output: List[PreprocessedDocument]):
        self.documents = preprocessed_output
        self.doc_ids = [doc.doc_id for doc in preprocessed_output]

        # Frequencies
        self.token_frequency = Counter()
        self.token_document_frequency = Counter()
        self.ngram_frequency = Counter()
        self.ngram_document_frequency = Counter()

        # bow + dtm
        self.bag_of_words = []
        self.vocab = {}
        self.reverse_vocab = []
        self.dtm = None

    def analyze_frequencies(self):
        for doc in self.documents:
            tokens = [t for t in doc.tokens if " " not in t]
            ngrams = [t for t in doc.tokens if " " in t]

            # token frequencies
            self.token_frequency.update(tokens)
            self.token_document_frequency.update(set(tokens))

            # ngram frequencies
            self.ngram_frequency.update(ngrams)
            self.ngram_document_frequency.update(set(ngrams))

    def build_bow(self):
        bow = []

        for doc in self.documents:
            entries = []
            unique = set()

            for term in doc.tokens:
                if term in unique:
                    continue
                unique.add(term)

                is_ngram = " " in term

                entry = {
                    "term": term,
                    "type": "ngram" if is_ngram else "word",
                    "span": len(term.split(" ")),
                    "freq": (
                        self.ngram_frequency[term]
                        if is_ngram
                        else self.token_frequency[term]
                    ),
                    "docs": (
                        self.ngram_document_frequency[term]
                        if is_ngram
                        else self.token_document_frequency[term]
                    ),
                    "filters": []
                }

                entries.append(entry)

            bow.append(entries)

        self.bag_of_words = bow
        return bow

    def build_vocabulary(self):
        all_terms = set()

        for doc in self.documents:
            for term in doc.tokens:
                all_terms.add(term)

        sorted_terms = sorted(all_terms)
        self.vocab = {term: i for i, term in enumerate(sorted_terms)}
        self.reverse_vocab = sorted_terms

        return self.vocab

    def build_dtm(self):
        if not self.vocab:
            self.build_vocabulary()

        num_docs = len(self.documents)
        num_terms = len(self.vocab)

        dtm = np.zeros((num_docs, num_terms), dtype=int)

        for i, doc in enumerate(self.documents):
            for term in doc.tokens:
                dtm[i, self.vocab[term]] += 1

        self.dtm = dtm
        return dtm
