"""Similarity backends, best available first: sentence-transformers > LSA > TF-IDF."""
import logging
import os
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger("resumelens")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, corpus: Optional[List[str]] = None):
        self.model = None
        self.lsa = None
        self.name = "tfidf"
        mode = os.getenv("RESUMELENS_EMBEDDER", "auto")  # auto | st | lsa | tfidf
        if mode in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(MODEL_NAME)
                self.name = "all-MiniLM-L6-v2"
            except Exception as exc:
                log.info("sentence-transformers not available (%s)", type(exc).__name__)
        if self.model is None and corpus and mode in ("auto", "lsa", "st"):
            try:
                vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(corpus)
                X = vec.transform(corpus)
                k = max(2, min(80, X.shape[1] - 1, X.shape[0] - 1))
                self.lsa = (vec, TruncatedSVD(k, random_state=0).fit(X))
                self.name = "lsa+tfidf"
            except Exception as exc:
                log.warning("LSA failed (%s); using TF-IDF", exc)

    @staticmethod
    def _tfidf(a, b):
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(a + b)
        except ValueError:
            return np.zeros((len(a), len(b)))
        return (vec.transform(a) @ vec.transform(b).T).toarray()

    def similarity(self, a: list, b: list) -> np.ndarray:
        if not a or not b:
            return np.zeros((len(a), len(b)))
        if self.model is not None:
            ea = self.model.encode(a, normalize_embeddings=True)
            eb = self.model.encode(b, normalize_embeddings=True)
            return np.asarray(ea @ eb.T)
        lex = self._tfidf(a, b)
        if self.lsa is not None:
            vec, svd = self.lsa
            ea = normalize(svd.transform(vec.transform(a)))
            eb = normalize(svd.transform(vec.transform(b)))
            return 0.5 * lex + 0.5 * np.clip(ea @ eb.T, 0, 1)
        return lex

    def scale(self, sims: np.ndarray) -> np.ndarray:
        lo, hi = {"all-MiniLM-L6-v2": (0.25, 0.65), "lsa+tfidf": (0.05, 0.45)}.get(self.name, (0.0, 0.30))
        return np.clip((sims - lo) / (hi - lo), 0, 1)
