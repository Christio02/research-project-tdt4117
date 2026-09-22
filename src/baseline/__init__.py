"""Assignment 1 baseline: Cranfield parsing, tokenization, inverted index, TF-IDF.

Ported from Assignment1/assignment1_tdt4117.ipynb so both the TF-IDF baseline
and the Random Indexing engine (src/random_indexing) reuse the same `docs`
parsing, `tokenize`, and `inverted_index` instead of duplicating them.
"""

from baseline.corpus import CRAN_PATH, parse_cranfield
from baseline.index import build_inverted_index, build_term_index
from baseline.tokenize import STOPWORDS, add_terms_to_docs, tokenize, unique_terms
from baseline.tfidf import (
    build_doc_term_freqs,
    build_idf,
    build_tfidf_matrix,
    compute_doc_norms,
    term_freq,
)

__all__ = [
    "CRAN_PATH",
    "STOPWORDS",
    "parse_cranfield",
    "tokenize",
    "add_terms_to_docs",
    "unique_terms",
    "build_inverted_index",
    "build_term_index",
    "term_freq",
    "build_doc_term_freqs",
    "build_idf",
    "build_tfidf_matrix",
    "compute_doc_norms",
]
