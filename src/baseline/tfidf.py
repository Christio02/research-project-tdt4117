"""TF-IDF weighting and the document-term matrix (Assignment 1 baseline).

Sparse representation: {doc_id: {term: tfidf_weight}} — only terms present
in the doc are stored
"""

from __future__ import annotations

import math

from baseline.tokenize import tokenize


def term_freq(text: str) -> dict[str, int]:
    """Raw term frequencies for one document's text (re-tokenizes, so it
    counts occurrences even though docs[...]["terms"] is deduped)."""
    freq: dict[str, int] = {}
    for t in tokenize(text):
        freq[t] = freq.get(t, 0) + 1
    return freq


def build_doc_term_freqs(docs: dict[int, dict]) -> dict[int, dict[str, int]]:
    return {doc_id: term_freq(doc["abstract"]) for doc_id, doc in docs.items()}


def build_idf(inverted_index: dict[str, list[int]], num_docs: int) -> dict[str, float]:
    """idf_t = log(N / df_t), df_t from the inverted index."""
    return {
        term: math.log(num_docs / len(doc_ids))
        for term, doc_ids in inverted_index.items()
    }


def build_tfidf_matrix(
    docs: dict[int, dict],
    idf: dict[str, float],
) -> dict[int, dict[str, float]]:
    """Document-term matrix: {doc_id: {term: tf * idf[term]}}."""
    return {
        doc_id: {
            term: tf * idf[term]
            for term, tf in term_freq(doc["abstract"]).items()
        }
        for doc_id, doc in docs.items()
    }


def compute_doc_norms(tfidf_matrix: dict[int, dict[str, float]]) -> dict[int, float]:
    """Euclidean norm per document vector, floored to avoid div-by-zero."""
    norms = {}
    for doc_id, weights in tfidf_matrix.items():
        norm = math.sqrt(sum(w * w for w in weights.values()))
        norms[doc_id] = norm if norm > 0 else 1e-12
    return norms
