"""Retrieval using the Random Indexing semantic space.

Two things are built on top of the term (context) vectors:
  - term-term similarity (nearest semantic neighbours), the direct analogue
    of Ramtin's hand-built term-term relation matrix, but learned instead
    of hand-authored.
  - document ranking for a query, by composing the query's term vectors and
    comparing against document vectors built the same way documents were
    used as contexts.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """TODO: implement standard cosine similarity, guarding against zero norms."""
    raise NotImplementedError


def nearest_terms(term: str, context_vectors: dict[str, np.ndarray], top_k: int = 10) -> list[tuple[str, float]]:
    """Rank all other terms by cosine similarity to `term`'s context vector.

    TODO:
    - look up `term`'s vector (raise/handle cleanly if out of vocabulary)
    - score every other term via cosine_similarity
    - return the top_k (term, score) pairs, highest first
    """
    raise NotImplementedError


def build_document_vectors(
    docs: dict[int, dict],
    context_vectors: dict[str, np.ndarray],
    method: str = "additive",
) -> dict[int, np.ndarray]:
    """Represent each document as a composed vector of its term vectors.

    TODO:
    - for each document, gather its terms' context vectors
    - compose them via compositional_vector_model.composition.compose_query
    - return {doc_id: document_vector}
    """
    raise NotImplementedError


def ri_retrieve(
    query: str,
    context_vectors: dict[str, np.ndarray],
    doc_vectors: dict[int, np.ndarray],
    method: str = "additive",
) -> list[tuple[int, float]]:
    """Rank documents for `query` using the RI semantic space.

    TODO:
    - tokenize `query` (reuse Assignment 1's tokenizer)
    - compose the query's term vectors via
      compositional_vector_model.composition.compose_query
    - score every document by cosine_similarity(query_vector, doc_vector)
    - return (doc_id, score) sorted by score descending, mirroring the
      shape of Assignment 1's tfidf_retrieve for easy comparison
    """
    raise NotImplementedError
