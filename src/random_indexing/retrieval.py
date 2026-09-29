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

from baseline.tokenize import tokenize
from compositional_vector_model.composition import compose_query


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """TODO: implement standard cosine similarity, guarding against zero norms."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def nearest_terms(
    term: str, context_vectors: dict[str, np.ndarray], top_k: int = 10
) -> list[tuple[str, float]]:
    """Rank all other terms by cosine similarity to `term`'s context vector.
    E.g: Which other words in the dictionary point in almost the same direction as
    'wing'?"
    TODO:
    - look up `term`'s vector (raise/handle cleanly if out of vocabulary)
    - score every other term via cosine_similarity
    - return the top_k (term, score) pairs, highest first
    """
    # get context vector
    if term not in context_vectors:
        return []
    target_vector = context_vectors[term]

    scores = []

    for other_term, other_vector in context_vectors.items():
        if other_term == term:
            continue
        similarity = cosine_similarity(target_vector, other_vector)
        scores.append((other_term, similarity))

    sorted_scores = sorted(scores, key=lambda pair: pair[1], reverse=True)
    top_k = sorted_scores[:top_k]
    return top_k


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
    document_vectors = {}
    for doc_id, doc_data in docs.items():
        terms = doc_data["terms"]
        filtered_terms = [term for term in terms if term in context_vectors]

        try:
            document_vectors[doc_id] = compose_query(
                terms=filtered_terms, context_vectors=context_vectors, method=method
            )
        except ValueError:
            continue

    return document_vectors


def ri_retrieve(
    query: str,
    context_vectors: dict[str, np.ndarray],
    doc_vectors: dict[int, np.ndarray],
    method: str = "additive",
    lam: float | None = None,
    a: float | None = None,
    b: float | None = None,
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
    tokens = tokenize(query)
    try:
        query_vector = compose_query(
            terms=tokens,
            context_vectors=context_vectors,
            method=method,
            lam=lam,
            a=a,
            b=b,
        )
        results = []
        for doc_id, doc_vector in doc_vectors.items():
            score = cosine_similarity(query_vector, doc_vector)
            if score > 0.0:
                results.append((doc_id, score))
        return sorted(results, key=lambda item: item[1], reverse=True)

    except ValueError:
        return []
