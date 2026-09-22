"""Inverted index and vocabulary indexing."""

from __future__ import annotations

from baseline.tokenize import unique_terms


def build_inverted_index(docs: dict[int, dict]) -> dict[str, list[int]]:
    """For each term, the list of document IDs where the term appears."""
    inverted_index: dict[str, list[int]] = {}
    for doc_id, doc in docs.items():
        for term in doc["terms"]:
            inverted_index.setdefault(term, []).append(doc_id)
    return inverted_index


def build_term_index(docs: dict[int, dict]) -> dict[str, int]:
    """Sorted vocabulary mapped to column indices: {term: index}."""
    return {t: i for i, t in enumerate(sorted(unique_terms(docs)))}
