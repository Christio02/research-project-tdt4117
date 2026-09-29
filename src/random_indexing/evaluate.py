"""Compare Random Indexing retrieval against the Assignment 1 baselines.

Per the project note section 5 ("Applying this on top of Assignment 1"):
  - compare RI top-k vs TF-IDF/cosine top-k and the inverted index, on the
    same Cranfield query set used in the notebook
  - compare memory footprint: |V| x DIMENSIONS (RI) vs |V| x N documents
    (TF-IDF matrix) vs inverted index posting lists
  - look specifically for vocabulary-mismatch cases: queries where RI finds
    a relevant document that shares no literal terms with it, but TF-IDF/
    Boolean retrieval miss
"""

from __future__ import annotations

from typing import Any


def _top_k_doc_ids(
    ranked_list: list[tuple[int, float]], k: int
) -> list[int]:
    """Return up to *k* document IDs from a score-sorted list.

    If the list is not already sorted descending by score, it is sorted first.
    """
    # Ensure descending order by score.
    sorted_list = sorted(ranked_list, key=lambda pair: pair[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_list[:k]]


def compare_top_k(
    ri_results: dict[str, list[tuple[int, float]]],
    tfidf_results: dict[str, list[tuple[int, float]]],
    k: int = 5,
) -> dict[str, Any]:
    """Compare RI vs TF-IDF top-k results per query.

    Returns a mapping ``query_id -> comparison_dict`` where each inner dict
    contains:

    - ``overlap`` (int): number of document IDs present in both top-k sets.
    - ``ri_only`` (list[int]): doc IDs returned by RI but not TF-IDF (sorted).
    - ``tfidf_only`` (list[int]): doc IDs returned by TF-IDF but not RI (sorted).
    - ``vocabulary_mismatch_candidates`` (list[int]): alias for ``ri_only``,
      highlighting potential cases where RI recovers a relevant document that
      shares no literal terms with the query (and which the literal-match
      TF-IDF baseline therefore missed).

    Only query IDs present in *both* result dictionaries are evaluated.
    """
    comparison: dict[str, Any] = {}
    common_queries = ri_results.keys() & tfidf_results.keys()

    for query_id in common_queries:
        ri_top = set(_top_k_doc_ids(ri_results[query_id], k))
        tfidf_top = set(_top_k_doc_ids(tfidf_results[query_id], k))

        overlap = ri_top & tfidf_top
        ri_only = sorted(ri_top - tfidf_top)
        tfidf_only = sorted(tfidf_top - ri_top)

        comparison[query_id] = {
            "overlap": len(overlap),
            "ri_only": ri_only,
            "tfidf_only": tfidf_only,
            "vocabulary_mismatch_candidates": ri_only,
        }

    return comparison


def estimate_memory_bytes(
    vocab_size: int,
    num_docs: int,
    dimensions: int,
) -> dict[str, int]:
    """Rough memory comparison between RI's term table and the TF-IDF matrix.

    Returns a mapping with the following keys:

    - ``ri_float32``: ``vocab_size * dimensions * 4``
    - ``ri_float64``: ``vocab_size * dimensions * 8``
    - ``tfidf_dense``: ``num_docs * vocab_size * 8`` (worst-case dense matrix
      of Python ``float`` objects, i.e. 8 bytes per cell).
    - ``tfidf_sparse_note``: not included as a numeric entry because the actual
      footprint depends heavily on sparsity. In the baseline implementation
      TF-IDF is stored as ``{doc_id: {term: weight}}`` — only non-zero cells
      are kept, so the effective size is typically orders of magnitude smaller
      than the dense estimate above.

    The inverted index is even more compact: it stores one integer posting per
    (term, document) occurrence, i.e. ``number_of_postings * 8`` bytes (plus
    Python container overhead), which is usually far smaller than either dense
    matrix.
    """
    return {
        "ri_float32": vocab_size * dimensions * 4,
        "ri_float64": vocab_size * dimensions * 8,
        "tfidf_dense": num_docs * vocab_size * 8,
    }
