"""Term (context) vectors, built by accumulating index vectors.

Reference: Sahlgren (2005) section 3, the core RI algorithm:

    for each document D:
        for each term t in D:
            context_vector[t] += index_vector[D]

This is a single streaming pass -- no full term-document matrix is ever
materialized, unlike LSA. Term weighting (TF, IDF) is folded into the
accumulation step via the optional `tfidf_matrix` argument (built by
`baseline.tfidf.build_tfidf_matrix`).
"""

from __future__ import annotations

import numpy as np


def build_context_vectors(
    docs: dict[int, dict],
    index_vectors: dict[int, np.ndarray],
    dim: int,
    weighted: bool = False,
    tfidf_matrix: dict[int, dict[str, float]] | None = None
) -> dict[str, np.ndarray]:
    """Accumulate one context vector per unique term across the corpus.

    `docs` follows the Assignment 1 shape: {doc_id: {"terms": [...], ...}}.

    - return {term: context_vector}
    """

    context_vectors = {}
    for doc in docs:
        inner_dict = docs[doc]
        terms = inner_dict["terms"]

        for term in terms:
            count = 0

            if count == 1:
                continue

            if term not in context_vectors:
                context_vectors[term] = np.zeros(shape=dim, dtype=np.float32)

            if weighted == True and tfidf_matrix is not None:
                term_tf_idf = tfidf_matrix.get(doc, {}).get(term, 0.0)
                context_vectors[term] += index_vectors[doc] * term_tf_idf

            else:
                context_vectors[term] += index_vectors[doc]

    return context_vectors
