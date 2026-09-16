"""Term (context) vectors, built by accumulating index vectors.

Reference: Sahlgren (2005) section 3, the core RI algorithm:

    for each document D:
        for each term t in D:
            context_vector[t] += index_vector[D]

This is a single streaming pass -- no full term-document matrix is ever
materialized, unlike LSA. Term weighting (TF, IDF) can be folded into the
accumulation step; see the TODO below and Assignment 1's TF-IDF cells
(src/../Assignment1/assignment1_tdt4117.ipynb, cells 16-20) for weights
already available to reuse.
"""

from __future__ import annotations

import numpy as np


def build_context_vectors(
    docs: dict[int, dict],
    index_vectors: dict[int, np.ndarray],
    dim: int,
    weighted: bool = False,
) -> dict[str, np.ndarray]:
    """Accumulate one context vector per unique term across the corpus.

    `docs` follows the Assignment 1 shape: {doc_id: {"terms": [...], ...}}.

    TODO:
    - initialize a zero vector of length `dim` per unique term
    - for each document, for each term occurrence, add that document's
      index vector to the term's context vector
    - if `weighted` is True, scale the contribution by a term weight
      (e.g. raw tf, or idf from Assignment 1) instead of adding it unweighted
    - decide here whether repeated terms in a doc should add index_vector
      once per occurrence, or once per document (note: Assignment 1's
      tokenizer currently dedupes terms per doc into a set -- revisit if
      per-occurrence frequency is wanted)
    - return {term: context_vector}
    """
    raise NotImplementedError
