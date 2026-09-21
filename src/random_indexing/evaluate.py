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


def compare_top_k(
    ri_results: dict[str, list[tuple[int, float]]],
    tfidf_results: dict[str, list[tuple[int, float]]],
    k: int = 5,
) -> dict[str, dict]:
    """Compare RI vs TF-IDF top-k results per query.

    TODO:
    - for each query id present in both result sets, take the top-k doc ids
    - report overlap (intersection size), and which doc ids are RI-only /
      TF-IDF-only
    - surface RI-only hits as candidate vocabulary-mismatch examples
    """
    raise NotImplementedError


def estimate_memory_bytes(
    vocab_size: int,
    num_docs: int,
    dimensions: int,
) -> dict[str, int]:
    """Rough memory comparison between RI's term table and the TF-IDF matrix.

    TODO:
    - RI: vocab_size * dimensions * 8 bytes (float64) or 4 (float32)
    - TF-IDF dense matrix: num_docs * vocab_size * 8 bytes
    - note TF-IDF is normally stored sparse in practice; call that out in
      the comparison rather than only comparing worst-case dense sizes
    """
    raise NotImplementedError
