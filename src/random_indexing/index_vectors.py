"""Sparse random index vectors (the "environmental" / context signatures).

Reference: Kanerva, Kristofersson & Holst (2000); Sahlgren (2005) section 3;
Achlioptas (2001) for the sparse ternary construction.

Each context (here: a Cranfield document) gets one fixed, sparse, randomly
generated vector of dimension `d` with `k` non-zero entries in {-1, +1}, all
other entries zero. These are generated once and never updated -- they are
what makes two contexts "nearly orthogonal" (Hecht-Nielsen, 1994), which is
the property the whole method leans on.
"""

from __future__ import annotations

import numpy as np


def make_index_vector(dim: int, num_nonzeros: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single sparse ternary random index vector of length `dim`.

    TODO:
    - pick `num_nonzeros` distinct positions in [0, dim) without replacement
    - assign each a value of +1 or -1 uniformly at random
    - return a dense np.ndarray (or a sparse vector, if memory becomes an issue
      once this is applied beyond the 1400 Cranfield documents)
    """
    raise NotImplementedError


def build_index_vectors(
    context_ids: list[int],
    dim: int,
    num_nonzeros: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Generate one index vector per context id (e.g. per document id).

    TODO:
    - seed a single np.random.Generator so the whole run is reproducible
    - call make_index_vector once per id in `context_ids`
    - return {context_id: index_vector}
    """
    raise NotImplementedError
