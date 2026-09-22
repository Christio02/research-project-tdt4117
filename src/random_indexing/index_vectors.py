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
from random_indexing.config import SEED

import numpy as np


def make_index_vector(
    dim: int, num_nonzeros: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a single sparse ternary random index vector of length `dim`.

    - `num_nonzeros` distinct positions in [0, dim), without replacement
    - each assigned +1 or -1 uniformly at random
    - returned as a dense np.ndarray
    """
    # create vector
    zero_vector = np.zeros(shape=dim, dtype=np.float32)
    # pick uniqueu indexes without replacement
    choice = rng.choice(dim, size=num_nonzeros, replace=False)
    # create -1.0 and 1.0 distirbution
    # From paper.  all zero mean distributions with unit variance
    dist = rng.choice([-1.0, 1.0], size=num_nonzeros, replace=True)

    # assign distribution to zero vector
    zero_vector[choice] = dist
    return zero_vector


def build_index_vectors(
    context_ids: list[int],
    dim: int,
    num_nonzeros: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Generate one index vector per context id (e.g. per document id).

    - return {context_id: index_vector}
    """
    random_generator = np.random.default_rng(seed)
    return {context_id: make_index_vector(dim=dim, num_nonzeros=num_nonzeros, rng=random_generator) for context_id in context_ids}
