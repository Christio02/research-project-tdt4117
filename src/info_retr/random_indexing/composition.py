"""Composition functions: combine term vectors into a phrase/query vector.

Reference: Mitchell & Lapata (2010), "Composition in Distributional Models
of Semantics", the framework p = f(u, v, R, K). We ignore syntactic relation
R and background knowledge K for now and only implement the vector-only
model family from their Table (section 2), applied to RI's dense vectors:

    additive:           p_i = u_i + v_i
    weighted additive:  p_i = a*u_i + b*v_i
    multiplicative:     p_i = u_i * v_i          (element-wise)
    dilation:           stretch v along u by factor lambda

Multiplication won best on dense vectors in their evaluation; it is expected
to also beat plain addition here since RI vectors are dense, unlike the
sparse LDA vectors where addition won instead. Circular convolution is
deliberately NOT implemented -- it is built for random/holographic vectors
and performed worst in their results on structured semantic vectors.
"""

from __future__ import annotations

import numpy as np


def compose_additive(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """p = u + v. TODO: implement; this is the baseline, order-insensitive."""
    raise NotImplementedError


def compose_weighted_additive(u: np.ndarray, v: np.ndarray, a: float, b: float) -> np.ndarray:
    """p = a*u + b*v. TODO: implement; lets head/modifier count unequally."""
    raise NotImplementedError


def compose_multiplicative(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """p_i = u_i * v_i (element-wise). TODO: implement.

    Note: this is symmetric (order-insensitive) despite "filtering" rather
    than "averaging" -- don't conflate it with syntax-awareness.
    """
    raise NotImplementedError


def compose_dilation(u: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    """Stretch v along the direction of u by factor `lam`.

    TODO: implement Mitchell & Lapata's dilation model:
        p = (v . v) * u + (lam - 1) * (u . v) * v
    This is the model family that can express asymmetric relations.
    """
    raise NotImplementedError


def compose_query(terms: list[str], context_vectors: dict[str, np.ndarray], method: str = "additive") -> np.ndarray:
    """Compose a multi-term query into a single vector using `method`.

    TODO:
    - look up each term's context vector (skip/handle out-of-vocabulary terms)
    - fold the term vectors together pairwise using the chosen compose_* fn
    - start with "additive" as the default/baseline, then compare against
      "multiplicative" and "dilation" per the project's next-steps
    """
    raise NotImplementedError
