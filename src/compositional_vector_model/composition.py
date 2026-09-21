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

from functools import partial, reduce

import numpy as np


def compose_additive(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """p = u + v."""
    return np.add(u, v)


def compose_weighted_additive(
    u: np.ndarray, v: np.ndarray, a: float, b: float
) -> np.ndarray:
    """p = a*u + b*v"""
    return np.add(a * u, b * v)


def compose_multiplicative(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """p_i = u_i * v_i (element-wise).

    this is symmetric (order-insensitive) despite "filtering" rather
    than "averaging"
    """
    return np.multiply(u, v)


def compose_dilation(u: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    """Stretch v along the direction of u by factor `lam`.

        p = (u . u) * v + (lam - 1) * (u . v) * u
    This is the model family that can express asymmetric relations.
    """
    return u @ u * v + ((lam - 1) * (u @ v)) * u


def compose_query(
    terms: list[str],
    context_vectors: dict[str, np.ndarray],
    lam: float | None = None,
    a: float | None = None,
    b: float | None = None,
    method: str = "additive",
) -> np.ndarray:
    """Compose a multi-term query into a single vector using `method`."""

    filtered_vectors = []
    for term in terms:
        if term in context_vectors:
            filtered_vectors.append(context_vectors[term])

    if len(filtered_vectors) == 0:
        raise ValueError("No query terms found in vocabulary")

    if len(filtered_vectors) == 1:
        return filtered_vectors[0]

    if method == "additive":
        result_vector = reduce(compose_additive, filtered_vectors)

    elif method == "multiplicative":
        result_vector = reduce(compose_multiplicative, filtered_vectors)

    elif method == "dilation":
        if lam is None:
            raise ValueError(
                "When method is dilation, lam is required and must be float"
            )
        partial_dilation = partial(compose_dilation, lam=lam)
        result_vector = reduce(partial_dilation, filtered_vectors)

    elif method == "weighted additive":
        if a is None or b is None:
            raise ValueError(
                "a and b must be set as float when method is weighted additive"
            )
        weighted_fn = partial(compose_weighted_additive, a=a, b=b)
        result_vector = reduce(weighted_fn, filtered_vectors)

    else:
        raise ValueError(
            "Wrong method name passed, allowed: additive, multiplicative, "
            "dilation, weighted additive"
        )

    return result_vector