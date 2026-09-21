"""Hyperparameters for Random Indexing.

Values per Sahlgren (2005) / Kanerva, Kristofersson & Holst (2000): d on the
order of thousands, k a small number of non-zeros. Mitchell & Lapata (2010)
use dense d=2048 Gaussian vectors for their own space; we use the sparse
ternary variant (Achlioptas, 2001) instead, so only DIMENSIONS is shared
ground with their setup.

TODO: tune DIMENSIONS / NUM_NONZEROS once retrieval quality can be measured
against the Assignment 1 TF-IDF baseline (see random_indexing/evaluate.py).
"""

# Dimensionality of index vectors and context (term) vectors.
DIMENSIONS = 2048

# Number of non-zero entries per sparse index vector (each +1 or -1).
NUM_NONZEROS = 8

# Random seed for reproducible index vector generation.
SEED = 42
