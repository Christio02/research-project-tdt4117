# Random Indexing

Scaffold only -- every function raises `NotImplementedError`. Fill in and
reuse `docs`, `tokenize`, `inverted_index`, and the TF-IDF vectors from
`../baseline` (port of Assignment 1) as the comparison baseline.

## Build order

1. `config.py` -- fixed hyperparameters (`DIMENSIONS`, `NUM_NONZEROS`, `SEED`).
2. `index_vectors.py` -- one sparse random ternary vector per Cranfield document.
3. `context_vectors.py` -- accumulate index vectors into one dense vector per term.
4. `retrieval.py` -- cosine similarity, term-neighbour lookup, document ranking.
5. `evaluate.py` -- compare against Assignment 1's TF-IDF/inverted-index baselines.

## Sources

- Sahlgren, M. (2005). _An Introduction to Random Indexing._
- Kanerva, P., Kristofersson, J., & Holst, A. (2000). _Random Indexing of text samples for LSA._
- Achlioptas, D. (2001). _Database-friendly random projections._

## Open question (not scaffolded)

Space-filling curves (Hilbert curves / geohashes) came up in the same
discussion but solve a low-dimensional geometric indexing problem, not
high-dimensional semantic similarity -- no module here for it until there's
a concrete, low-dimensional use (e.g. geographic/temporal metadata) to
attach it to.
