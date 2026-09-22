# Assignment 1 Baseline

Port of `Assignment1/assignment1_tdt4117.ipynb` into reusable modules, so the
TF-IDF baseline and the Random Indexing engine (`src/random_indexing`) share
one implementation of corpus parsing, tokenization, and the inverted index.

## Contents

- `corpus.py` — `parse_cranfield` → `{doc_id: {"id", "title", "abstract"}}`
- `tokenize.py` — `tokenize`, stopword list, `add_terms_to_docs` (adds deduped
  `terms` per doc), `unique_terms` (vocabulary)
- `index.py` — `build_inverted_index` ({term: [doc_ids]}), `build_term_index`
  (sorted vocab → column indices)
- `tfidf.py` — `term_freq`, `build_doc_term_freqs`, `build_idf`
  (log(N/df)), `build_tfidf_matrix` ({doc_id: {term: tfidf}}),
  `compute_doc_norms`
- `main.py` — runs the whole pipeline and prints stats

## Usage

```python
from baseline import (
    parse_cranfield, add_terms_to_docs, build_inverted_index,
    build_idf, build_tfidf_matrix,
)

docs = parse_cranfield()          # reads data/cran.all.1400
add_terms_to_docs(docs)           # docs[d]["terms"] = [...]
inverted_index = build_inverted_index(docs)
idf = build_idf(inverted_index, len(docs))
tfidf_matrix = build_tfidf_matrix(docs, idf)
```

Run the full pipeline: `python -m baseline.main` (from `src/`).

Note: `terms` per doc is deduped (as in Assignment 1); TF counts are computed
by re-tokenizing in `tfidf.term_freq`, so occurrence counts are unaffected.
