"""Builds the full Assignment 1 baseline pipeline end-to-end and prints stats.

Run from the src/ directory:  python -m baseline.main
"""

from baseline.corpus import parse_cranfield
from baseline.index import build_inverted_index, build_term_index
from baseline.tokenize import add_terms_to_docs, unique_terms
from baseline.tfidf import (
    build_doc_term_freqs,
    build_idf,
    build_tfidf_matrix,
    compute_doc_norms,
)


def main() -> None:
    docs = parse_cranfield()
    add_terms_to_docs(docs)

    vocab = unique_terms(docs)
    term_index = build_term_index(docs)
    inverted_index = build_inverted_index(docs)

    doc_term_freq = build_doc_term_freqs(docs)
    idf = build_idf(inverted_index, len(docs))
    tfidf_matrix = build_tfidf_matrix(docs, idf)
    doc_norms = compute_doc_norms(tfidf_matrix)

    print(f"Number of unique terms: {len(vocab)}")
    print(f"Inverted index entries: {len(inverted_index)}")
    print(f"TF-IDF matrix: {len(tfidf_matrix)} docs over {len(idf)} terms")
    print(f"Term index size: {len(term_index)}")
    print("Example vector (doc 1, first 5 terms):", list(tfidf_matrix[1].items())[:5])
    print(f"Example norm (doc 1): {doc_norms[1]:.4f}")


if __name__ == "__main__":
    main()
