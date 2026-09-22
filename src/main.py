from random_indexing.config import DIMENSIONS
from random_indexing.context_vectors import build_context_vectors
from baseline import parse_cranfield, build_tfidf_matrix, build_inverted_index, build_idf, add_terms_to_docs
from random_indexing import build_index_vectors
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    cranfield = parse_cranfield()
    cranfield = add_terms_to_docs(cranfield)

    inverted_index = build_inverted_index(cranfield)
    idfs = build_idf(inverted_index, len(cranfield))
    tfidf_matrix = build_tfidf_matrix(cranfield, idfs)
    index_vectors = build_index_vectors(context_ids=list(cranfield.keys()), dim=DIMENSIONS, num_nonzeros=8, seed=42)
    context_vectors = build_context_vectors(docs=cranfield, index_vectors=index_vectors, dim=DIMENSIONS, weighted=True, tfidf_matrix=tfidf_matrix)

    max = 10
    count = 0
    for term, context_vector in context_vectors.items():
        if count == max:
            break
        print(f"Term: {term} \t  Norm:  {np.linalg.norm(context_vector)} \t Non-zero count: {np.count_nonzero(context_vector)}\n")
        count += 1
