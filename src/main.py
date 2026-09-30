"""Comprehensive experiment runner comparing Random Indexing against Baseline TF-IDF.

Generates a structured report covering:
1. Memory footprint & theoretical scaling
2. Semantic nearest-neighbors (distributional semantics)
3. Top-k retrieval comparison on 10 standard Cranfield queries
4. Vocabulary mismatch case study
5. Mitchell & Lapata composition method comparison
"""

from __future__ import annotations

from baseline import (
    add_terms_to_docs,
    build_idf,
    build_inverted_index,
    build_tfidf_matrix,
    compute_doc_norms,
    parse_cranfield,
    tfidf_retrieve,
)
from random_indexing.config import DIMENSIONS, NUM_NONZEROS, SEED
from random_indexing.context_vectors import build_context_vectors
from random_indexing.evaluate import (
    compare_top_k,
    estimate_memory_bytes,
    precision_at_k,
    recall_at_k,
)
from random_indexing.index_vectors import build_index_vectors
from random_indexing.retrieval import (
    build_document_vectors,
    nearest_terms,
    ri_retrieve,
)

# 10 standard Cranfield queries from Assignment 1
TEST_QUERIES = [
    "gas pressure",
    "structural aeroelastic flight high speed aircraft",
    "heat conduction composite slabs",
    "boundary layer control",
    "compressible flow nozzle",
    "combustion chamber injection",
    "laminar turbulent transition",
    "fatigue crack growth",
    "wing tip vortices",
    "propulsion efficiency",
]


def parse_cran_queries(path: str = "data/cran.qry") -> dict[int, str]:
    """Parse queries from cran.qry into {query_id: query_text}."""
    queries: dict[int, str] = {}
    current_id = None
    buf: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(".I "):
                if current_id is not None:
                    queries[current_id] = " ".join(buf).strip()
                current_id = int(line.split()[1])
                buf = []
            elif line.startswith("."):
                continue
            else:
                buf.append(line)
        if current_id is not None:
            queries[current_id] = " ".join(buf).strip()
    return queries


def parse_cranqrel(path: str = "data/cranqrel") -> dict[int, set[int]]:
    """Parse relevance judgments from cranqrel into {query_id: set_of_relevant_doc_ids}."""
    qrels: dict[int, set[int]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = int(parts[0])
                doc_id = int(parts[1])
                grade = int(parts[2]) if len(parts) >= 3 else 1
                if grade in (1, 2, 3, 4):
                    qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def print_separator(title: str = "") -> None:
    print("\n" + "=" * 80)
    if title:
        print(f"  {title.upper()}")
        print("=" * 80)


def main() -> None:
    print_separator("Cranfield Experiment: Random Indexing vs Baseline TF-IDF")

    # Step 1: Ingestion & Baseline Indexing
    print("[1/5] Ingesting Cranfield corpus & building baseline indices...")
    docs = parse_cranfield()
    add_terms_to_docs(docs)
    num_docs = len(docs)

    inverted_index = build_inverted_index(docs)
    vocab_size = len(inverted_index)
    idf = build_idf(inverted_index, num_docs)
    tfidf_matrix = build_tfidf_matrix(docs, idf)
    doc_norms = compute_doc_norms(tfidf_matrix)

    print(f"      Corpus size: {num_docs} documents")
    print(f"      Vocabulary size: {vocab_size} unique terms")

    # Step 2: Random Indexing Construction
    print("\n[2/5] Constructing Random Indexing semantic space...")
    print(
        f"      Hyperparameters: DIMENSIONS={DIMENSIONS}, NONZEROS={NUM_NONZEROS}, SEED={SEED}"
    )
    index_vectors = build_index_vectors(
        context_ids=list(docs.keys()),
        dim=DIMENSIONS,
        num_nonzeros=NUM_NONZEROS,
        seed=SEED,
    )
    context_vectors = build_context_vectors(
        docs=docs,
        index_vectors=index_vectors,
        dim=DIMENSIONS,
        weighted=True,
        tfidf_matrix=tfidf_matrix,
    )
    doc_vectors_add = build_document_vectors(docs, context_vectors, method="additive")
    print(f"      Built context vectors for {len(context_vectors)} terms")
    print(f"      Composed document vectors for {len(doc_vectors_add)} documents")

    # Experiment 1: Memory Footprint & Scaling Analysis
    print_separator("Experiment 1: Memory Footprint & Scaling Analysis")
    mem = estimate_memory_bytes(
        vocab_size=vocab_size, num_docs=num_docs, dimensions=DIMENSIONS
    )
    ri_mb = mem["ri_float32"] / (1024 * 1024)
    tfidf_dense_mb = mem["tfidf_dense"] / (1024 * 1024)

    print(f"  Vocabulary size (|V|):  {vocab_size:,}")
    print(f"  Number of documents (N): {num_docs:,}")
    print(f"  RI Term Table (float32): {ri_mb:.2f} MB   [fixed O(|V| * d)]")
    print(f"  TF-IDF Dense Matrix:    {tfidf_dense_mb:.2f} MB   [scales O(|V| * N)]")
    print("\n  Scaling takeaway:")
    print("  - TF-IDF dense matrix scales linearly with document count N.")
    print("    At 1,000,000 documents, dense TF-IDF requires ~65,000 MB (65 GB).")
    print("  - Random Indexing term vectors remain strictly constant at ~67 MB")
    print("    regardless of how many documents are indexed.")

    # Experiment 2: Semantic Nearest Neighbors (Distributional Semantics)
    print_separator("Experiment 2: Learned Semantic Neighborhoods")
    sample_terms = ["slipstream", "boundary-layer", "supersonic", "flutter"]
    print("  Demonstration that Random Indexing learns domain semantics:\n")
    for term in sample_terms:
        if term not in context_vectors:
            continue
        neighbors = nearest_terms(term, context_vectors, top_k=5)
        neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
        print(f"  Term: '{term:15s}' => Nearest: {neighbor_str}")

    # Experiment 3: Batch Retrieval Comparison (Top-5 Overlap)
    print_separator("Experiment 3: Batch Retrieval Comparison on 10 Queries (Top-5)")
    ri_results: dict[str, list[tuple[int, float]]] = {}
    tfidf_results: dict[str, list[tuple[int, float]]] = {}

    for i, q in enumerate(TEST_QUERIES, 1):
        qid = f"Q{i:02d}"
        ri_results[qid] = ri_retrieve(
            q, context_vectors, doc_vectors_add, method="additive"
        )
        tfidf_results[qid] = tfidf_retrieve(q, idf, tfidf_matrix, doc_norms)

    comparison = compare_top_k(ri_results, tfidf_results, k=5)

    print(
        f"  {'Query ID':8s} | {'Query String':40s} | {'Overlap':7s} | {'Rel. P@5':8s} | {'RI Top-5 Docs':20s} | {'TF-IDF Top-5 Docs'}"
    )
    print("  " + "-" * 120)

    total_overlap = 0
    for i, q in enumerate(TEST_QUERIES, 1):
        qid = f"Q{i:02d}"
        comp = comparison[qid]
        overlap_count = comp["overlap"]
        total_overlap += overlap_count
        rel_p = overlap_count / 5.0

        ri_top_ids = [d for d, _ in ri_results[qid][:5]]
        tf_top_ids = [d for d, _ in tfidf_results[qid][:5]]

        print(
            f"  {qid:8s} | {q[:40]:40s} | {overlap_count}/5     | {rel_p * 100:5.1f}%   | "
            f"{ri_top_ids!s:20s} | {tf_top_ids!s}"
        )

    avg_overlap = total_overlap / len(TEST_QUERIES)
    print("  " + "-" * 120)
    print(
        f"  Average Agreement with TF-IDF: {avg_overlap:.2f} / 5 documents ({avg_overlap * 20:.1f}%)"
    )

    # Experiment 3b: Ground-Truth Precision & Recall on Cranfield Benchmark (cranqrel)
    print_separator("Experiment 3b: Ground-Truth Precision & Recall (Cranfield Benchmark)")
    cran_queries = parse_cran_queries()
    qrels = parse_cranqrel()
    k_eval = 5

    ri_prec_list, ri_rec_list = [], []
    tf_prec_list, tf_rec_list = [], []

    print(
        f"  Evaluating first 10 Cranfield benchmark queries with human relevance judgments (k={k_eval}):\n"
    )
    print(
        f"  {'QID':5s} | {'Relevant':8s} | {'RI P@5':8s} | {'RI R@5':8s} | {'TF-IDF P@5':10s} | {'TF-IDF R@5':10s} | Query Snippet"
    )
    print("  " + "-" * 110)

    eval_count = 0
    for qid in sorted(cran_queries.keys()):
        if qid not in qrels or len(qrels[qid]) == 0:
            continue
        if eval_count >= 10:
            break

        qtext = cran_queries[qid]
        rel_set = qrels[qid]

        ri_res_b = ri_retrieve(
            qtext, context_vectors, doc_vectors_add, method="additive"
        )
        tf_res_b = tfidf_retrieve(qtext, idf, tfidf_matrix, doc_norms)

        ri_p = precision_at_k(ri_res_b, rel_set, k=k_eval)
        ri_r = recall_at_k(ri_res_b, rel_set, k=k_eval)
        tf_p = precision_at_k(tf_res_b, rel_set, k=k_eval)
        tf_r = recall_at_k(tf_res_b, rel_set, k=k_eval)

        ri_prec_list.append(ri_p)
        ri_rec_list.append(ri_r)
        tf_prec_list.append(tf_p)
        tf_rec_list.append(tf_r)

        print(
            f"  Q{qid:03d} | {len(rel_set):5d}    | {ri_p * 100:5.1f}%  | {ri_r * 100:5.1f}%  | "
            f"{tf_p * 100:7.1f}%  | {tf_r * 100:7.1f}%  | {qtext[:40]}..."
        )
        eval_count += 1

    mean_ri_p = sum(ri_prec_list) / len(ri_prec_list)
    mean_ri_r = sum(ri_rec_list) / len(ri_rec_list)
    mean_tf_p = sum(tf_prec_list) / len(tf_prec_list)
    mean_tf_r = sum(tf_rec_list) / len(tf_rec_list)

    print("  " + "-" * 110)
    print(
        f"  Random Indexing: Mean P@5 = {mean_ri_p * 100:.1f}% | Mean R@5 = {mean_ri_r * 100:.1f}%"
    )
    print(
        f"  Baseline TF-IDF: Mean P@5 = {mean_tf_p * 100:.1f}% | Mean R@5 = {mean_tf_r * 100:.1f}%"
    )

    # Experiment 4: Vocabulary Mismatch Case Study
    print_separator("Experiment 4: Vocabulary Mismatch Case Study")
    print(
        "  Looking for documents retrieved in RI's Top-5 that TF-IDF missed entirely (score = 0)\n"
    )

    mismatch_found = False
    for i, q in enumerate(TEST_QUERIES, 1):
        qid = f"Q{i:02d}"
        q_tokens = set(q.lower().split())

        for doc_id, ri_score in ri_results[qid][:5]:
            doc_terms = set(docs[doc_id]["terms"])
            shared_terms = q_tokens & doc_terms

            # If document shares 0 literal query terms
            if len(shared_terms) == 0:
                print(f'  Query: "{q}"')
                print(
                    f"  --> Found Document {doc_id} with score {ri_score:.4f} (shares 0 literal query terms!)"
                )
                print(f"      Title: {docs[doc_id]['title']}")
                print(f"      Abstract snippet: {docs[doc_id]['abstract'][:160]}...")
                mismatch_found = True
                break
        if mismatch_found:
            break

    if not mismatch_found:
        # Fall back to first document unique to RI
        for i, q in enumerate(TEST_QUERIES, 1):
            qid = f"Q{i:02d}"
            comp = comparison[qid]
            if comp["ri_only"]:
                cand_id = comp["ri_only"][0]
                print(f'  Query: "{q}"')
                print(f"  --> RI-only Top-5 Candidate: Document {cand_id}")
                print(f"      Title: {docs[cand_id]['title']}")
                print(f"      Abstract snippet: {docs[cand_id]['abstract'][:160]}...")
                break

    # Experiment 5: Mitchell & Lapata Composition Function Comparison
    print_separator("Experiment 5: Mitchell & Lapata Query Composition Models")
    comp_query = "structural aeroelastic flight"
    print(f'  Query: "{comp_query}"\n')
    print("  Comparing how different query composition models alter document ranking:")

    res_additive = ri_retrieve(
        comp_query, context_vectors, doc_vectors_add, method="additive"
    )
    res_dilation = ri_retrieve(
        comp_query, context_vectors, doc_vectors_add, method="dilation", lam=2.0
    )

    print(
        f"  {'Rank':4s} | {'Additive (p = u + v)':45s} | {'Dilation (stretch along u, λ=2.0)':45s}"
    )
    print("  " + "-" * 100)
    for rank in range(3):
        add_doc = res_additive[rank][0] if rank < len(res_additive) else "N/A"
        dil_doc = res_dilation[rank][0] if rank < len(res_dilation) else "N/A"
        add_title = docs[add_doc]["title"][:42] if add_doc != "N/A" else ""
        dil_title = docs[dil_doc]["title"][:42] if dil_doc != "N/A" else ""
        print(
            f"  #{rank + 1:<3d} | Doc {add_doc}: {add_title:35s} | Doc {dil_doc}: {dil_title:35s}"
        )

    print_separator("End of Report")


if __name__ == "__main__":
    main()
