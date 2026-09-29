"""Tests for random_indexing.evaluate module."""

import pytest

from random_indexing.evaluate import compare_top_k, estimate_memory_bytes


class TestCompareTopK:
    """Tests for compare_top_k."""

    def test_empty_common_queries(self) -> None:
        """If there are no common query IDs, result should be empty."""
        ri_results = {"q1": [(1, 0.9), (2, 0.8)]}
        tfidf_results = {"q2": [(1, 0.9), (3, 0.7)]}
        assert compare_top_k(ri_results, tfidf_results, k=2) == {}

    def test_perfect_overlap(self) -> None:
        """Identical ranked lists produce full overlap and no exclusives."""
        ri_results = {"q1": [(1, 0.9), (2, 0.8), (3, 0.7)]}
        tfidf_results = {"q1": [(1, 0.9), (2, 0.8), (3, 0.7)]}

        result = compare_top_k(ri_results, tfidf_results, k=2)
        assert result["q1"]["overlap"] == 2
        assert result["q1"]["ri_only"] == []
        assert result["q1"]["tfidf_only"] == []
        assert result["q1"]["vocabulary_mismatch_candidates"] == []

    def test_ri_only(self) -> None:
        """RI-only hits are surfaced as vocabulary-mismatch candidates."""
        ri_results = {"q1": [(1, 0.9), (4, 0.85), (5, 0.8)]}
        tfidf_results = {"q1": [(1, 0.9), (2, 0.85), (3, 0.8)]}

        result = compare_top_k(ri_results, tfidf_results, k=3)
        assert result["q1"]["overlap"] == 1
        assert result["q1"]["ri_only"] == [4, 5]
        assert result["q1"]["tfidf_only"] == [2, 3]
        assert result["q1"]["vocabulary_mismatch_candidates"] == [4, 5]

    def test_k_truncation(self) -> None:
        """Only the top-k doc IDs are considered from longer lists."""
        ri_results = {"q1": [(1, 0.9), (2, 0.8), (3, 0.7)]}
        tfidf_results = {"q1": [(1, 0.9), (2, 0.8), (4, 0.7)]}

        # With k=2, doc_id 3 and 4 never enter the comparison.
        result = compare_top_k(ri_results, tfidf_results, k=2)
        assert result["q1"]["overlap"] == 2
        assert result["q1"]["ri_only"] == []
        assert result["q1"]["tfidf_only"] == []

    def test_unsorted_input(self) -> None:
        """Inner lists are re-sorted by score before slicing top-k."""
        ri_results = {"q1": [(3, 0.5), (1, 0.9), (2, 0.8)]}
        tfidf_results = {"q1": [(2, 0.8), (1, 0.9), (4, 0.3)]}

        result = compare_top_k(ri_results, tfidf_results, k=2)
        assert result["q1"]["overlap"] == 2  # 1 and 2 both in top-2
        assert result["q1"]["ri_only"] == []
        assert result["q1"]["tfidf_only"] == []

    def test_multiple_queries(self) -> None:
        """Multiple common queries each get their own comparison dict."""
        ri_results = {
            "q1": [(1, 0.9), (2, 0.8)],
            "q2": [(10, 0.9)],
            "q3": [(7, 0.5)],
        }
        tfidf_results = {
            "q1": [(1, 0.9), (3, 0.7)],
            "q2": [(10, 0.9)],
            "q4": [(99, 0.1)],
        }

        result = compare_top_k(ri_results, tfidf_results, k=2)
        assert set(result.keys()) == {"q1", "q2"}
        assert result["q1"]["overlap"] == 1
        assert result["q2"]["overlap"] == 1


class TestEstimateMemoryBytes:
    """Tests for estimate_memory_bytes."""

    def test_typical_values(self) -> None:
        """Smoke test with plausible Cranfield-ish numbers."""
        mem = estimate_memory_bytes(vocab_size=5000, num_docs=1400, dimensions=2048)
        assert mem["ri_float32"] == 5000 * 2048 * 4
        assert mem["ri_float64"] == 5000 * 2048 * 8
        assert mem["tfidf_dense"] == 1400 * 5000 * 8

    def test_zero_dimensions(self) -> None:
        """Edge case where dimensions is zero."""
        mem = estimate_memory_bytes(vocab_size=100, num_docs=50, dimensions=0)
        assert mem["ri_float32"] == 0
        assert mem["ri_float64"] == 0
        assert mem["tfidf_dense"] == 100 * 50 * 8

    def test_all_zero(self) -> None:
        """All-zero input gives all-zero outputs."""
        mem = estimate_memory_bytes(vocab_size=0, num_docs=0, dimensions=0)
        assert mem == {
            "ri_float32": 0,
            "ri_float64": 0,
            "tfidf_dense": 0,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
