"""Tests for evaluation metrics."""

from videorag.eval.metrics import hit_at_k, mean_reciprocal_rank, precision_at_k, recall_at_k


def test_hit_at_k():
    """Test Hit@K metric."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "f", "g"}

    assert hit_at_k(retrieved, relevant, k=3) == 1.0  # "c" in top-3
    assert hit_at_k(retrieved, relevant, k=2) == 0.0  # "c" not in top-2
    assert hit_at_k(retrieved, {"x", "y"}, k=5) == 0.0  # No hits


def test_precision_at_k():
    """Test Precision@K metric."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}

    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3  # "a", "c" in top-3
    assert precision_at_k(retrieved, relevant, k=5) == 3 / 5  # All 3 in top-5
    assert precision_at_k(retrieved, {"x"}, k=5) == 0.0  # No relevant


def test_recall_at_k():
    """Test Recall@K metric."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}

    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3  # Found 2 out of 3
    assert recall_at_k(retrieved, relevant, k=5) == 1.0  # Found all 3
    assert recall_at_k(retrieved, {"a"}, k=1) == 1.0  # Found the only one


def test_mrr():
    """Test Mean Reciprocal Rank."""
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0  # Rank 1
    assert mean_reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5  # Rank 2
    assert mean_reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3  # Rank 3
    assert mean_reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0  # Not found
