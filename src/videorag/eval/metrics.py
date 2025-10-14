"""Evaluation metrics for VideoRAG retrieval quality."""
from typing import Dict, List, Set


def hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Hit@K: whether any relevant item appears in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if len(top_k & relevant_ids) > 0 else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Precision@K: fraction of top-k that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision score [0.0, 1.0]
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    num_relevant = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return num_relevant / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Recall@K: fraction of relevant items in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall score [0.0, 1.0]
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    num_retrieved_relevant = len(top_k & relevant_ids)
    return num_retrieved_relevant / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Compute MRR: reciprocal rank of first relevant item.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs

    Returns:
        MRR score [0.0, 1.0]
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute all metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k_values: List of k values to evaluate

    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}

    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)

    return metrics
