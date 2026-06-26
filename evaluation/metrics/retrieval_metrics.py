"""Tier 1 — Deterministic retrieval quality metrics.

All metrics here are computed without any LLM calls, making them
fast, reproducible, and suitable for CI/CD pipelines.

Metrics implemented:
- Hit Rate (Recall@K)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Context Precision (Precision@K)
"""

from __future__ import annotations

import math
from typing import Sequence

from evaluation.dataset import EvalResult


# ──────────────────────────────────────────────────────────────────────
# Per-sample metrics
# ──────────────────────────────────────────────────────────────────────


def hit_rate(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Recall@K — fraction of relevant chunks found in the retrieved set.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth relevant chunk IDs.

    Returns:
        Value in [0.0, 1.0].  Returns 0.0 if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for rid in relevant_ids if rid in retrieved_set)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """MRR — reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs (rank 1 = index 0).
        relevant_ids: Ground-truth relevant chunk IDs.

    Returns:
        Value in (0.0, 1.0].  Returns 0.0 if no relevant item is found.
    """
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: a retrieved item is relevant (1) if it appears in
    ``relevant_ids``, otherwise 0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        NDCG@K in [0.0, 1.0].
    """
    relevant_set = set(relevant_ids)

    # DCG: sum of relevance / log2(rank + 1) for positions 1..k
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because ranks start at 1

    # Ideal DCG: all relevant items at the top
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def context_precision(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Precision@K — fraction of retrieved chunks that are actually relevant.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Ground-truth relevant chunk IDs.

    Returns:
        Value in [0.0, 1.0].  Returns 0.0 if nothing was retrieved.
    """
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
    return hits / len(retrieved_ids)


# ──────────────────────────────────────────────────────────────────────
# Batch scoring
# ──────────────────────────────────────────────────────────────────────


def compute_retrieval_metrics(result: EvalResult, k: int = 10) -> dict[str, float]:
    """Compute all retrieval metrics for a single evaluation result.

    Populates ``result.scores`` with the computed values and returns them.
    """
    retrieved = result.retrieved_chunk_ids
    relevant = result.sample.reference_chunk_ids

    scores = {
        "retrieval_hit_rate": hit_rate(retrieved, relevant),
        "retrieval_mrr": mean_reciprocal_rank(retrieved, relevant),
        "retrieval_ndcg": ndcg_at_k(retrieved, relevant, k=k),
        "retrieval_precision": context_precision(retrieved, relevant),
        "retrieval_latency_ms": result.retrieval_latency_ms,
    }

    result.scores.update(scores)
    return scores


def aggregate_retrieval_metrics(results: Sequence[EvalResult]) -> dict[str, float]:
    """Compute average retrieval metrics across all evaluated samples.

    Returns:
        Dictionary of averaged metric values.  Returns zeros if no results.
    """
    if not results:
        return {
            "avg_hit_rate": 0.0,
            "avg_mrr": 0.0,
            "avg_ndcg": 0.0,
            "avg_precision": 0.0,
            "avg_retrieval_latency_ms": 0.0,
        }

    n = len(results)
    return {
        "avg_hit_rate": sum(r.scores.get("retrieval_hit_rate", 0.0) for r in results) / n,
        "avg_mrr": sum(r.scores.get("retrieval_mrr", 0.0) for r in results) / n,
        "avg_ndcg": sum(r.scores.get("retrieval_ndcg", 0.0) for r in results) / n,
        "avg_precision": sum(r.scores.get("retrieval_precision", 0.0) for r in results) / n,
        "avg_retrieval_latency_ms": sum(r.scores.get("retrieval_latency_ms", 0.0) for r in results) / n,
    }
