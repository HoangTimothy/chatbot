"""Tier 1 — Deterministic generation quality metrics.

All metrics here are computed without any LLM calls using
string/token overlap comparisons.

Metrics implemented:
- BLEU (via simple n-gram implementation — no heavy dependency)
- ROUGE-L (Longest Common Subsequence)
- Token Overlap F1
- Citation Accuracy
- Refusal Accuracy
- Answer Length (token count)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from evaluation.dataset import EvalResult


# ──────────────────────────────────────────────────────────────────────
# Tokeniser helper
# ──────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser that works for both Vietnamese and English."""
    return re.findall(r"\w+", text.lower())


# ──────────────────────────────────────────────────────────────────────
# BLEU (simplified corpus-free implementation)
# ──────────────────────────────────────────────────────────────────────


def _count_ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """Compute sentence-level BLEU score (modified precision with brevity penalty).

    Uses uniform weights across n-gram orders 1..max_n.

    Args:
        prediction: Model generated answer.
        reference: Ground-truth reference answer.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Clipped precision per n-gram order
    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = _count_ngrams(pred_tokens, n)
        ref_ngrams = _count_ngrams(ref_tokens, n)

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)

        precision = clipped / total
        if precision == 0:
            return 0.0  # Any zero precision kills BLEU
        log_precisions.append(math.log(precision))

    # Geometric mean of precisions
    avg_log_precision = sum(log_precisions) / len(log_precisions)

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))

    return bp * math.exp(avg_log_precision)


# ──────────────────────────────────────────────────────────────────────
# ROUGE-L
# ──────────────────────────────────────────────────────────────────────


def _lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    """Compute length of the longest common subsequence using DP."""
    m, n = len(x), len(y)
    # Space-optimised: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on Longest Common Subsequence.

    Args:
        prediction: Model generated answer.
        reference: Ground-truth reference answer.

    Returns:
        ROUGE-L F1 in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# Token Overlap F1
# ──────────────────────────────────────────────────────────────────────


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score (unigram overlap).

    Args:
        prediction: Model generated answer.
        reference: Ground-truth reference answer.

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# Citation Accuracy
# ──────────────────────────────────────────────────────────────────────


def citation_accuracy(
    generated_citations: Sequence[str],
    context_chunk_ids: Sequence[str],
) -> float:
    """Fraction of model-generated citations that actually exist in the provided context.

    Args:
        generated_citations: Chunk IDs the model cited.
        context_chunk_ids: Chunk IDs that were actually in the context window.

    Returns:
        Accuracy in [0.0, 1.0].  Returns 1.0 if no citations were generated
        (no wrong citations = perfect).
    """
    if not generated_citations:
        return 1.0  # vacuously true: nothing was cited incorrectly
    valid_set = set(context_chunk_ids)
    valid = sum(1 for c in generated_citations if c in valid_set)
    return valid / len(generated_citations)


# ──────────────────────────────────────────────────────────────────────
# Refusal Accuracy
# ──────────────────────────────────────────────────────────────────────


def refusal_accuracy(predicted_insufficient: bool, expected_insufficient: bool) -> bool:
    """Check whether the system correctly refused or answered.

    Returns:
        True if the prediction matches the expectation.
    """
    return predicted_insufficient == expected_insufficient


# ──────────────────────────────────────────────────────────────────────
# Batch scoring
# ──────────────────────────────────────────────────────────────────────


def compute_generation_metrics(result: EvalResult) -> dict[str, float | bool]:
    """Compute all generation metrics for a single evaluation result.

    Populates ``result.scores`` with the computed values and returns them.
    """
    pred = result.generated_answer
    ref = result.sample.reference_answer

    scores: dict[str, float | bool] = {
        "generation_bleu": bleu_score(pred, ref),
        "generation_rouge_l": rouge_l(pred, ref),
        "generation_token_f1": token_f1(pred, ref),
        "generation_citation_accuracy": citation_accuracy(
            result.generated_citations, result.retrieved_chunk_ids
        ),
        "generation_refusal_correct": refusal_accuracy(
            result.predicted_insufficient, result.sample.expected_insufficient
        ),
        "generation_answer_tokens": float(len(_tokenize(pred))),
        "generation_latency_ms": result.generation_latency_ms,
    }

    result.scores.update(scores)
    return scores


def aggregate_generation_metrics(results: Sequence[EvalResult]) -> dict[str, float]:
    """Compute average generation metrics across all evaluated samples."""
    if not results:
        return {
            "avg_bleu": 0.0,
            "avg_rouge_l": 0.0,
            "avg_token_f1": 0.0,
            "avg_citation_accuracy": 0.0,
            "avg_refusal_accuracy": 0.0,
            "avg_answer_tokens": 0.0,
            "avg_generation_latency_ms": 0.0,
        }

    n = len(results)
    refusal_correct_count = sum(
        1 for r in results if r.scores.get("generation_refusal_correct", False)
    )

    return {
        "avg_bleu": sum(r.scores.get("generation_bleu", 0.0) for r in results) / n,
        "avg_rouge_l": sum(r.scores.get("generation_rouge_l", 0.0) for r in results) / n,
        "avg_token_f1": sum(r.scores.get("generation_token_f1", 0.0) for r in results) / n,
        "avg_citation_accuracy": sum(
            r.scores.get("generation_citation_accuracy", 0.0) for r in results
        ) / n,
        "avg_refusal_accuracy": refusal_correct_count / n,
        "avg_answer_tokens": sum(
            r.scores.get("generation_answer_tokens", 0.0) for r in results
        ) / n,
        "avg_generation_latency_ms": sum(
            r.scores.get("generation_latency_ms", 0.0) for r in results
        ) / n,
    }
