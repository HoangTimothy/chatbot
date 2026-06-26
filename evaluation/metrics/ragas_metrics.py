"""Tier 2 — RAGAS framework integration.

Wraps the ``ragas`` library to compute industry-standard RAG evaluation metrics.
RAGAS is treated as an **optional** dependency — if not installed, this module
gracefully degrades and returns empty scores with a warning.

Metrics wrapped:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
"""

from __future__ import annotations

import logging
from typing import Sequence

from evaluation.dataset import EvalResult

logger = logging.getLogger("evaluation.metrics.ragas")

# ──────────────────────────────────────────────────────────────────────
# Lazy RAGAS import with graceful fallback
# ──────────────────────────────────────────────────────────────────────

_RAGAS_AVAILABLE = False

try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    logger.info(
        "RAGAS package not installed. Tier 2 metrics will be skipped. "
        "Install with: pip install ragas datasets"
    )


def is_available() -> bool:
    """Check whether RAGAS dependencies are installed."""
    return _RAGAS_AVAILABLE


# ──────────────────────────────────────────────────────────────────────
# Single-batch RAGAS evaluation
# ──────────────────────────────────────────────────────────────────────


def compute_ragas_metrics(results: Sequence[EvalResult]) -> dict[str, float]:
    """Run RAGAS evaluation on a batch of EvalResults.

    Each EvalResult must already contain:
    - ``sample.question``
    - ``generated_answer``
    - ``retrieved_texts`` (list of context strings)
    - ``sample.reference_answer``

    Returns:
        Dictionary with averaged RAGAS metric scores.
        Returns zeros with a warning if RAGAS is unavailable.
    """
    empty_scores = {
        "ragas_faithfulness": 0.0,
        "ragas_answer_relevancy": 0.0,
        "ragas_context_precision": 0.0,
        "ragas_context_recall": 0.0,
    }

    if not _RAGAS_AVAILABLE:
        logger.warning("RAGAS is not installed. Returning empty Tier 2 scores.")
        return empty_scores

    if not results:
        return empty_scores

    # Build HuggingFace Dataset in the format RAGAS expects
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for r in results:
        questions.append(r.sample.question)
        answers.append(r.generated_answer)
        contexts.append(r.retrieved_texts if r.retrieved_texts else [""])
        ground_truths.append(r.sample.reference_answer)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    try:
        ragas_result = ragas_evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        scores = {
            "ragas_faithfulness": float(ragas_result.get("faithfulness", 0.0)),
            "ragas_answer_relevancy": float(ragas_result.get("answer_relevancy", 0.0)),
            "ragas_context_precision": float(ragas_result.get("context_precision", 0.0)),
            "ragas_context_recall": float(ragas_result.get("context_recall", 0.0)),
        }

        # Also populate per-sample scores from the dataframe if available
        if hasattr(ragas_result, "to_pandas"):
            df = ragas_result.to_pandas()
            for i, r in enumerate(results):
                if i < len(df):
                    r.scores["ragas_faithfulness"] = float(
                        df.iloc[i].get("faithfulness", 0.0)
                    )
                    r.scores["ragas_answer_relevancy"] = float(
                        df.iloc[i].get("answer_relevancy", 0.0)
                    )
                    r.scores["ragas_context_precision"] = float(
                        df.iloc[i].get("context_precision", 0.0)
                    )
                    r.scores["ragas_context_recall"] = float(
                        df.iloc[i].get("context_recall", 0.0)
                    )

        return scores

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {type(e).__name__}: {e}")
        return empty_scores
