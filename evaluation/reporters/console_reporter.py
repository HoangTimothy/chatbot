"""Console reporter — pretty-print evaluation results to the terminal.

Uses box-drawing characters and ANSI colors for a clean, readable output.
No external dependencies required (no ``rich`` needed).
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from evaluation.dataset import EvalResult


# ──────────────────────────────────────────────────────────────────────
# ANSI color helpers
# ──────────────────────────────────────────────────────────────────────

_SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap text in ANSI color codes if terminal supports it."""
    if _SUPPORTS_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _red(text: str) -> str:
    return _c("31", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _dim(text: str) -> str:
    return _c("90", text)


def _score_color(value: float, high: float = 0.8, low: float = 0.5) -> str:
    """Color a score green/yellow/red based on thresholds."""
    formatted = f"{value:.4f}"
    if value >= high:
        return _green(formatted)
    elif value >= low:
        return _yellow(formatted)
    else:
        return _red(formatted)


# ──────────────────────────────────────────────────────────────────────
# Table rendering
# ──────────────────────────────────────────────────────────────────────


def _render_table(title: str, rows: list[tuple[str, str]], width: int = 60) -> str:
    """Render a simple two-column table with box-drawing characters."""
    col1_width = max(len(r[0]) for r in rows) + 2 if rows else 20
    col2_width = width - col1_width - 5  # borders + padding

    lines = [
        "",
        _bold(_cyan(f"  ┌─ {title} " + "─" * max(0, width - len(title) - 5) + "┐")),
    ]

    for label, value in rows:
        padded_label = f"  │ {label:<{col1_width}}"
        padded_value = f"{value:>{col2_width}} │"
        lines.append(f"{padded_label}{padded_value}")

    lines.append(_cyan(f"  └{'─' * (width - 2)}┘"))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def print_header(dataset_path: str, sample_count: int, tiers: list[int]) -> None:
    """Print the evaluation run header."""
    print()
    print(_bold("=" * 64))
    print(_bold(_cyan("  [EVAL] RAG Evaluation Report")))
    print(_bold("=" * 64))
    print(f"  Dataset:  {dataset_path}")
    print(f"  Samples:  {sample_count}")
    print(f"  Tiers:    {', '.join(f'Tier {t}' for t in tiers)}")
    print(f"  Time:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(_bold("=" * 64))


def print_tier1_results(
    retrieval_agg: dict[str, float],
    generation_agg: dict[str, float],
) -> None:
    """Print Tier 1 — Component Metrics results."""
    retrieval_rows = [
        ("Hit Rate (Recall@K)", _score_color(retrieval_agg.get("avg_hit_rate", 0.0))),
        ("MRR", _score_color(retrieval_agg.get("avg_mrr", 0.0))),
        ("NDCG@K", _score_color(retrieval_agg.get("avg_ndcg", 0.0))),
        ("Context Precision", _score_color(retrieval_agg.get("avg_precision", 0.0))),
        ("Avg Latency", f"{retrieval_agg.get('avg_retrieval_latency_ms', 0.0):.1f} ms"),
    ]

    generation_rows = [
        ("BLEU", _score_color(generation_agg.get("avg_bleu", 0.0))),
        ("ROUGE-L", _score_color(generation_agg.get("avg_rouge_l", 0.0))),
        ("Token F1", _score_color(generation_agg.get("avg_token_f1", 0.0))),
        ("Citation Accuracy", _score_color(generation_agg.get("avg_citation_accuracy", 0.0))),
        ("Refusal Accuracy", _score_color(generation_agg.get("avg_refusal_accuracy", 0.0))),
        ("Avg Answer Tokens", f"{generation_agg.get('avg_answer_tokens', 0.0):.0f}"),
        ("Avg Latency", f"{generation_agg.get('avg_generation_latency_ms', 0.0):.1f} ms"),
    ]

    print(_render_table("Tier 1 — Retrieval Metrics", retrieval_rows))
    print(_render_table("Tier 1 — Generation Metrics", generation_rows))


def print_tier2_results(ragas_scores: dict[str, float]) -> None:
    """Print Tier 2 — RAGAS Metrics results."""
    rows = [
        ("Faithfulness", _score_color(ragas_scores.get("ragas_faithfulness", 0.0))),
        ("Answer Relevancy", _score_color(ragas_scores.get("ragas_answer_relevancy", 0.0))),
        ("Context Precision", _score_color(ragas_scores.get("ragas_context_precision", 0.0))),
        ("Context Recall", _score_color(ragas_scores.get("ragas_context_recall", 0.0))),
    ]
    print(_render_table("Tier 2 — RAGAS Metrics", rows))


def print_tier3_results(judge_scores: dict[str, float]) -> None:
    """Print Tier 3 — LLM Judge Metrics results."""
    rows = [
        ("Faithfulness", _score_color(judge_scores.get("avg_judge_faithfulness", 0.0))),
        ("Completeness", _score_color(judge_scores.get("avg_judge_completeness", 0.0))),
        ("Language Consistency", _score_color(judge_scores.get("avg_judge_language_consistency", 0.0))),
        ("Formatting Quality", f"{judge_scores.get('avg_judge_formatting_quality', 0.0):.1f} / 5"),
    ]
    print(_render_table("Tier 3 — LLM Judge Metrics", rows))


def print_per_sample_detail(result: EvalResult, index: int) -> None:
    """Print per-sample detailed scores (used when --verbose is set)."""
    print()
    print(_dim(f"  ── Sample {index + 1} ──"))
    print(f"  Q: {result.sample.question[:80]}...")
    print(f"  A: {result.generated_answer[:80]}...")

    # Print all scores
    for key, val in sorted(result.scores.items()):
        if isinstance(val, bool):
            display = _green("✓") if val else _red("✗")
        elif isinstance(val, float):
            display = _score_color(val)
        else:
            display = str(val)
        print(f"     {key:<35} {display}")


def print_footer() -> None:
    """Print the evaluation run footer."""
    print()
    print(_bold("=" * 64))
    print(_bold(_green("  [OK] Evaluation complete.")))
    print(_bold("=" * 64))
    print()


# ──────────────────────────────────────────────────────────────────────
# JSON report export
# ──────────────────────────────────────────────────────────────────────


def save_json_report(
    output_dir: str,
    results: Sequence[EvalResult],
    aggregated: dict[str, Any],
    dataset_path: str,
    tiers: list[int],
) -> str:
    """Save the full evaluation report as a JSON file.

    Returns:
        Path to the saved JSON report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{timestamp}.json"
    filepath = out_dir / filename

    report = {
        "metadata": {
            "dataset": dataset_path,
            "sample_count": len(results),
            "tiers": tiers,
            "timestamp": timestamp,
        },
        "aggregated_scores": aggregated,
        "per_sample": [
            {
                "question": r.sample.question,
                "reference_answer": r.sample.reference_answer,
                "generated_answer": r.generated_answer,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "generated_citations": r.generated_citations,
                "predicted_insufficient": r.predicted_insufficient,
                "scores": {
                    k: v if not isinstance(v, bool) else int(v)
                    for k, v in r.scores.items()
                },
            }
            for r in results
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  [REPORT] JSON report saved to: {_cyan(str(filepath))}")
    return str(filepath)
