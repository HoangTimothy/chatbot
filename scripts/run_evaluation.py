#!/usr/bin/env python3
"""CLI entry point for running RAG evaluation.

Usage:
    python scripts/run_evaluation.py --dataset evaluation_data/golden_qa.jsonl --tiers 1
    python scripts/run_evaluation.py --dataset evaluation_data/golden_qa.jsonl --tiers 1,2,3 --verbose
    python scripts/run_evaluation.py --tiers 1 --max-samples 10

Options:
    --dataset       Path to golden Q&A JSONL file (default: evaluation_data/golden_qa.jsonl)
    --tiers         Comma-separated tier numbers to run: 1,2,3 (default: 1)
    --output-dir    Directory for JSON reports (default: evaluation_data/results)
    --max-samples   Max samples to evaluate, 0=all (default: 0)
    --retrieval-k   Top-K for retrieval evaluation (default: 10)
    --rerank-limit  Number of reranked candidates (default: 5)
    --judge-provider  LLM provider for Tier 3: openai or google (default: google)
    --judge-model     Model name for Tier 3 judge (default: gemini-2.5-flash)
    --verbose       Print per-sample details
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.config import EvalConfig
from evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Runner — Multi-tier evaluation for RAG chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation_data/golden_qa.jsonl",
        help="Path to the golden Q&A JSONL dataset",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="1",
        help="Comma-separated tier numbers to run (e.g., '1', '1,2', '1,2,3')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_data/results",
        help="Directory for JSON evaluation reports",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Top-K parameter for retrieval evaluation",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=5,
        help="Number of candidates after reranking",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="google",
        choices=["openai", "google"],
        help="LLM provider for Tier 3 judge",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name for Tier 3 LLM judge",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="",
        help="API key for Tier 3 judge (defaults to env GOOGLE_API_KEY or OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample detailed scores",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Enable HyDE query expansion for retrieval",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse tiers
    tiers = [int(t.strip()) for t in args.tiers.split(",") if t.strip()]
    for t in tiers:
        if t not in (1, 2, 3):
            print(f"Error: Invalid tier number: {t}. Must be 1, 2, or 3.")
            sys.exit(1)

    # Build config
    config = EvalConfig(
        tiers=tiers,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
        rerank_limit=args.rerank_limit,
        ragas_enabled=2 in tiers,
        llm_judge_provider=args.judge_provider,
        llm_judge_model=args.judge_model,
        llm_judge_api_key=args.judge_api_key,
        max_samples=args.max_samples,
        verbose=args.verbose,
        hyde_enabled=args.hyde,
    )

    # Run evaluation
    try:
        run_evaluation(config)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print(f"  Hint: Create a golden dataset at '{args.dataset}' before running evaluation.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  [WARN] Evaluation interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
