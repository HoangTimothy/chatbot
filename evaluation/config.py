"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Central configuration for an evaluation run.

    Attributes:
        tiers: Which evaluation tiers to execute (1=component, 2=RAGAS, 3=LLM-judge).
        dataset_path: Path to the golden Q&A JSONL file.
        output_dir: Directory for JSON result reports.
        retrieval_k: Number of top-K results to evaluate retrieval against.
        rerank_limit: Number of candidates after reranking.
        ragas_enabled: Whether RAGAS metrics should be computed (requires ``ragas`` package).
        llm_judge_provider: LLM provider for Tier 3 judge ("openai" or "google").
        llm_judge_model: Model name for the judge LLM.
        llm_judge_api_key: API key for the judge LLM provider.
        judge_prompt_path: Path to the judge prompt template markdown file.
        max_samples: Maximum number of samples to evaluate (0 = all).
        verbose: Print per-sample details to console.
    """

    tiers: list[int] = field(default_factory=lambda: [1])
    dataset_path: str = "evaluation_data/golden_qa.jsonl"
    output_dir: str = "evaluation_data/results"
    retrieval_k: int = 10
    rerank_limit: int = 5
    ragas_enabled: bool = False
    llm_judge_provider: str = "google"
    llm_judge_model: str = "gemini-2.5-flash"
    llm_judge_api_key: str = ""
    judge_prompt_path: str = "evaluation/prompts/judge_faithfulness.md"
    max_samples: int = 0
    verbose: bool = False
    hyde_enabled: bool = False
