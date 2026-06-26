"""Golden evaluation dataset schema and loader.

Each line in the JSONL dataset represents one evaluation sample containing a
question, reference answer, relevant chunk IDs, and metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("evaluation.dataset")


@dataclass
class EvalSample:
    """A single evaluation sample (golden Q&A pair).

    Attributes:
        question: The user question to evaluate.
        reference_answer: The expected ground-truth answer.
        reference_chunk_ids: Chunk IDs that contain the correct information.
        expected_insufficient: Whether the system should refuse (insufficient context).
        metadata: Arbitrary tags — domain, difficulty, language, etc.
    """

    question: str
    reference_answer: str
    reference_chunk_ids: list[str] = field(default_factory=list)
    expected_insufficient: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Container for one sample's evaluation outputs across all pipeline stages.

    Populated by the evaluation runner as it executes each pipeline step.
    """

    sample: EvalSample
    # Retrieval outputs
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    # Generation outputs
    generated_answer: str = ""
    generated_citations: list[str] = field(default_factory=list)
    predicted_insufficient: bool = False
    generation_latency_ms: float = 0.0
    # Context string passed to generator
    context_text: str = ""
    # Metric scores (populated by metric calculators)
    scores: dict[str, float | bool | str] = field(default_factory=dict)


def load_dataset(path: str | Path, max_samples: int = 0) -> Sequence[EvalSample]:
    """Load a golden evaluation dataset from a JSONL file.

    Args:
        path: Path to the .jsonl file.
        max_samples: Maximum samples to load (0 = all).

    Returns:
        List of EvalSample instances.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    samples: list[EvalSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_no}: {e}")
                continue

            sample = EvalSample(
                question=data["question"],
                reference_answer=data.get("reference_answer", ""),
                reference_chunk_ids=data.get("reference_chunk_ids", []),
                expected_insufficient=data.get("expected_insufficient", False),
                metadata=data.get("metadata", {}),
            )
            samples.append(sample)

            if max_samples > 0 and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} evaluation samples from {path}")
    return samples
