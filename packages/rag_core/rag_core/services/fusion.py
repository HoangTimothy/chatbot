from typing import Sequence
from collections import defaultdict

from rag_core.contracts.types import RetrievalCandidate, Chunk


def reciprocal_rank_fusion(
    keyword_candidates: Sequence[RetrievalCandidate],
    vector_candidates: Sequence[RetrievalCandidate],
    k: int = 60,
    keyword_weight: float = 0.5,
    vector_weight: float = 1.0
) -> Sequence[RetrievalCandidate]:
    """Merge keyword and vector candidates using Reciprocal Rank Fusion (RRF) algorithm."""
    # Maps chunk_id -> Chunk object
    chunk_map = {}
    
    # Maps chunk_id -> RRF score sum
    rrf_scores = defaultdict(float)

    # 1. Process keyword candidates ranks
    for rank, candidate in enumerate(keyword_candidates, start=1):
        chunk_id = candidate.chunk.chunk_id
        chunk_map[chunk_id] = candidate.chunk
        rrf_scores[chunk_id] += keyword_weight * (1.0 / (k + rank))

    # 2. Process vector candidates ranks
    for rank, candidate in enumerate(vector_candidates, start=1):
        chunk_id = candidate.chunk.chunk_id
        chunk_map[chunk_id] = candidate.chunk
        rrf_scores[chunk_id] += vector_weight * (1.0 / (k + rank))

    # 3. Assemble and sort merged candidates by RRF score descending
    merged_candidates = []
    for chunk_id, score in rrf_scores.items():
        merged_candidates.append(
            RetrievalCandidate(
                chunk=chunk_map[chunk_id],
                score=score,
                source="hybrid_rrf"
            )
        )

    # Sort descending by RRF score
    merged_candidates.sort(key=lambda x: x.score, reverse=True)
    return merged_candidates


# ---------------------------------------------------------------------------
# Generalized multi-source fusion
# ---------------------------------------------------------------------------

# Default weights for each source type
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "keyword": 0.5,
    "vector": 1.0,
    "web_search": 0.7,
    "knowledge_graph": 0.8,
}


def multi_source_fusion(
    candidate_lists: dict[str, Sequence[RetrievalCandidate]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> Sequence[RetrievalCandidate]:
    """Generalized Reciprocal Rank Fusion for N named sources.

    Args:
        candidate_lists: Mapping of source name → ranked candidate list.
            e.g. {"keyword": [...], "vector": [...], "web_search": [...]}
        weights: Per-source weight multiplier. Defaults to DEFAULT_SOURCE_WEIGHTS.
        k: RRF smoothing constant (default 60).

    Returns:
        Merged candidate list sorted by weighted RRF score descending.
    """
    effective_weights = dict(DEFAULT_SOURCE_WEIGHTS)
    if weights:
        effective_weights.update(weights)

    chunk_map: dict[str, "Chunk"] = {}
    source_map: dict[str, str] = {}   # chunk_id → first source seen
    rrf_scores: dict[str, float] = defaultdict(float)

    for source_name, candidates in candidate_lists.items():
        weight = effective_weights.get(source_name, 1.0)
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate.chunk.chunk_id
            chunk_map[chunk_id] = candidate.chunk
            rrf_scores[chunk_id] += weight * (1.0 / (k + rank))
            if chunk_id not in source_map:
                source_map[chunk_id] = candidate.source

    merged: list[RetrievalCandidate] = []
    for chunk_id, score in rrf_scores.items():
        merged.append(
            RetrievalCandidate(
                chunk=chunk_map[chunk_id],
                score=score,
                source=source_map.get(chunk_id, "multi_source_rrf"),
            )
        )

    merged.sort(key=lambda x: x.score, reverse=True)
    return merged
