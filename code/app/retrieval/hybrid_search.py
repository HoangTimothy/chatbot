import asyncio
from collections import defaultdict

from app.retrieval.interfaces import KeywordSearchPort, VectorSearchPort
from app.schemas.retrieval import RetrievalSource, RetrievedChunk


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        self.k = k

    def merge(
        self,
        keyword_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        scores: defaultdict[str, float] = defaultdict(float)

        for ranked_list in (keyword_results, vector_results):
            for rank, chunk in enumerate(ranked_list, start=1):
                by_id.setdefault(chunk.chunk_id, chunk)
                scores[chunk.chunk_id] += 1.0 / (self.k + rank)

        merged = []
        for chunk_id, score in scores.items():
            chunk = by_id[chunk_id].model_copy(update={"score": score, "source": RetrievalSource.HYBRID})
            merged.append(chunk)

        return sorted(merged, key=lambda item: item.score, reverse=True)[:limit]


class HybridRetriever:
    def __init__(
        self,
        keyword_search: KeywordSearchPort,
        vector_search: VectorSearchPort,
        max_candidates: int,
        merger: ReciprocalRankFusion | None = None,
    ) -> None:
        self.keyword_search = keyword_search
        self.vector_search = vector_search
        self.max_candidates = max_candidates
        self.merger = merger or ReciprocalRankFusion()

    async def retrieve(self, query: str, branch_path: list[str]) -> list[RetrievedChunk]:
        per_source_limit = max(self.max_candidates // 2, 5)
        keyword_results, vector_results = await asyncio.gather(
            self.keyword_search.search(query, branch_path, per_source_limit),
            self.vector_search.search(query, branch_path, per_source_limit),
        )
        return self.merger.merge(keyword_results, vector_results, self.max_candidates)

