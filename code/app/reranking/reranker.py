from typing import Protocol

from app.schemas.retrieval import RetrievalSource, RetrievedChunk


class RerankerPort(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by query relevance."""
        ...


class ScorePreservingReranker:
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        reranked = sorted(chunks, key=lambda item: item.score, reverse=True)[:top_k]
        return [chunk.model_copy(update={"source": RetrievalSource.RERANKED}) for chunk in reranked]


class BgeCrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        rescored = [
            chunk.model_copy(update={"score": float(score), "source": RetrievalSource.RERANKED})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
