from typing import Any, Protocol

from app.retrieval.interfaces import VectorSearchPort
from app.schemas.retrieval import ChunkMetadata, RetrievalSource, RetrievedChunk


class EmbedderPort(Protocol):
    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into the vector space used by the collection."""
        ...


class QdrantVectorStore(VectorSearchPort):
    def __init__(self, client: Any, collection_name: str, embedder: EmbedderPort) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        query_vector = await self.embedder.embed_query(query)
        query_filter = self._build_filter(branch_path)

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            metadata = ChunkMetadata(
                document_id=payload["document_id"],
                source_uri=payload.get("source_uri"),
                title=payload.get("title"),
                section=payload.get("section"),
                page_number=payload.get("page_number"),
                hierarchy_path=payload.get("hierarchy_path", []),
                token_count=payload.get("token_count", 0),
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    text=payload["text"],
                    score=float(point.score),
                    source=RetrievalSource.VECTOR,
                    metadata=metadata,
                )
            )
        return chunks

    @staticmethod
    def _build_filter(branch_path: list[str]) -> Any | None:
        if not branch_path:
            return None
        return {
            "must": [
                {
                    "key": "hierarchy_path",
                    "match": {"any": branch_path},
                }
            ]
        }
