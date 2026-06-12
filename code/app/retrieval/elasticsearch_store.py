from typing import Any

from app.retrieval.interfaces import KeywordSearchPort
from app.schemas.retrieval import ChunkMetadata, RetrievalSource, RetrievedChunk


class ElasticsearchKeywordStore(KeywordSearchPort):
    def __init__(self, client: Any, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        filters = []
        if branch_path:
            filters.append({"terms": {"hierarchy_path.keyword": branch_path}})

        response = await self.client.search(
            index=self.index_name,
            size=limit,
            query={
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "section^2", "text"],
                            "type": "best_fields",
                        }
                    },
                    "filter": filters,
                }
            },
        )

        chunks: list[RetrievedChunk] = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            metadata = ChunkMetadata(
                document_id=source["document_id"],
                source_uri=source.get("source_uri"),
                title=source.get("title"),
                section=source.get("section"),
                page_number=source.get("page_number"),
                hierarchy_path=source.get("hierarchy_path", []),
                token_count=source.get("token_count", 0),
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=hit["_id"],
                    text=source["text"],
                    score=float(hit["_score"] or 0.0),
                    source=RetrievalSource.KEYWORD,
                    metadata=metadata,
                )
            )
        return chunks

