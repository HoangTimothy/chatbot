from enum import StrEnum

from pydantic import BaseModel, Field


class RetrievalSource(StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class ChunkMetadata(BaseModel):
    document_id: str
    source_uri: str | None = None
    title: str | None = None
    section: str | None = None
    page_number: int | None = None
    hierarchy_path: list[str] = Field(default_factory=list)
    token_count: int = 0


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: RetrievalSource
    metadata: ChunkMetadata


class RoutedQuery(BaseModel):
    query: str
    branch_path: list[str]
    confidence: float
    rationale: str

