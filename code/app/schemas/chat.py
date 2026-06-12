from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    session_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    chunk_id: str
    document_title: str | None = None
    section: str | None = None
    page_number: int | None = None
    source_uri: str | None = None
    hierarchy_path: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    routed_branch: list[str] = Field(default_factory=list)
    route_confidence: float = 0.0
    insufficient_context: bool = False

