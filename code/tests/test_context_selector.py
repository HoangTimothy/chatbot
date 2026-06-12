from app.retrieval.context_selector import TokenBudgetContextSelector
from app.schemas.retrieval import ChunkMetadata, RetrievalSource, RetrievedChunk


def make_chunk(chunk_id: str, score: float, token_count: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="x " * token_count,
        score=score,
        source=RetrievalSource.RERANKED,
        metadata=ChunkMetadata(document_id="doc", token_count=token_count),
    )


def test_context_selector_respects_chunk_and_token_limits() -> None:
    selector = TokenBudgetContextSelector(max_chunks=2, max_tokens=100)
    chunks = [
        make_chunk("a", 0.9, 50),
        make_chunk("b", 0.8, 60),
        make_chunk("c", 0.7, 20),
    ]

    selected = selector.select(chunks)

    assert [chunk.chunk_id for chunk in selected] == ["a", "c"]

