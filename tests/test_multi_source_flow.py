from unittest.mock import MagicMock
from rag_core.contracts.types import (
    Chunk,
    DocumentRef,
    KGQueryResult,
    QueryRoute,
    RetrievalCandidate,
    RetrievalStrategy,
    WebSearchResult,
)
from rag_core.flows.multi_source_flow import MultiSourceRetrievalPipeline


def test_multi_source_retrieval_flow():
    # 1. Create mock dependencies
    mock_router = MagicMock()
    mock_kb_pipeline = MagicMock()
    mock_reranker = MagicMock()
    mock_web_searcher = MagicMock()
    mock_kg = MagicMock()

    doc_ref = DocumentRef(
        workspace_id="ws-1",
        document_id="doc-1",
        document_version_id="ver-1",
        file_name="doc.txt",
        file_hash="hash"
    )

    # Mock routing decision: KB_AND_WEB
    mock_route = QueryRoute(
        question="test question",
        workspace_id="ws-1",
        strategy=RetrievalStrategy.KB_AND_WEB,
        branch_path=("general",),
        confidence=0.9
    )
    mock_router.route.return_value = mock_route

    # Mock KB candidates
    kb_candidate = RetrievalCandidate(
        chunk=Chunk(chunk_id="kb-chunk-1", document=doc_ref, text="KB text content", token_count=10, knowledge_branch_path=("general",)),
        score=0.8,
        source="qdrant"
    )
    mock_kb_pipeline.retrieve.return_value = (None, [kb_candidate])

    # Mock Web results
    web_result = WebSearchResult(
        title="Web Page Title",
        url="http://webpage.com",
        snippet="Web snippet content",
        content="Web text content",
        score=0.75
    )
    mock_web_searcher.search.return_value = [web_result]

    # Mock Reranker behavior (just returns candidates passed to it)
    def mock_rerank(question, candidates, limit):
        return candidates[:limit]
    mock_reranker.rerank.side_effect = mock_rerank

    # 2. Instantiate pipeline
    pipeline = MultiSourceRetrievalPipeline(
        query_router=mock_router,
        kb_pipeline=mock_kb_pipeline,
        reranker=mock_reranker,
        web_searcher=mock_web_searcher,
        knowledge_graph=mock_kg,
        web_search_enabled=True,
        kg_available=True
    )

    # 3. Run retrieval
    route, candidates, kg_res = pipeline.retrieve(
        workspace_id="ws-1",
        question="test question",
        available_branches=[("general",)],
        search_limit=5,
        rerank_limit=5
    )

    # 4. Assertions
    mock_router.route.assert_called_once()
    mock_kb_pipeline.retrieve.assert_called_once()
    mock_web_searcher.search.assert_called_once_with(query="test question", limit=5)
    
    assert route.strategy == RetrievalStrategy.KB_AND_WEB
    assert len(candidates) == 2
    assert any(c.source == "web_search" for c in candidates)
    assert any(c.chunk.chunk_id == "kb-chunk-1" for c in candidates)
