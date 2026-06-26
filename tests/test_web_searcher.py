from unittest.mock import MagicMock
from rag_core.adapters.web_searcher import TavilyWebSearcher
from rag_core.contracts.types import WebSearchResult


def test_tavily_web_searcher_disabled():
    # When api_key is empty, it should be disabled and return empty results
    searcher = TavilyWebSearcher(api_key="")
    results = searcher.search("test query")
    assert len(results) == 0


def test_tavily_web_searcher_success():
    searcher = TavilyWebSearcher(api_key="mock-key")
    # Manually enable and set client mock
    searcher._is_enabled = True
    mock_client = MagicMock()
    searcher.client = mock_client

    mock_client.search.return_value = {
        "results": [
            {
                "title": "Phượng Hải Equipment",
                "url": "http://phuonghai.com/products",
                "content": "Phượng Hải manufactures smart laboratory equipment.",
                "score": 0.92,
            }
        ]
    }

    results = searcher.search("Phượng Hải products", limit=3)
    mock_client.search.assert_called_once_with(
        query="Phượng Hải products",
        search_depth="basic",
        max_results=3,
        include_answer=False,
    )

    assert len(results) == 1
    assert results[0].title == "Phượng Hải Equipment"
    assert results[0].url == "http://phuonghai.com/products"
    assert results[0].snippet == "Phượng Hải manufactures smart laboratory equipment."
    assert results[0].score == 0.92


def test_to_retrieval_candidates():
    results = [
        WebSearchResult(
            title="Search Result 1",
            url="http://example.com/1",
            snippet="Snippet 1",
            content="Full content of page 1",
            score=0.88
        )
    ]
    
    candidates = TavilyWebSearcher.to_retrieval_candidates(results, workspace_id="ws-123")
    
    assert len(candidates) == 1
    c = candidates[0]
    assert c.source == "web_search"
    assert c.score == 0.88
    assert c.chunk.document.workspace_id == "ws-123"
    assert c.chunk.document.document_id == "web:http://example.com/1"
    assert c.chunk.text == "Full content of page 1"
    assert c.chunk.features["source_type"] == "web_search"
    assert c.chunk.features["url"] == "http://example.com/1"
    assert c.chunk.features["title"] == "Search Result 1"
