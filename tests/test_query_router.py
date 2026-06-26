from unittest.mock import MagicMock, patch
from rag_core.contracts.types import RetrievalStrategy
from rag_core.services.query_router import QueryRouter


def test_query_router_fallback():
    # If both web search and KG are disabled, it should fallback immediately to DomainRouter
    branches = [("products", "manuals"), ("general",)]
    router = QueryRouter(available_branches=branches)
    
    # Mock DomainRouter's route method to avoid LLM call
    router._domain_router = MagicMock()
    mock_routed = MagicMock()
    mock_routed.question = "Hỏi về sản phẩm"
    mock_routed.workspace_id = "ws-1"
    mock_routed.branch_path = ("products", "manuals")
    mock_routed.confidence = 0.95
    router._domain_router.route.return_value = mock_routed

    route = router.route(
        workspace_id="ws-1",
        question="Hỏi về sản phẩm",
        available_branches=branches,
        web_search_enabled=False,
        kg_available=False,
    )

    assert route.strategy == RetrievalStrategy.KB_ONLY
    assert route.branch_path == ("products", "manuals")
    assert route.confidence == 0.95
    assert "Fallback to domain-only routing" in route.reasoning


def test_query_router_llm_routing():
    branches = [("products", "manuals"), ("general",)]
    router = QueryRouter(available_branches=branches)
    router.provider = "openai" # Force provider to openai for the test
    router.openai_client = MagicMock() # Set mock client to trigger LLM routing

    # Pre-canned JSON response from LLM
    mock_json_response = """
    {
        "strategy": "kb_and_web",
        "branch_path": ["products", "manuals"],
        "confidence": 0.9,
        "reasoning": "Product details might need manuals and online reviews",
        "sub_queries": ["phuong hai equipment manual", "phuong hai news"]
    }
    """

    with patch.object(router, "_call_openai", return_value=mock_json_response) as mock_call:
        route = router.route(
            workspace_id="ws-1",
            question="Tell me about Phượng Hải products",
            available_branches=branches,
            web_search_enabled=True,
            kg_available=True,
        )
        
        mock_call.assert_called_once()
        assert route.strategy == RetrievalStrategy.KB_AND_WEB
        assert route.branch_path == ("products", "manuals")
        assert route.confidence == 0.9
        assert "online reviews" in route.reasoning
        assert len(route.sub_queries) == 2
        assert "phuong hai news" in route.sub_queries
