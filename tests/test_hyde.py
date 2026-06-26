import os
import sys
from typing import Sequence

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "rag_core")))

import pytest
from rag_core.contracts.types import Chunk, DocumentRef, RetrievalCandidate, RoutedQuestion
from rag_core.ports.interfaces import VectorSearchPort
from rag_core.adapters.searchers import HydeVectorSearcher
from rag_core.adapters.generator import OpenAIGenerator


class DummyVectorSearcher(VectorSearchPort):
    def __init__(self):
        self.last_routed_question = None

    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        self.last_routed_question = routed_question
        return []


class MockHydeGenerator:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def generate_hypothetical_document(self, question: str) -> str:
        if self.should_fail:
            raise RuntimeError("LLM offline")
        return f"Hypothetical answer for {question}"


def test_hyde_vector_searcher_successful_generation():
    """Verify that HydeVectorSearcher successfully generates hypothetical doc and passes it to the searcher."""
    base_searcher = DummyVectorSearcher()
    generator = MockHydeGenerator(should_fail=False)
    hyde_searcher = HydeVectorSearcher(vector_searcher=base_searcher, generator=generator)

    routed_q = RoutedQuestion(
        question="what is water?",
        workspace_id="ws-123",
        branch_path=("general",),
        confidence=1.0
    )

    hyde_searcher.search(routed_q, limit=5)

    assert hyde_searcher.last_hyde_doc == "Hypothetical answer for what is water?"
    assert base_searcher.last_routed_question is not None
    assert base_searcher.last_routed_question.question == "Hypothetical answer for what is water?"
    assert base_searcher.last_routed_question.workspace_id == "ws-123"
    assert base_searcher.last_routed_question.branch_path == ("general",)


def test_hyde_vector_searcher_fallback_generation():
    """Verify that HydeVectorSearcher robustly falls back to original query text when generation fails."""
    base_searcher = DummyVectorSearcher()
    generator = MockHydeGenerator(should_fail=True)
    hyde_searcher = HydeVectorSearcher(vector_searcher=base_searcher, generator=generator)

    routed_q = RoutedQuestion(
        question="what is water?",
        workspace_id="ws-123",
        branch_path=("general",),
        confidence=1.0
    )

    # In case of failure, generator.generate_hypothetical_document will return the original question
    hyde_searcher.search(routed_q, limit=5)

    assert hyde_searcher.last_hyde_doc == "what is water?"
    assert base_searcher.last_routed_question is not None
    assert base_searcher.last_routed_question.question == "what is water?"


def test_openai_generator_hyde_fallback_when_offline(monkeypatch):
    """Verify that OpenAIGenerator's generate_hypothetical_document runs offline fallback without API key."""
    # Temporarily clear provider env variables so it triggers the offline/mock check
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Instantiating generator with no keys
    generator = OpenAIGenerator(openai_api_key="", model_name="gpt-mock")
    
    # Provider is "openai", self.client is None, provider != "google", so it goes to offline fallback
    doc = generator.generate_hypothetical_document("câu hỏi test")
    
    assert "Đây là tài liệu giả định trả lời cho câu hỏi:" in doc
    assert "câu hỏi test" in doc


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
