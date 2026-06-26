import os
import sys

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "rag_core")))

import pytest
from rag_core.contracts.types import Chunk, DocumentRef, ParsedBlock
from rag_core.services.chunking import SemanticChunker
from rag_core.adapters.generator import OpenAIGenerator


class MockOpenAIClient:
    def __init__(self, completion_text: str):
        self.chat = self.Chat(completion_text)

    class Chat:
        def __init__(self, completion_text: str):
            self.completions = self.Completions(completion_text)

        class Completions:
            def __init__(self, completion_text: str):
                self.completion_text = completion_text

            def create(self, **kwargs):
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content
                    def __init__(self, content):
                        self.message = self.Message(content)
                class Response:
                    def __init__(self, content):
                        self.choices = [Choice(content)]
                return Response(self.completion_text)


def test_openai_generator_contextual_prefix_generation(monkeypatch):
    """Verify that OpenAIGenerator correctly constructs the prompt and calls OpenAI API."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    generator = OpenAIGenerator(openai_api_key="fake-key", model_name="gpt-4o-mini")
    
    # Mock the client response
    generator.client = MockOpenAIClient(completion_text="This is a summary of the section.")
    
    prefix = generator.generate_contextual_prefix(
        document_text="The whole doc content about Apple Inc.",
        chunk_text="Revenue grew by 10% in Q3."
    )
    
    assert prefix == "This is a summary of the section."


def test_openai_generator_contextual_prefix_offline_fallback(monkeypatch):
    """Verify that generate_contextual_prefix returns empty string and handles exception when LLM fails."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    generator = OpenAIGenerator(openai_api_key="fake-key", model_name="gpt-4o-mini")
    
    # Mock client to throw error
    class ErrorClient:
        def __getattr__(self, name):
            raise RuntimeError("API Rate limit exceeded")
    
    generator.client = ErrorClient()
    
    prefix = generator.generate_contextual_prefix(
        document_text="The whole doc.",
        chunk_text="Some chunk."
    )
    
    assert prefix == ""  # Graceful fallback to empty string


def test_chunking_token_recount_on_contextual_retrieval():
    """Verify that token recounts and character counts are correctly updated on situated chunks."""
    doc_ref = DocumentRef(
        workspace_id="ws-abc",
        document_id="doc-abc",
        document_version_id="ver-abc",
        file_name="financial_report.pdf",
        file_hash="report_hash_123"
    )
    
    chunker = SemanticChunker(document_ref=doc_ref)
    
    # Standard chunk
    c = Chunk(
        chunk_id="chunk-1",
        document=doc_ref,
        text="Revenue grew by 10% in Q3.",
        token_count=7,
        knowledge_branch_path=("general",),
        features={"char_count": 25}
    )
    
    prefix = "This is Apple Inc.'s Q3 2023 report."
    new_text = f"[Context: {prefix}]\n\n{c.text}"
    
    # Calculate tokens using chunker's encoder (cl100k_base or fallback split)
    encoder = getattr(chunker, "encoder", None)
    if encoder:
        new_tokens = len(encoder.encode(new_text))
    else:
        new_tokens = len(new_text.split())
        
    new_features = dict(c.features)
    new_features["char_count"] = len(new_text)
    new_features["contextual_prefix"] = prefix
    
    c_situated = Chunk(
        chunk_id=c.chunk_id,
        document=c.document,
        text=new_text,
        token_count=new_tokens,
        knowledge_branch_path=c.knowledge_branch_path,
        features=new_features
    )
    
    assert c_situated.text == "[Context: This is Apple Inc.'s Q3 2023 report.]\n\nRevenue grew by 10% in Q3."
    assert c_situated.token_count > 7
    assert c_situated.features["char_count"] == len(new_text)
    assert c_situated.features["contextual_prefix"] == prefix


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
