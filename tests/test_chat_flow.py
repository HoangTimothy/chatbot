import os
import sys

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "rag_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))

import pytest
from fastapi.testclient import TestClient

from app.adapters.db.session import SessionLocal
from app.main import app
from shared.models import User, Workspace, Chunk as DbChunk, Document, DocumentVersion
from rag_core.contracts.types import Chunk, DocumentRef, RetrievalCandidate
from rag_core.services.context_selector import TokenAwareContextSelector

client = TestClient(app)


def test_token_aware_context_selector():
    """Verify TokenAwareContextSelector correctly bounds retrieval chunks by token budget."""
    doc_ref = DocumentRef(
        workspace_id="ws-123",
        document_id="doc-123",
        document_version_id="ver-123",
        file_name="test.txt",
        file_hash="hash"
    )
    
    c1 = RetrievalCandidate(
        chunk=Chunk(chunk_id="chk-1", document=doc_ref, text="Chunk 1 text", token_count=1500, knowledge_branch_path=()),
        score=0.9,
        source="vector"
    )
    c2 = RetrievalCandidate(
        chunk=Chunk(chunk_id="chk-2", document=doc_ref, text="Chunk 2 text", token_count=2000, knowledge_branch_path=()),
        score=0.8,
        source="vector"
    )
    c3 = RetrievalCandidate(
        chunk=Chunk(chunk_id="chk-3", document=doc_ref, text="Chunk 3 text", token_count=500, knowledge_branch_path=()),
        score=0.7,
        source="keyword"
    )

    # Selector with budget 3000 should only take c1 (1500 tokens). 
    # Adding c2 would be 3500 tokens, which exceeds 3000, so it stops.
    selector = TokenAwareContextSelector(max_tokens=3000)
    context = selector.select([c1, c2, c3])
    
    assert len(context.chunks) == 1
    assert context.chunks[0].chunk_id == "chk-1"
    assert context.total_tokens == 1500

    # Selector with budget 4500 should take all since 1500 + 2000 + 500 = 4000 <= 4500.
    selector_large = TokenAwareContextSelector(max_tokens=4500)
    context_large = selector_large.select([c1, c2, c3])
    assert len(context_large.chunks) == 3
    assert context_large.total_tokens == 4000


def test_chat_pipeline_sessions_flow():
    """Verify end-to-end chat routes, history, guardrail refusal, and traces."""
    db = SessionLocal()
    try:
        # 1. Resolve seed default credentials and default workspace
        user = db.query(User).filter(User.email == "admin@enterprise-rag.com").first()
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()
        
        assert user is not None
        assert workspace is not None

        # Clean existing chunks to prevent test pollution
        db.query(DbChunk).filter(DbChunk.workspace_id == workspace.id).delete()
        
        doc = db.query(Document).filter(
            Document.workspace_id == workspace.id,
            Document.name == "mock_chat_doc.txt"
        ).first()
        if doc:
            db.delete(doc)
            db.commit()

        doc = Document(
            workspace_id=workspace.id,
            name="mock_chat_doc.txt",
            file_path="mock/path.txt",
            file_size=100,
            content_type="text/plain"
        )
        db.add(doc)
        db.flush()

        version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            file_hash="mockhash456",
            file_path="mock/path.txt"
        )
        db.add(version)
        db.flush()

        # Seed mock chunk containing policy rules in finance/policy
        chunk1 = DbChunk(
            id="chunk-chat-finance-01",
            workspace_id=workspace.id,
            document_id=doc.id,
            document_version_id=version.id,
            source_file_name=doc.name,
            source_file_hash=version.file_hash,
            knowledge_branch_path="finance/policy",
            text="All employees must submit expense reports by the 5th of each month.",
            token_count=13,
            char_count=80
        )
        db.add(chunk1)
        db.commit()

        # 2. Authenticate user to obtain JWT token
        login_data = {
            "username": "admin@enterprise-rag.com",
            "password": "admin-password-123",
        }
        login_res = client.post("/auth/login", data=login_data)
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Workspace-ID": workspace.id
        }

        # 3. Create chat session
        session_res = client.post("/chat/sessions", json={"title": "Test Chat"}, headers=headers)
        assert session_res.status_code == 201
        session_data = session_res.json()
        session_id = session_data["id"]
        assert session_data["title"] == "Test Chat"

        # 4. List chat sessions
        list_res = client.get("/chat/sessions", headers=headers)
        assert list_res.status_code == 200
        sessions_list = list_res.json()
        assert len(sessions_list) >= 1
        assert any(s["id"] == session_id for s in sessions_list)

        # 5. POST Message - SUCCESS grounded response (using matching keyword overlap)
        msg_payload = {"content": "Tell me about the policy rule regarding expense reports"}
        msg_res = client.post(f"/chat/sessions/{session_id}/messages", json=msg_payload, headers=headers)
        assert msg_res.status_code == 200
        ans_data = msg_res.json()
        assert ans_data["insufficient_context"] is False
        assert "chunk-chat-finance-01" in ans_data["citations"]
        # In mock fallback mode the filename is prefixed, in LLM mode it is not.
        assert "mock_chat_doc.txt" in ans_data["answer"] or "expense reports" in ans_data["answer"].lower() or "báo cáo chi phí" in ans_data["answer"].lower()
        message_id = ans_data["message_id"]

        # 6. GET Trace for the message
        trace_res = client.get(f"/chat/sessions/{session_id}/trace/{message_id}", headers=headers)
        assert trace_res.status_code == 200
        trace_data = trace_res.json()
        assert trace_data["message_id"] == message_id
        assert trace_data["routed_branch"] == "finance/policy"
        assert trace_data["query_tokens"] is not None

        # 7. POST Message - REFUSAL response (irrelevant query)
        refuse_payload = {"content": "recipe for chocolate banana bread dessert"}
        refuse_res = client.post(f"/chat/sessions/{session_id}/messages", json=refuse_payload, headers=headers)
        assert refuse_res.status_code == 200
        ref_data = refuse_res.json()
        
        assert ref_data["insufficient_context"] is True
        assert ref_data["citations"] == []
        assert ref_data["answer"] == "I cannot find sufficient information in the knowledge base."

    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
