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
from shared.models import User, Workspace, Chunk as DbChunk, Document, DocumentVersion, Conversation, Message, AnswerFeedback, RetrievalTrace
from shared.enums import IngestionJobStatus, UserRole

client = TestClient(app)


def test_phase5_api_endpoints():
    """Verify document list/delete, feedback, session messages and admin tracking endpoints."""
    db = SessionLocal()
    try:
        # 1. Setup default credentials and workspace ID
        user = db.query(User).filter(User.email == "admin@enterprise-rag.com").first()
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()
        
        assert user is not None
        assert workspace is not None

        # Authenticate user
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

        # Seed mock documents and versions to test GET and DELETE documents
        doc = Document(
            workspace_id=workspace.id,
            name="mock_phase5_doc.txt",
            file_path="mock/path.txt",
            file_size=100,
            content_type="text/plain"
        )
        db.add(doc)
        db.flush()

        version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            file_hash="mockhash789",
            file_path="mock/path.txt"
        )
        db.add(version)
        db.commit()

        # Seed a mock message & trace to test message log fetching, trace auditing, and feedback rating
        session = Conversation(
            workspace_id=workspace.id,
            user_id=user.id,
            title="Phase5 Session"
        )
        db.add(session)
        db.flush()

        user_msg = Message(
            conversation_id=session.id,
            role="user",
            content="Hello"
        )
        assistant_msg = Message(
            conversation_id=session.id,
            role="assistant",
            content="Grounded Response"
        )
        db.add_all([user_msg, assistant_msg])
        db.flush()

        trace = RetrievalTrace(
            message_id=assistant_msg.id,
            routed_branch="technical/specs",
            hybrid_results=[],
            reranked_results=[{"chunk_id": "c1", "text": "Mock text", "score": 0.8, "source": "sql"}],
            query_tokens=2
        )
        db.add(trace)
        db.commit()

        # --- A. Test GET /documents ---
        docs_res = client.get("/documents", headers=headers)
        assert docs_res.status_code == 200
        docs_list = docs_res.json()
        assert len(docs_list) >= 1
        assert any(d["name"] == "mock_phase5_doc.txt" for d in docs_list)

        # --- B. Test GET /chat/sessions/{id}/messages ---
        msg_res = client.get(f"/chat/sessions/{session.id}/messages", headers=headers)
        assert msg_res.status_code == 200
        msg_list = msg_res.json()
        assert len(msg_list) == 2
        assert msg_list[0]["role"] == "user"
        assert msg_list[1]["role"] == "assistant"
        assert msg_list[1]["content"] == "Grounded Response"

        # --- C. Test POST /chat/sessions/{id}/messages/{msg_id}/feedback ---
        feedback_payload = {"rating": "upvote", "comment": "Great accurate response"}
        fb_res = client.post(
            f"/chat/sessions/{session.id}/messages/{assistant_msg.id}/feedback",
            json=feedback_payload,
            headers=headers
        )
        assert fb_res.status_code == 201
        
        # Verify db persistence
        db_fb = db.query(AnswerFeedback).filter(AnswerFeedback.message_id == assistant_msg.id).first()
        assert db_fb is not None
        assert db_fb.rating == "upvote"
        assert db_fb.comment == "Great accurate response"

        # --- D. Test GET /admin/jobs ---
        jobs_res = client.get("/admin/jobs", headers=headers)
        assert jobs_res.status_code == 200
        assert isinstance(jobs_res.json(), list)

        # --- E. Test GET /admin/audit-logs ---
        audit_res = client.get("/admin/audit-logs", headers=headers)
        assert audit_res.status_code == 200
        assert isinstance(audit_res.json(), list)

        # --- F. Test GET /admin/retrieval-traces ---
        traces_res = client.get("/admin/retrieval-traces", headers=headers)
        assert traces_res.status_code == 200
        traces_list = traces_res.json()
        assert len(traces_list) >= 1
        assert any(t["routed_branch"] == "technical/specs" for t in traces_list)

        # --- G. Test GET /admin/retrieval-traces/{trace_id} ---
        trace_detail_res = client.get(f"/admin/retrieval-traces/{trace.id}", headers=headers)
        assert trace_detail_res.status_code == 200
        trace_detail = trace_detail_res.json()
        assert trace_detail["id"] == trace.id
        assert len(trace_detail["reranked_results"]) == 1
        assert trace_detail["reranked_results"][0]["chunk_id"] == "c1"

        # --- H. Test DELETE /documents/{id} ---
        del_res = client.delete(f"/documents/{doc.id}", headers=headers)
        assert del_res.status_code == 200
        assert del_res.json()["detail"] == "Document deleted successfully."
        
        # Verify document deleted in db
        deleted_doc = db.query(Document).filter(Document.id == doc.id).first()
        assert deleted_doc is None

    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
