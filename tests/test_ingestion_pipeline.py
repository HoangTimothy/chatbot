import os
import sys

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "rag_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "worker")))

import pytest
from fastapi.testclient import TestClient

from app.adapters.db.session import SessionLocal
from app.main import app
from worker.main import process_job
from shared.models import User, Workspace, Document, IngestionJob, Chunk
from shared.enums import DocumentStatus, IngestionJobStatus

client = TestClient(app)


def test_end_to_end_ingestion():
    """Verify end-to-end document ingestion pipeline: API upload, DB queuing, worker execution, and semantic chunking."""
    db = SessionLocal()
    try:
        # Resolve the seeded admin credentials and workspace ID
        user = db.query(User).filter(User.email == "admin@enterprise-rag.com").first()
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()

        assert user is not None
        assert workspace is not None

        # Clean up any existing duplicate document to ensure the test is repeatable
        existing_doc = db.query(Document).filter(
            Document.workspace_id == workspace.id,
            Document.name == "test_policy.txt"
        ).first()
        if existing_doc:
            db.delete(existing_doc)
            db.commit()

        # 1. Authenticate to retrieve token
        login_data = {
            "username": "admin@enterprise-rag.com",
            "password": "admin-password-123",
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 200
        token = response.json()["access_token"]

        headers = {
            "Authorization": f"Bearer {token}",
            "X-Workspace-ID": workspace.id,
        }

        # 2. Upload a test plaintext document containing policy language
        test_file_content = (
            "Enterprise Policy Document\n\n"
            "This is the official security policy. All employees must choose strong passwords "
            "and are prohibited from sharing their workspace API keys. Sharing credentials "
            "shall result in direct disciplinary actions.\n\n"
            "Technical Specifications\n\n"
            "The system specs require Python version 3.11 or greater. Qdrant operates on port 6333 "
            "and Elasticsearch is accessible on port 9200."
        )
        
        file_payload = {"file": ("test_policy.txt", test_file_content.encode("utf-8"), "text/plain")}
        
        upload_response = client.post("/documents/upload", headers=headers, files=file_payload)
        assert upload_response.status_code == 201
        
        data = upload_response.json()
        doc_id = data["document_id"]
        job_id = data["job_id"]
        assert data["name"] == "test_policy.txt"
        assert data["status"] == "uploaded"

        # 3. Verify job is queued in database
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        assert job is not None
        assert job.status == IngestionJobStatus.QUEUED

        # 4. Programmatically run worker job processing cycle
        # We manually claim it as RUNNING first to match worker atomic transaction flow
        job.status = IngestionJobStatus.RUNNING
        job.document.status = DocumentStatus.PROCESSING
        db.commit()

        # Run process_job
        process_job(job_id)

        # Refresh database session to fetch updated values
        db.expire_all()
        
        # 5. Assert completed job status and document status
        updated_job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        assert updated_job.status == IngestionJobStatus.COMPLETED
        
        updated_doc = db.query(Document).filter(Document.id == doc_id).first()
        assert updated_doc.status == DocumentStatus.READY

        # 6. Verify chunks were successfully parsed and stored
        chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
        assert len(chunks) >= 1

        # Verify semantic chunk text and token sizes
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.char_count > 0
            # Check if feature extraction correctly found contains_policy_language
            if "policy" in chunk.text.lower():
                assert chunk.contains_policy_language is True
            if "spec" in chunk.text.lower():
                assert chunk.contains_product_spec is True

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
