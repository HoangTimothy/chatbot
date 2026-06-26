import os
import sys
import pytest
from unittest.mock import AsyncMock, patch

# Add project paths to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))

from fastapi.testclient import TestClient
from app.main import app
from app.adapters.db.session import SessionLocal
from app.routes.documents import parse_drive_url
from shared.models import User, Workspace, Document, DocumentVersion, IngestionJob
from shared.enums import UserRole, DocumentStatus

client = TestClient(app)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_parse_drive_url():
    """Verify correct URL parsing and ID extraction for Docs and Sheets."""
    doc_url = "https://docs.google.com/document/d/1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ/edit?tab=t.1y07157g645h#heading=h.nqsntncdipx"
    doc_type, doc_id = parse_drive_url(doc_url)
    assert doc_type == "document"
    assert doc_id == "1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ"

    sheet_url = "https://docs.google.com/spreadsheets/d/1t_2BuzbQc_9OQp-0N96X8u9o3uM8R0pQ-rN3O0qS8v8/edit#gid=0"
    sheet_type, sheet_id = parse_drive_url(sheet_url)
    assert sheet_type == "spreadsheet"
    assert sheet_id == "1t_2BuzbQc_9OQp-0N96X8u9o3uM8R0pQ-rN3O0qS8v8"

    invalid_url = "https://docs.google.com/presentation/d/1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ/edit"
    with pytest.raises(ValueError, match="Invalid Google Drive URL"):
        parse_drive_url(invalid_url)


@pytest.mark.anyio
@patch("httpx.AsyncClient.get")
@patch("app.adapters.storage.client.ObjectStorageClient.upload_file")
async def test_import_drive_endpoint_flow(mock_upload, mock_get):
    """Verify import-drive endpoint fetches, persists documents, updates and deduplicates."""
    db = SessionLocal()
    try:
        # 1. Setup mock responses
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"Mocked document content for Google Doc testing"
        mock_get.return_value = mock_response

        mock_upload.return_value = "workspaces/ws-123/GoogleDriveDoc_mockedid.docx"

        # Resolve seeded admin user and workspace
        user = db.query(User).filter(User.email == "admin@enterprise-rag.com").first()
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()
        assert user is not None
        assert workspace is not None

        # Clean existing Google Drive test docs in workspace to prevent test pollution
        db.query(Document).filter(
            Document.workspace_id == workspace.id,
            Document.name.like("GoogleDriveDoc_%")
        ).delete()
        db.commit()

        # Get JWT auth headers
        login_data = {
            "username": "admin@enterprise-rag.com",
            "password": "admin-password-123",
        }
        login_res = client.post("/auth/login", data=login_data)
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Workspace-ID": workspace.id,
        }

        # 2. Call import-drive endpoint (First Import - Creation)
        doc_url = "https://docs.google.com/document/d/1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ/edit"
        payload = {"url": doc_url}
        response = client.post("/documents/import-drive", json=payload, headers=headers)
        assert response.status_code == 201
        
        data = response.json()
        assert data["name"] == "GoogleDriveDoc_1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ.docx"
        assert data["version_number"] == 1
        assert data["status"] == "uploaded"
        doc_id = data["document_id"]
        job_id = data["job_id"]

        # Verify DB records
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert doc.name == "GoogleDriveDoc_1DOcPaJclGEsDiVJtszw2cN0ykgtMyVOLBkoJ-DvLtvQ.docx"
        assert doc.status == DocumentStatus.UPLOADED

        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        assert job is not None
        assert job.document_id == doc.id

        # 3. Call import-drive endpoint with same content (Deduplication - Expecting no change)
        response_dup = client.post("/documents/import-drive", json=payload, headers=headers)
        assert response_dup.status_code == 400
        assert "document has not changed" in response_dup.json()["detail"]

        # 4. Mock changed content for Google Doc
        mock_response_changed = AsyncMock()
        mock_response_changed.status_code = 200
        mock_response_changed.content = b"Modified document content for Google Doc testing - New Version"
        mock_get.return_value = mock_response_changed

        # Call import-drive endpoint (Second Import - Update Version)
        response_update = client.post("/documents/import-drive", json=payload, headers=headers)
        assert response_update.status_code == 201
        
        data_update = response_update.json()
        assert data_update["document_id"] == doc_id
        assert data_update["version_number"] == 2
        
        # Verify db version records
        versions = db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).all()
        assert len(versions) == 2

    finally:
        db.close()
