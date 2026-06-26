import hashlib
import pathlib
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.session import get_db_session
from app.adapters.storage.client import ObjectStorageClient
from app.routes.auth import get_current_user
from app.routes.workspaces import get_current_workspace
from shared.enums import DocumentStatus, DocumentVisibility, IngestionJobStatus, UserRole
from shared.models import AuditLog, Document, DocumentVersion, IngestionJob, User, Workspace

router = APIRouter(prefix="/documents", tags=["documents"])
storage_client = ObjectStorageClient()


class DocumentUploadResponse(BaseModel):
    """Pydantic response model representing details of newly uploaded document."""

    document_id: str
    name: str
    version_id: str
    version_number: int
    job_id: str
    status: DocumentStatus


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    file: UploadFile = File(...),
):
    """Upload a new knowledge base document, persist original binary, and enqueue parser job."""
    # Enforce role logic: Only Knowledge Managers, Owners, and Admins can upload documents
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN, UserRole.KNOWLEDGE_MANAGER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to upload documents to this workspace.",
        )

    # Read binary content
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty files cannot be processed.",
        )

    # Compute SHA-256 hash
    file_hash = hashlib.sha256(content).hexdigest()

    # Deduplicate: Check if identical file content is already present in workspace
    dup_stmt = (
        select(DocumentVersion)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            Document.workspace_id == workspace.id,
            DocumentVersion.file_hash == file_hash,
        )
    )
    dup_result = await db.execute(dup_stmt)
    duplicate = dup_result.scalars().first()
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A document with this identical content already exists in this workspace.",
        )

    # Preserve file extension
    file_ext = pathlib.Path(file.filename).suffix
    if not file_ext:
        file_ext = ".txt"  # default to plaintext fallback

    # Generate unique stored filename
    stored_filename = f"{uuid.uuid4()}{file_ext}"

    # Save content to ObjectStorage
    relative_path = storage_client.upload_file(
        file_content=content,
        file_name=stored_filename,
        folder=f"workspaces/{workspace.id}",
    )

    # Insert Document metadata record
    document = Document(
        workspace_id=workspace.id,
        name=file.filename,
        file_path=relative_path,
        file_size=len(content),
        content_type=file.content_type or "application/octet-stream",
        visibility=DocumentVisibility.PUBLIC,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    await db.flush()  # populate document.id

    # Insert DocumentVersion record
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        file_hash=file_hash,
        file_path=relative_path,
    )
    db.add(version)
    await db.flush()  # populate version.id

    # Update document with current active version pointer
    document.current_version_id = version.id

    # Insert IngestionJob queue tracker in QUEUED state
    job = IngestionJob(
        document_id=document.id,
        workspace_id=workspace.id,
        status=IngestionJobStatus.QUEUED,
    )
    db.add(job)

    # Insert security audit log
    audit = AuditLog(
        workspace_id=workspace.id,
        user_id=current_user.id,
        action="document_upload",
        target_type="document",
        target_id=document.id,
    )
    db.add(audit)

    await db.commit()

    return DocumentUploadResponse(
        document_id=document.id,
        name=document.name,
        version_id=version.id,
        version_number=version.version_number,
        job_id=job.id,
        status=document.status,
    )


import re
import httpx

class DriveImportRequest(BaseModel):
    url: str

def parse_drive_url(url: str) -> tuple[str, str]:
    """Parse Google Drive URL and return (doc_type, doc_id).
    doc_type can be 'document' or 'spreadsheet'.
    """
    doc_match = re.search(r"docs\.google\.com/document/(?:u/\d+/)?d/([a-zA-Z0-9_-]+)", url)
    if doc_match:
        return "document", doc_match.group(1)
        
    sheet_match = re.search(r"docs\.google\.com/spreadsheets/(?:u/\d+/)?d/([a-zA-Z0-9_-]+)", url)
    if sheet_match:
        return "spreadsheet", sheet_match.group(1)
        
    raise ValueError("Invalid Google Drive URL. Must be a Google Doc or Google Sheet link.")

@router.post("/import-drive", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def import_google_drive(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    request: DriveImportRequest,
):
    """Import a public Google Doc or Google Sheet, saving it locally and queuing parser job."""
    # Enforce role logic: Only Knowledge Managers, Owners, and Admins can upload/import documents
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN, UserRole.KNOWLEDGE_MANAGER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to import documents to this workspace.",
        )

    # 1. Parse link
    try:
        doc_type, doc_id = parse_drive_url(request.url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    # 2. Construct export URL and file details
    if doc_type == "document":
        download_url = f"https://docs.google.com/document/d/{doc_id}/export?format=docx"
        file_name = f"GoogleDriveDoc_{doc_id}.docx"
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        file_ext = ".docx"
    else:
        download_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=xlsx"
        file_name = f"GoogleDriveSheet_{doc_id}.xlsx"
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_ext = ".xlsx"

    # 3. Fetch file content from Google Drive public export
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(download_url, timeout=45.0)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to export Google Drive file. Google returned status code {response.status_code}. Make sure the link is shared publicly ('Anyone with link can view')."
                )
            content = response.content
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to reach Google Drive endpoints: {e}"
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty files cannot be processed.",
        )

    # Compute SHA-256 hash
    file_hash = hashlib.sha256(content).hexdigest()

    # 4. Check if document already exists (Update case)
    dup_doc_stmt = (
        select(Document)
        .where(
            Document.workspace_id == workspace.id,
            Document.name == file_name,
        )
    )
    dup_doc_result = await db.execute(dup_doc_stmt)
    document = dup_doc_result.scalars().first()

    if document:
        # Update case: Verify if hash has changed
        ver_stmt = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .order_by(DocumentVersion.version_number.desc())
        )
        ver_result = await db.execute(ver_stmt)
        last_version = ver_result.scalars().first()
        
        if last_version and last_version.file_hash == file_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This Google Drive document has not changed since the last import.",
            )
            
        next_version_num = (last_version.version_number + 1) if last_version else 1
        stored_filename = f"{uuid.uuid4()}{file_ext}"

        # Save content to ObjectStorage
        relative_path = storage_client.upload_file(
            file_content=content,
            file_name=stored_filename,
            folder=f"workspaces/{workspace.id}",
        )

        # Create new version
        version = DocumentVersion(
            document_id=document.id,
            version_number=next_version_num,
            file_hash=file_hash,
            file_path=relative_path,
        )
        db.add(version)
        await db.flush()

        # Update document pointers
        document.file_path = relative_path
        document.file_size = len(content)
        document.status = DocumentStatus.UPLOADED
        document.current_version_id = version.id
        
        action = "document_update_drive"
    else:
        # New Document Import case: Deduplicate by hash check
        dup_hash_stmt = (
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.workspace_id == workspace.id,
                DocumentVersion.file_hash == file_hash,
            )
        )
        dup_hash_result = await db.execute(dup_hash_stmt)
        duplicate = dup_hash_result.scalars().first()
        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A document with this identical content already exists in this workspace.",
            )

        stored_filename = f"{uuid.uuid4()}{file_ext}"

        # Save content to ObjectStorage
        relative_path = storage_client.upload_file(
            file_content=content,
            file_name=stored_filename,
            folder=f"workspaces/{workspace.id}",
        )

        # Insert Document metadata
        document = Document(
            workspace_id=workspace.id,
            name=file_name,
            file_path=relative_path,
            file_size=len(content),
            content_type=content_type,
            visibility=DocumentVisibility.PUBLIC,
            status=DocumentStatus.UPLOADED,
        )
        db.add(document)
        await db.flush()

        # Insert Version
        version = DocumentVersion(
            document_id=document.id,
            version_number=1,
            file_hash=file_hash,
            file_path=relative_path,
        )
        db.add(version)
        await db.flush()

        # Update document active version pointer
        document.current_version_id = version.id
        action = "document_import_drive"

    # Insert IngestionJob queue tracker in QUEUED state
    job = IngestionJob(
        document_id=document.id,
        workspace_id=workspace.id,
        status=IngestionJobStatus.QUEUED,
    )
    db.add(job)

    # Security audit log
    audit = AuditLog(
        workspace_id=workspace.id,
        user_id=current_user.id,
        action=action,
        target_type="document",
        target_id=document.id,
    )
    db.add(audit)

    await db.commit()

    return DocumentUploadResponse(
        document_id=document.id,
        name=document.name,
        version_id=version.id,
        version_number=version.version_number,
        job_id=job.id,
        status=document.status,
    )


from datetime import datetime
from typing import List


class DocumentResponse(BaseModel):
    id: str
    name: str
    file_size: int
    content_type: str
    status: DocumentStatus
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve details of all uploaded documents in the current workspace."""
    stmt = (
        select(Document)
        .where(Document.workspace_id == workspace.id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Permanently delete a document and all related chunks in the workspace context."""
    # Only Owner, Admin, and Knowledge Manager can delete documents
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN, UserRole.KNOWLEDGE_MANAGER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete documents in this workspace."
        )

    stmt = select(Document).where(
        Document.id == document_id,
        Document.workspace_id == workspace.id
    )
    result = await db.execute(stmt)
    document = result.scalars().first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    # Clear from search clusters (Qdrant & Elasticsearch)
    try:
        from app.config import settings
        from rag_core.adapters.indexers import ElasticsearchIndexer, QdrantIndexer
        
        # clear ES
        es_indexer = ElasticsearchIndexer(es_url=settings.ELASTICSEARCH_URL, index_name=settings.QDRANT_COLLECTION)
        if es_indexer._is_enabled and es_indexer.client:
            es_indexer.client.delete_by_query(
                index=es_indexer.index_name,
                body={"query": {"term": {"document_id": document_id}}}
            )
        # clear Qdrant
        qdrant_indexer = QdrantIndexer(
            qdrant_url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION,
            embedding_model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        if qdrant_indexer._is_enabled and qdrant_indexer.client:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qdrant_indexer.client.delete(
                collection_name=qdrant_indexer.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
    except Exception as index_err:
        import logging
        logging.getLogger("api.documents").warning(f"Failed to clear search indexes for deleted document {document_id}: {index_err}")

    await db.delete(document)
    await db.commit()
    return {"detail": "Document deleted successfully."}

