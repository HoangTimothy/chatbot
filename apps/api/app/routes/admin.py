import logging
from datetime import datetime
from typing import Annotated, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.session import get_db_session
from app.routes.auth import get_current_user
from app.routes.workspaces import get_current_workspace
from shared.enums import UserRole, IngestionJobStatus
from shared.models import User, Workspace, IngestionJob, AuditLog, RetrievalTrace, Message, Conversation, Document

logger = logging.getLogger("api.admin")
router = APIRouter(prefix="/admin", tags=["admin"])


# --- Pydantic Schema Models ---

class JobResponse(BaseModel):
    id: str
    document_name: str | None = None
    status: IngestionJobStatus
    error_message: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    id: str
    user_email: str | None = None
    action: str
    target_type: str | None = None
    target_id: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class TraceResponse(BaseModel):
    id: str
    message_content: str | None = None
    routed_branch: str | None = None
    query_tokens: int | None = None
    total_tokens: int | None = None
    had_fallback: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


# --- FastAPI Endpoint Definitions ---

@router.get("/jobs", response_model=List[JobResponse])
async def list_ingestion_jobs(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """List details and status of all document ingestion jobs in the workspace."""
    # Enforce role: admin or owner only
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace Owners and Admins can view ingestion jobs."
        )

    # Join Document to select related document name
    stmt = (
        select(IngestionJob, Document)
        .join(Document, IngestionJob.document_id == Document.id)
        .where(IngestionJob.workspace_id == workspace.id)
        .order_by(IngestionJob.created_at.desc())
    )
    result = await db.execute(stmt)
    
    formatted = []
    for row in result.all():
        job_record = row[0]
        doc_record = row[1]
        formatted.append(
            JobResponse(
                id=job_record.id,
                document_name=doc_record.name if doc_record else "Unknown Document",
                status=job_record.status,
                error_message=job_record.error_message,
                created_at=job_record.created_at
            )
        )
    return formatted


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def list_audit_logs(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve security audit trails and operation logs in the workspace."""
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace Owners and Admins can view audit logs."
        )

    stmt = (
        select(AuditLog, User)
        .outerjoin(User, AuditLog.user_id == User.id)
        .where(AuditLog.workspace_id == workspace.id)
        .order_by(AuditLog.created_at.desc())
    )
    result = await db.execute(stmt)
    
    formatted = []
    for row in result.all():
        log_record = row[0]
        user_record = row[1]
        formatted.append(
            AuditLogResponse(
                id=log_record.id,
                user_email=user_record.email if user_record else "System",
                action=log_record.action,
                target_type=log_record.target_type,
                target_id=log_record.target_id,
                created_at=log_record.created_at
            )
        )
    return formatted


@router.get("/retrieval-traces", response_model=List[TraceResponse])
async def list_retrieval_traces(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve query routing and hybrid/vector retrieval metrics traces."""
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace Owners and Admins can view retrieval traces."
        )

    stmt = (
        select(RetrievalTrace, Message)
        .join(Message, RetrievalTrace.message_id == Message.id)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.workspace_id == workspace.id)
        .order_by(RetrievalTrace.created_at.desc())
    )
    result = await db.execute(stmt)
    
    formatted = []
    for row in result.all():
        trace_record = row[0]
        message_record = row[1]
        
        # Calculate/get total tokens
        total_tokens = None
        if trace_record.hybrid_results and isinstance(trace_record.hybrid_results, dict):
            token_metrics = trace_record.hybrid_results.get("token_metrics")
            if token_metrics:
                total_tokens = token_metrics.get("total_tokens")
        
        if total_tokens is None:
            # Fallback estimation
            q_tok = trace_record.query_tokens or 0
            r_tok = 0
            if message_record:
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    r_tok = len(enc.encode(message_record.content))
                except Exception:
                    r_tok = len(message_record.content.split())
            
            c_tok = 0
            if trace_record.reranked_results:
                ctx_text = "".join([c.get("text", "") for c in trace_record.reranked_results])
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    c_tok = len(enc.encode(ctx_text))
                except Exception:
                    c_tok = len(ctx_text.split())
            total_tokens = q_tok + r_tok + c_tok

        # Calculate/get had_fallback flag
        had_fallback = False
        if trace_record.hybrid_results and isinstance(trace_record.hybrid_results, dict):
            fallbacks = trace_record.hybrid_results.get("fallbacks")
            if fallbacks and isinstance(fallbacks, dict):
                had_fallback = any(fallbacks.values())

        formatted.append(
            TraceResponse(
                id=trace_record.id,
                message_content=message_record.content if message_record else "Unknown Message",
                routed_branch=trace_record.routed_branch,
                query_tokens=trace_record.query_tokens,
                total_tokens=total_tokens,
                had_fallback=had_fallback,
                created_at=trace_record.created_at
            )
        )
    return formatted


class TraceDetailResponse(BaseModel):
    id: str
    message_id: str
    routed_branch: str | None = None
    hybrid_results: Any
    reranked_results: Any
    query_tokens: int | None = None
    response_tokens: int | None = None
    context_tokens: int | None = None
    total_tokens: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/retrieval-traces/{trace_id}", response_model=TraceDetailResponse)
async def get_retrieval_trace_detail(
    trace_id: str,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve the detailed metadata of a specific RAG query trace, including ranked candidate chunks."""
    current_role = getattr(workspace, "current_user_role", None)
    if current_role not in [UserRole.OWNER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace Owners and Admins can view retrieval traces."
        )

    stmt = (
        select(RetrievalTrace)
        .join(Message, RetrievalTrace.message_id == Message.id)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            RetrievalTrace.id == trace_id,
            Conversation.workspace_id == workspace.id
        )
    )
    result = await db.execute(stmt)
    trace = result.scalars().first()
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace not found."
        )

    # Resolve token metrics
    response_tokens = None
    context_tokens = None
    total_tokens = None
    
    if trace.hybrid_results and isinstance(trace.hybrid_results, dict):
        token_metrics = trace.hybrid_results.get("token_metrics")
        if token_metrics:
            response_tokens = token_metrics.get("response_tokens")
            context_tokens = token_metrics.get("context_tokens")
            total_tokens = token_metrics.get("total_tokens")

    if response_tokens is None:
        # Fallback counting on message content
        msg_stmt = select(Message).where(Message.id == trace.message_id)
        msg_result = await db.execute(msg_stmt)
        msg = msg_result.scalars().first()
        if msg:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                response_tokens = len(enc.encode(msg.content))
            except Exception:
                response_tokens = len(msg.content.split())
        else:
            response_tokens = 0
            
    if context_tokens is None:
        ctx_text = "".join([c.get("text", "") for c in trace.reranked_results]) if trace.reranked_results else ""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            context_tokens = len(enc.encode(ctx_text))
        except Exception:
            context_tokens = len(ctx_text.split())

    if total_tokens is None:
        q_tok = trace.query_tokens or 0
        r_tok = response_tokens or 0
        c_tok = context_tokens or 0
        total_tokens = q_tok + r_tok + c_tok

    return TraceDetailResponse(
        id=trace.id,
        message_id=trace.message_id,
        routed_branch=trace.routed_branch,
        hybrid_results=trace.hybrid_results,
        reranked_results=trace.reranked_results,
        query_tokens=trace.query_tokens,
        response_tokens=response_tokens,
        context_tokens=context_tokens,
        total_tokens=total_tokens,
        created_at=trace.created_at
    )

