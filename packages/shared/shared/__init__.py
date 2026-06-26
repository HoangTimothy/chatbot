"""Shared product contracts for enterprise RAG."""

from shared.enums import (
    DocumentStatus,
    DocumentVisibility,
    IngestionJobStatus,
    UserRole,
)
from shared.models import (
    Base,
    User,
    Workspace,
    UserWorkspaceRole,
    Document,
    DocumentVersion,
    Chunk,
    IngestionJob,
    Conversation,
    Message,
    RetrievalTrace,
    AnswerFeedback,
    AuditLog,
)

__all__ = [
    "DocumentStatus",
    "DocumentVisibility",
    "IngestionJobStatus",
    "UserRole",
    "Base",
    "User",
    "Workspace",
    "UserWorkspaceRole",
    "Document",
    "DocumentVersion",
    "Chunk",
    "IngestionJob",
    "Conversation",
    "Message",
    "RetrievalTrace",
    "AnswerFeedback",
    "AuditLog",
]
