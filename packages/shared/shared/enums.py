"""Shared enum placeholders."""

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Document lifecycle states."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class IngestionJobStatus(StrEnum):
    """Ingestion job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UserRole(StrEnum):
    """Enterprise RAG user roles."""

    OWNER = "owner"
    ADMIN = "admin"
    KNOWLEDGE_MANAGER = "knowledge_manager"
    MEMBER = "member"
    AUDITOR = "auditor"


class DocumentVisibility(StrEnum):
    """Document visibility and permission rules."""

    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"



class TenantType(StrEnum):
    """Enterprise RAG tenant types"""
    TRIAL = "trial"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    FREE = "free"


class RetrievalStrategy(StrEnum):
    """Query routing retrieval strategies for multi-source retrieval."""

    KB_ONLY = "kb_only"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    KB_AND_WEB = "kb_and_web"
    KB_AND_KG = "kb_and_kg"
    ALL = "all"