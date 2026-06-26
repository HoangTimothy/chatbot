import uuid # use primary key for DB
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Enum as SQLEnum,
    JSON,
    Float
)
from sqlalchemy.orm import declarative_base, relationship

from shared.enums import DocumentStatus, IngestionJobStatus, UserRole, DocumentVisibility

Base = declarative_base()

def generate_uuid() -> str:
    """Generate a unique string UUID."""
    return str(uuid.uuid4())

def get_utc_now() -> datetime:
    """Get the current UTC timestamp."""
    return datetime.now(timezone.utc)


class User(Base):
    """User representation representing identities with email and passsowrd hash."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    fullname = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    workspace_roles = relationship("UserWorkspaceRole", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("AnswerFeedback", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")

class Workspace(Base):
    """Workspace boundary for company tenant and billing isolation."""

    __tablename__ = "workspaces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    user_roles = relationship("UserWorkspaceRole", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")
    ingestion_jobs = relationship("IngestionJob", back_populates="workspace", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="workspace", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="workspace", cascade="all, delete-orphan")

class UserWorkspaceRole(Base):
    __tablename__ = "user_workspace_roles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    user = relationship("User", back_populates="workspace_roles")
    workspace = relationship("Workspace", back_populates="user_roles")
class Document(Base):
    """A user uploaded document asset."""

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(255), nullable=False)
    visibility = Column(SQLEnum(DocumentVisibility), default=DocumentVisibility.PUBLIC, nullable=False)
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    current_version_id = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    workspace = relationship("Workspace", back_populates="documents")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    ingestion_jobs = relationship("IngestionJob", back_populates="document", cascade="all, delete-orphan")

class DocumentVersion(Base):
    """Historical version tracker for documents."""

    __tablename__ = "document_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256
    file_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    document = relationship("Document", back_populates="versions")
    chunks = relationship("Chunk", back_populates="document_version")

class Chunk(Base):
    """Granular semantic chunk extracted from parsed document text."""

    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    document_version_id = Column(String(36), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False)
    source_file_name = Column(String(255), nullable=False)
    source_file_hash = Column(String(64), nullable=False)
    page_number = Column(Integer, nullable=True)
    sheet_name = Column(String(255), nullable=True)
    section_title = Column(String(255), nullable=True)
    heading_path = Column(JSON, nullable=True)  # List/array of hierarchy headings
    knowledge_branch_path = Column(String(1024), nullable=True)  # Path for routing
    language = Column(String(10), nullable=True)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    table_count = Column(Integer, default=0, nullable=False)
    image_count = Column(Integer, default=0, nullable=False)
    contains_policy_language = Column(Boolean, default=False, nullable=False)
    contains_product_spec = Column(Boolean, default=False, nullable=False)
    contains_procedure_steps = Column(Boolean, default=False, nullable=False)
    chunk_quality_score = Column(Float, default=1.0, nullable=False)
    embedding_model = Column(String(255), nullable=True)
    chunking_strategy = Column(String(50), nullable=True)
    chunk_version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    workspace = relationship("Workspace")
    document = relationship("Document", back_populates="chunks")
    document_version = relationship("DocumentVersion", back_populates="chunks")

class IngestionJob(Base): # Save state of implicit processes (doing, debug, hash file)
    """Background document parsing and ingestion tracker."""

    __tablename__ = "ingestion_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    status = Column(SQLEnum(IngestionJobStatus), default=IngestionJobStatus.QUEUED, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    document = relationship("Document", back_populates="ingestion_jobs")
    workspace = relationship("Workspace", back_populates="ingestion_jobs")

class Conversation(Base):
    """Conversation session representing a chat history stream."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now, nullable=False)

    workspace = relationship("Workspace", back_populates="conversations")
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """A singular query or response in a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    traces = relationship("RetrievalTrace", back_populates="message", cascade="all, delete-orphan")
    feedbacks = relationship("AnswerFeedback", back_populates="message", cascade="all, delete-orphan")

class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    routed_branch = Column(String(1024), nullable=True)
    hybrid_results = Column(JSON, nullable=True)  # Candidates + retrieval metrics
    reranked_results = Column(JSON, nullable=True)  # Top matches + cross-encoder scores
    query_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="traces")

class AnswerFeedback(Base):
    """User performance rating feedback on LLM responses."""

    __tablename__ = "answer_feedbacks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(String(10), nullable=False)  # "upvote", "downvote"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    message = relationship("Message", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")

class AuditLog(Base): 
    """Audit log tracking structural database modifications and access control actions."""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(255), nullable=False)
    target_type = Column(String(50), nullable=True)
    target_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    workspace = relationship("Workspace", back_populates="audit_logs")
    user = relationship("User", back_populates="audit_logs")
    
    