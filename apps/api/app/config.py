from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""

    APP_ENV: str = "local"
    APP_NAME: str = "Enterprise RAG Platform"

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/rag"

    # Queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Object Storage
    OBJECT_STORAGE_PROVIDER: str = "local"
    OBJECT_STORAGE_BUCKET: str = "enterprise-rag-local"
    OBJECT_STORAGE_LOCAL_PATH: str = "./storage/files"

    # Search & Vector Store
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "enterprise_chunks"
    ENABLE_HYDE: bool = False
    ENABLE_CONTEXTUAL_RETRIEVAL: bool = False

    # Query Routing & Multi-Source Retrieval
    ENABLE_QUERY_ROUTING: bool = False         # Intelligent multi-source query routing
    ENABLE_WEB_SEARCH: bool = False            # Web search as retrieval source
    ENABLE_KNOWLEDGE_GRAPH: bool = False       # Knowledge graph as retrieval source
    TAVILY_API_KEY: str = ""                   # Tavily web search API key
    KG_PERSIST_PATH: str = "./storage/kg"      # Knowledge graph JSON storage path
    KG_AUTO_EXTRACT: bool = False              # Auto-extract KG entities during ingestion

    # AI Configurations
    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    RERANKER_PROVIDER: str = "bge"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4.1-mini"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    GOOGLE_API_KEY: str = ""

    # Security / Auth
    JWT_SECRET_KEY: str = "super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

import os
if settings.OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = settings.OPENAI_BASE_URL
if settings.GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
if settings.EMBEDDING_PROVIDER:
    os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER
if settings.LLM_PROVIDER:
    os.environ["LLM_PROVIDER"] = settings.LLM_PROVIDER
if settings.LLM_MODEL:
    os.environ["LLM_MODEL"] = settings.LLM_MODEL
