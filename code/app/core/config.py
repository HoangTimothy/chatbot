from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise Knowledge Assistant"
    environment: str = "local"
    log_level: str = "INFO"

    elasticsearch_url: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "enterprise_chunks"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_provider: str = "disabled"
    openai_api_key: SecretStr | None = None

    max_retrieval_candidates: int = Field(default=40, ge=5, le=200)
    max_context_chunks: int = Field(default=5, ge=1, le=10)
    max_context_tokens: int = Field(default=3200, ge=500, le=16000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()

