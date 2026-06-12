from typing import Protocol

from app.schemas.retrieval import RetrievedChunk


class KeywordSearchPort(Protocol):
    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Run lexical retrieval, usually BM25."""
        ...


class VectorSearchPort(Protocol):
    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Run semantic vector retrieval."""
        ...


class EmptyKeywordSearchPort:
    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        return []


class EmptyVectorSearchPort:
    async def search(
        self,
        query: str,
        branch_path: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        return []
