"""RAG infrastructure ports.

Adapters in apps or infra packages should implement these protocols.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from rag_core.contracts.types import (
    Chunk,
    GroundedAnswer,
    KGEntity,
    KGQueryResult,
    KGRelation,
    ParsedBlock,
    QueryRoute,
    RetrievalCandidate,
    RoutedQuestion,
    SelectedContext,
    WebSearchResult,
)


class DocumentParserPort(Protocol):
    def parse(self, file_path: str) -> Sequence[ParsedBlock]:
        """Parse a file into structured text blocks."""
        ...


class ChunkerPort(Protocol):
    def chunk(self, blocks: Sequence[ParsedBlock], branch_path: tuple[str, ...]) -> Sequence[Chunk]:
        """Create semantic chunks from parsed blocks."""
        ...


class RouterPort(Protocol):
    def route(self, workspace_id: str, question: str) -> RoutedQuestion:
        """Route question to the most relevant knowledge branch."""
        ...


class KeywordSearchPort(Protocol):
    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        """Run keyword retrieval."""
        ...


class VectorSearchPort(Protocol):
    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        """Run vector retrieval."""
        ...


class RerankerPort(Protocol):
    def rerank(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Rerank merged retrieval candidates."""
        ...


class ContextSelectorPort(Protocol):
    def select(self, candidates: Sequence[RetrievalCandidate]) -> SelectedContext:
        """Select compact evidence for generation."""
        ...


class GeneratorPort(Protocol):
    def generate(
        self,
        question: str,
        context: SelectedContext,
        chat_history: Sequence[dict[str, str]] | None = None
    ) -> GroundedAnswer:
        """Generate a grounded answer."""
        ...


class HydeGeneratorPort(Protocol):
    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document/answer for the given question."""
        ...


class ContextualGeneratorPort(Protocol):
    def generate_contextual_prefix(self, document_text: str, chunk_text: str) -> str:
        """Generate a short 1-2 sentence context to situate the chunk within the overall document."""
        ...


# ---------------------------------------------------------------------------
# Multi-Source Retrieval Ports
# ---------------------------------------------------------------------------

class QueryRouterPort(Protocol):
    def route(
        self,
        workspace_id: str,
        question: str,
        available_branches: Sequence[tuple[str, ...]],
        web_search_enabled: bool = False,
        kg_available: bool = False,
    ) -> QueryRoute:
        """Classify question and decide retrieval strategy + knowledge branch."""
        ...


class WebSearchPort(Protocol):
    def search(self, query: str, limit: int = 5) -> Sequence[WebSearchResult]:
        """Run a web search and return structured results."""
        ...


class KnowledgeGraphPort(Protocol):
    def query(self, question: str, workspace_id: str) -> KGQueryResult:
        """Query the knowledge graph for entities and relations relevant to the question."""
        ...

    def add_entities(self, entities: Sequence[KGEntity]) -> None:
        """Insert entities into the knowledge graph."""
        ...

    def add_relations(self, relations: Sequence[KGRelation]) -> None:
        """Insert relations into the knowledge graph."""
        ...
