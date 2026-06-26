"""Core data contracts for the RAG pipeline.

These are intentionally small and framework-independent.
Implementation-specific fields can be added during the coding phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Query Routing Strategy
# ---------------------------------------------------------------------------

class RetrievalStrategy(str, Enum):
    """Determines which retrieval sources to use for a given query."""

    KB_ONLY = "kb_only"                    # Internal KB hybrid search only (default)
    WEB_SEARCH = "web_search"              # Web search only
    KNOWLEDGE_GRAPH = "knowledge_graph"    # Knowledge graph query only
    KB_AND_WEB = "kb_and_web"              # KB + Web in parallel
    KB_AND_KG = "kb_and_kg"                # KB + Knowledge Graph in parallel
    ALL = "all"                            # KB + Web + KG


@dataclass(frozen=True)
class DocumentRef:
    workspace_id: str
    document_id: str
    document_version_id: str
    file_name: str
    file_hash: str


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    page_number: int | None = None
    section_title: str | None = None
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: DocumentRef
    text: str
    token_count: int
    knowledge_branch_path: tuple[str, ...]
    features: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutedQuestion:
    question: str
    workspace_id: str
    branch_path: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: Chunk
    score: float
    source: str


@dataclass(frozen=True)
class SelectedContext:
    chunks: tuple[Chunk, ...]
    total_tokens: int


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: tuple[str, ...]
    insufficient_context: bool


# ---------------------------------------------------------------------------
# Extended Routing Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryRoute:
    """Result of intelligent query routing — decides which sources to query."""

    question: str
    workspace_id: str
    strategy: RetrievalStrategy
    branch_path: tuple[str, ...]
    confidence: float
    reasoning: str = ""                                   # LLM explanation for strategy choice
    sub_queries: tuple[str, ...] = ()                     # Decomposed sub-queries if needed


# ---------------------------------------------------------------------------
# Web Search Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WebSearchResult:
    """A single result from a web search provider."""

    title: str
    url: str
    snippet: str
    content: str          # Full extracted page content
    score: float = 0.0


# ---------------------------------------------------------------------------
# Knowledge Graph Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KGEntity:
    """A node in the knowledge graph."""

    entity_id: str
    name: str
    entity_type: str                                      # "person", "product", "department", etc.
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KGRelation:
    """A directed edge in the knowledge graph."""

    source: str            # source entity_id
    target: str            # target entity_id
    relation_type: str     # "manages", "belongs_to", "contains", etc.
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KGQueryResult:
    """Result of a knowledge graph query."""

    entities: tuple[KGEntity, ...] = ()
    relations: tuple[KGRelation, ...] = ()
    natural_language_answer: str = ""     # LLM-generated answer from KG triplets


