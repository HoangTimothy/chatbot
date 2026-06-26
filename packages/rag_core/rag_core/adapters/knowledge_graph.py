"""In-memory knowledge graph adapter using NetworkX.

Provides entity/relation storage, persistence (JSON), and query
capabilities for the RAG pipeline. Designed as a Phase 1 KG solution
that can be replaced by Neo4j when scaling is needed.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Sequence

import networkx as nx  # type: ignore[import-untyped]

from rag_core.contracts.types import (
    Chunk,
    DocumentRef,
    KGEntity,
    KGQueryResult,
    KGRelation,
    RetrievalCandidate,
)
from rag_core.ports.interfaces import KnowledgeGraphPort

logger = logging.getLogger("rag_core.knowledge_graph")


class NetworkXKnowledgeGraph(KnowledgeGraphPort):
    """In-memory knowledge graph backed by NetworkX DiGraph.

    Features:
    - Add entities and relations
    - Query by entity name matching
    - 1-2 hop neighborhood traversal
    - JSON persistence and loading
    - Convert KG results to RetrievalCandidates for fusion
    """

    def __init__(self, persist_path: str = ""):
        self.graph = nx.DiGraph()
        self.persist_path = persist_path
        self._entity_index: dict[str, str] = {}  # lowercase name → node_id

        # Auto-load if persist file exists
        if persist_path:
            full_path = os.path.join(persist_path, "knowledge_graph.json")
            if os.path.exists(full_path):
                self.load(full_path)
                logger.info(
                    f"Loaded KG from {full_path}: "
                    f"{self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges"
                )

    # ------------------------------------------------------------------
    # Entity & Relation Management
    # ------------------------------------------------------------------

    def add_entities(self, entities: Sequence[KGEntity]) -> None:
        """Insert entities as nodes into the graph."""
        for entity in entities:
            node_id = self._find_or_create_node_id(entity.name, entity.entity_type)
            self.graph.add_node(
                node_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties=dict(entity.properties),
            )
            self._entity_index[entity.name.lower()] = node_id

    def add_relations(self, relations: Sequence[KGRelation]) -> None:
        """Insert relations as directed edges between entity nodes."""
        for rel in relations:
            source_id = self._find_or_create_node_id(rel.source)
            target_id = self._find_or_create_node_id(rel.target)

            self.graph.add_edge(
                source_id,
                target_id,
                relation_type=rel.relation_type,
                properties=dict(rel.properties),
            )

    def _find_or_create_node_id(
        self,
        name: str,
        entity_type: str = "unknown",
    ) -> str:
        """Find existing node by name or create a new one."""
        key = name.lower()
        if key in self._entity_index:
            return self._entity_index[key]

        node_id = f"ent_{uuid.uuid4().hex[:10]}"
        self.graph.add_node(
            node_id,
            name=name,
            entity_type=entity_type,
            properties={},
        )
        self._entity_index[key] = node_id
        return node_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, workspace_id: str) -> KGQueryResult:
        """Query the knowledge graph for entities and relations relevant to the question.

        Strategy:
        1. Extract candidate entity names from the question
        2. Match against known nodes (fuzzy substring match)
        3. Traverse 1-2 hop neighborhood
        4. Format as natural language summary
        """
        if self.graph.number_of_nodes() == 0:
            return KGQueryResult()

        # 1. Find matching entities
        matched_node_ids = self._match_entities(question)

        if not matched_node_ids:
            return KGQueryResult()

        # 2. Collect neighborhood (1-2 hops)
        entities: list[KGEntity] = []
        relations: list[KGRelation] = []
        visited_nodes: set[str] = set()
        visited_edges: set[tuple[str, str]] = set()

        for node_id in matched_node_ids:
            self._traverse(node_id, depth=2, entities=entities, relations=relations,
                           visited_nodes=visited_nodes, visited_edges=visited_edges)

        # 3. Generate natural language summary from triplets
        nl_answer = self._format_triplets(entities, relations)

        return KGQueryResult(
            entities=tuple(entities),
            relations=tuple(relations),
            natural_language_answer=nl_answer,
        )

    def _match_entities(self, question: str) -> list[str]:
        """Find entity nodes whose names appear in the question."""
        question_lower = question.lower()
        question_tokens = set(re.findall(r"\w+", question_lower))
        matched: list[str] = []

        for name_lower, node_id in self._entity_index.items():
            # Direct substring match
            if name_lower in question_lower:
                matched.append(node_id)
                continue

            # Token overlap (for multi-word entity names)
            name_tokens = set(re.findall(r"\w+", name_lower))
            if len(name_tokens) >= 2 and name_tokens.issubset(question_tokens):
                matched.append(node_id)
                continue

            # Single-word match if word is long enough (>=4 chars)
            for token in name_tokens:
                if len(token) >= 4 and token in question_tokens:
                    matched.append(node_id)
                    break

        return matched[:10]  # Cap to prevent explosion

    def _traverse(
        self,
        node_id: str,
        depth: int,
        entities: list[KGEntity],
        relations: list[KGRelation],
        visited_nodes: set[str],
        visited_edges: set[tuple[str, str]],
    ) -> None:
        """BFS traversal of node neighborhood."""
        if depth <= 0 or node_id in visited_nodes:
            return

        visited_nodes.add(node_id)
        node_data = self.graph.nodes.get(node_id, {})

        entities.append(
            KGEntity(
                entity_id=node_id,
                name=node_data.get("name", node_id),
                entity_type=node_data.get("entity_type", "unknown"),
                properties=dict(node_data.get("properties", {})),
            )
        )

        # Outgoing edges
        for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
            edge_key = (node_id, target_id)
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                relations.append(
                    KGRelation(
                        source=node_data.get("name", node_id),
                        target=self.graph.nodes.get(target_id, {}).get("name", target_id),
                        relation_type=edge_data.get("relation_type", "related_to"),
                        properties=dict(edge_data.get("properties", {})),
                    )
                )
                self._traverse(target_id, depth - 1, entities, relations,
                               visited_nodes, visited_edges)

        # Incoming edges
        for source_id, _, edge_data in self.graph.in_edges(node_id, data=True):
            edge_key = (source_id, node_id)
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                relations.append(
                    KGRelation(
                        source=self.graph.nodes.get(source_id, {}).get("name", source_id),
                        target=node_data.get("name", node_id),
                        relation_type=edge_data.get("relation_type", "related_to"),
                        properties=dict(edge_data.get("properties", {})),
                    )
                )
                self._traverse(source_id, depth - 1, entities, relations,
                               visited_nodes, visited_edges)

    @staticmethod
    def _format_triplets(
        entities: list[KGEntity],
        relations: list[KGRelation],
    ) -> str:
        """Format KG results as natural language for the generator."""
        if not relations:
            if entities:
                names = ", ".join(e.name for e in entities[:5])
                return f"Found entities: {names}"
            return ""

        lines: list[str] = []
        for rel in relations[:15]:  # Cap output
            lines.append(f"- {rel.source} → [{rel.relation_type}] → {rel.target}")

        return "Knowledge Graph relationships:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversion to RetrievalCandidate (for fusion)
    # ------------------------------------------------------------------

    @staticmethod
    def to_retrieval_candidates(
        kg_result: KGQueryResult,
        workspace_id: str = "",
    ) -> Sequence[RetrievalCandidate]:
        """Convert KG query result into RetrievalCandidates for unified fusion."""
        if not kg_result.natural_language_answer:
            return []

        doc_ref = DocumentRef(
            workspace_id=workspace_id,
            document_id="knowledge_graph",
            document_version_id="kg_live",
            file_name="Knowledge Graph",
            file_hash="",
        )

        chunk = Chunk(
            chunk_id=f"kg_{uuid.uuid4().hex[:12]}",
            document=doc_ref,
            text=kg_result.natural_language_answer,
            token_count=len(kg_result.natural_language_answer.split()),
            knowledge_branch_path=("knowledge_graph",),
            features={
                "source_type": "knowledge_graph",
                "entity_count": len(kg_result.entities),
                "relation_count": len(kg_result.relations),
            },
        )

        return [
            RetrievalCandidate(
                chunk=chunk,
                score=0.85,  # KG results have inherently high relevance
                source="knowledge_graph",
            )
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str = "") -> None:
        """Serialize graph to JSON file."""
        save_path = path or self.persist_path
        if not save_path:
            logger.warning("No persist path configured. Skipping KG save.")
            return

        # Ensure directory exists
        dir_path = save_path if os.path.isdir(save_path) else os.path.dirname(save_path)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        file_path = (
            os.path.join(save_path, "knowledge_graph.json")
            if os.path.isdir(save_path) or not save_path.endswith(".json")
            else save_path
        )

        data = nx.node_link_data(self.graph)
        # Also save entity index
        data["_entity_index"] = dict(self._entity_index)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Persisted KG to {file_path}: "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def load(self, path: str) -> None:
        """Load graph from JSON file."""
        if not os.path.exists(path):
            logger.warning(f"KG file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entity_index = data.pop("_entity_index", {})
        self.graph = nx.node_link_graph(data, directed=True)
        self._entity_index = dict(entity_index)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()
