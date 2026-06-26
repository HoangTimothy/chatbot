"""Multi-source retrieval flow pipeline.

Extends the existing RetrievalPipeline to support multiple data sources
(KB, Web Search, Knowledge Graph) based on QueryRouter strategy decisions.
"""

import logging
from typing import Sequence

from rag_core.contracts.types import (
    KGQueryResult,
    QueryRoute,
    RetrievalCandidate,
    RetrievalStrategy,
    RoutedQuestion,
)
from rag_core.flows.retrieval_flow import RetrievalPipeline
from rag_core.ports.interfaces import (
    KnowledgeGraphPort,
    RerankerPort,
    WebSearchPort,
)
from rag_core.services.fusion import multi_source_fusion
from rag_core.services.query_router import QueryRouter

logger = logging.getLogger("rag_core.multi_source_flow")


class MultiSourceRetrievalPipeline:
    """Extended retrieval pipeline that orchestrates multiple sources
    based on QueryRouter strategy decisions.

    When web search and knowledge graph are disabled, this behaves
    identically to the existing RetrievalPipeline.
    """

    def __init__(
        self,
        query_router: QueryRouter,
        kb_pipeline: RetrievalPipeline,
        reranker: RerankerPort,
        web_searcher: WebSearchPort | None = None,
        knowledge_graph: KnowledgeGraphPort | None = None,
        web_search_enabled: bool = False,
        kg_available: bool = False,
    ):
        self.query_router = query_router
        self.kb_pipeline = kb_pipeline
        self.reranker = reranker
        self.web_searcher = web_searcher
        self.knowledge_graph = knowledge_graph
        self.web_search_enabled = web_search_enabled
        self.kg_available = kg_available

    def retrieve(
        self,
        workspace_id: str,
        question: str,
        available_branches: Sequence[tuple[str, ...]] | None = None,
        search_limit: int = 20,
        rerank_limit: int = 5,
    ) -> tuple[QueryRoute, Sequence[RetrievalCandidate], KGQueryResult | None]:
        """Run multi-source retrieval based on query routing.

        Returns:
            A tuple of (QueryRoute, ranked candidates, optional KG result).
        """
        branches = list(available_branches) if available_branches else []

        # 1. Route the query to determine strategy
        query_route = self.query_router.route(
            workspace_id=workspace_id,
            question=question,
            available_branches=branches,
            web_search_enabled=self.web_search_enabled,
            kg_available=self.kg_available,
        )

        logger.info(
            f"QueryRouter decision: strategy={query_route.strategy.value}, "
            f"branch={query_route.branch_path}, "
            f"confidence={query_route.confidence:.2f}, "
            f"reasoning='{query_route.reasoning[:100]}'"
        )

        # 2. Dispatch retrieval based on strategy
        strategy = query_route.strategy
        candidate_lists: dict[str, Sequence[RetrievalCandidate]] = {}
        kg_result: KGQueryResult | None = None

        # --- KB retrieval ---
        if strategy in (
            RetrievalStrategy.KB_ONLY,
            RetrievalStrategy.KB_AND_WEB,
            RetrievalStrategy.KB_AND_KG,
            RetrievalStrategy.ALL,
        ):
            kb_candidates = self._retrieve_from_kb(
                query_route, search_limit
            )
            if kb_candidates:
                # Split into keyword and vector sources for weighted fusion
                keyword_cands = [c for c in kb_candidates if c.source in ("elasticsearch", "sql_database")]
                vector_cands = [c for c in kb_candidates if c.source in ("qdrant", "hybrid_rrf")]
                other_cands = [c for c in kb_candidates if c.source not in ("elasticsearch", "sql_database", "qdrant", "hybrid_rrf")]

                if keyword_cands:
                    candidate_lists["keyword"] = keyword_cands
                if vector_cands:
                    candidate_lists["vector"] = vector_cands
                if other_cands:
                    candidate_lists["kb_other"] = other_cands

                # If no source separation, just add as unified KB
                if not keyword_cands and not vector_cands:
                    candidate_lists["keyword"] = kb_candidates

        # --- Web search ---
        if strategy in (
            RetrievalStrategy.WEB_SEARCH,
            RetrievalStrategy.KB_AND_WEB,
            RetrievalStrategy.ALL,
        ):
            web_candidates = self._retrieve_from_web(question, workspace_id, search_limit)
            if web_candidates:
                candidate_lists["web_search"] = web_candidates

        # --- Knowledge Graph ---
        if strategy in (
            RetrievalStrategy.KNOWLEDGE_GRAPH,
            RetrievalStrategy.KB_AND_KG,
            RetrievalStrategy.ALL,
        ):
            kg_result, kg_candidates = self._retrieve_from_kg(question, workspace_id)
            if kg_candidates:
                candidate_lists["knowledge_graph"] = kg_candidates

        # 3. Multi-source fusion
        if not candidate_lists:
            logger.warning("No candidates retrieved from any source.")
            return query_route, [], kg_result

        merged_candidates = multi_source_fusion(candidate_lists)

        # 4. Rerank unified candidates
        reranked = self.reranker.rerank(
            question=question,
            candidates=merged_candidates,
            limit=rerank_limit,
        )

        logger.info(
            f"Multi-source retrieval complete: "
            f"{len(merged_candidates)} merged → {len(reranked)} reranked. "
            f"Sources: {list(candidate_lists.keys())}"
        )

        return query_route, reranked, kg_result

    # ------------------------------------------------------------------
    # Source-specific retrieval methods
    # ------------------------------------------------------------------

    def _retrieve_from_kb(
        self,
        query_route: QueryRoute,
        search_limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Retrieve from internal KB using existing hybrid pipeline."""
        try:
            # Convert QueryRoute to RoutedQuestion for backward compatibility
            routed_question = QueryRouter.to_routed_question(query_route)

            # Use existing KB pipeline (keyword + vector + RRF)
            _, candidates = self.kb_pipeline.retrieve(
                workspace_id=query_route.workspace_id,
                question=query_route.question,
                search_limit=search_limit,
                rerank_limit=search_limit,  # Don't rerank here, do it after fusion
            )
            return candidates

        except Exception as e:
            logger.error(f"KB retrieval failed: {e}")
            return []

    def _retrieve_from_web(
        self,
        question: str,
        workspace_id: str,
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Retrieve from web search."""
        if not self.web_searcher:
            logger.warning("Web searcher not configured.")
            return []

        try:
            from rag_core.adapters.web_searcher import TavilyWebSearcher

            web_results = self.web_searcher.search(query=question, limit=min(limit, 5))
            candidates = TavilyWebSearcher.to_retrieval_candidates(
                web_results, workspace_id=workspace_id
            )
            logger.info(f"Web search returned {len(candidates)} candidates.")
            return candidates

        except Exception as e:
            logger.error(f"Web search retrieval failed: {e}")
            return []

    def _retrieve_from_kg(
        self,
        question: str,
        workspace_id: str,
    ) -> tuple[KGQueryResult | None, Sequence[RetrievalCandidate]]:
        """Query the knowledge graph."""
        if not self.knowledge_graph:
            logger.warning("Knowledge graph not configured.")
            return None, []

        try:
            from rag_core.adapters.knowledge_graph import NetworkXKnowledgeGraph

            kg_result = self.knowledge_graph.query(
                question=question,
                workspace_id=workspace_id,
            )

            candidates = NetworkXKnowledgeGraph.to_retrieval_candidates(
                kg_result, workspace_id=workspace_id
            )
            logger.info(
                f"KG query returned {len(kg_result.entities)} entities, "
                f"{len(kg_result.relations)} relations, "
                f"{len(candidates)} candidates."
            )
            return kg_result, candidates

        except Exception as e:
            logger.error(f"Knowledge graph query failed: {e}")
            return None, []
