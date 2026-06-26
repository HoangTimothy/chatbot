"""Retrieval flow pipeline orchestration.

The enterprise retrieval philosophy is fixed:
hierarchical routing first, hybrid search second, vector search third.
"""

from typing import Sequence

from rag_core.contracts.types import RetrievalCandidate, RoutedQuestion
from rag_core.ports.interfaces import KeywordSearchPort, RerankerPort, RouterPort, VectorSearchPort
from rag_core.services.fusion import reciprocal_rank_fusion


def retrieval_flow_steps() -> tuple[str, ...]:
    """Return the approved retrieval pipeline order."""
    return (
        "normalize_question",
        "resolve_workspace_permissions",
        "route_to_knowledge_branch",
        "keyword_search",
        "vector_search",
        "merge_candidates",
        "rerank_candidates",
        "select_context",
        "build_retrieval_trace",
    )


class RetrievalPipeline:
    """The central RAG retrieval orchestrator executing routing, hybrid search, RRF, and reranking."""

    def __init__(
        self,
        router: RouterPort,
        keyword_searcher: KeywordSearchPort,
        vector_searcher: VectorSearchPort,
        reranker: RerankerPort,
    ):
        self.router = router
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher
        self.reranker = reranker

    def retrieve(
        self,
        workspace_id: str,
        question: str,
        search_limit: int = 20,
        rerank_limit: int = 5,
    ) -> tuple[RoutedQuestion, Sequence[RetrievalCandidate]]:
        """Run the end-to-end retrieval flow for a query in a workspace.

        Returns:
            A tuple of (RoutedQuestion metadata, Top ranked candidates list).
        """
        # 1. Normalize and Route to domain branch
        routed_question = self.router.route(workspace_id, question)

        # 2. Retrieve keyword candidates (Elasticsearch/BM25)
        keyword_candidates = self.keyword_searcher.search(routed_question, limit=search_limit)

        # 3. Retrieve vector candidates (Qdrant)
        vector_candidates = self.vector_searcher.search(routed_question, limit=search_limit)

        # 4. Merge results using Reciprocal Rank Fusion (RRF)
        merged_candidates = reciprocal_rank_fusion(
            keyword_candidates=keyword_candidates,
            vector_candidates=vector_candidates,
        )

        # 5. Rerank merged list to select the absolute top candidates
        reranked_candidates = self.reranker.rerank(
            question=question,
            candidates=merged_candidates,
            limit=rerank_limit,
        )

        return routed_question, reranked_candidates


