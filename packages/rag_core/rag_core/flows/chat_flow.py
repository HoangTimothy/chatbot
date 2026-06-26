from __future__ import annotations

from typing import Sequence

from rag_core.contracts.types import (
    GroundedAnswer,
    KGQueryResult,
    QueryRoute,
    RetrievalCandidate,
    RetrievalStrategy,
    RoutedQuestion,
    SelectedContext,
)
from rag_core.flows.retrieval_flow import RetrievalPipeline
from rag_core.ports.interfaces import ContextSelectorPort, GeneratorPort


def chat_flow_steps() -> tuple[str, ...]:
    """Return the approved chat pipeline order."""
    return (
        "authenticate_user",
        "load_workspace_policy",
        "run_retrieval_flow",
        "check_context_sufficiency",
        "generate_grounded_answer",
        "attach_citations",
        "persist_conversation_message",
        "return_answer",
    )


class ChatPipeline:
    """The central RAG chat orchestrator managing retrieval, context filtering and grounded generation.

    Supports two modes:
    - Legacy: Uses RetrievalPipeline (kb_only, existing flow)
    - Multi-source: Uses MultiSourceRetrievalPipeline (kb + web + kg)
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_selector: ContextSelectorPort,
        generator: GeneratorPort,
        multi_source_pipeline: "MultiSourceRetrievalPipeline | None" = None,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.context_selector = context_selector
        self.generator = generator
        self.multi_source_pipeline = multi_source_pipeline

    def generate_response(
        self,
        workspace_id: str,
        question: str,
        chat_history: Sequence[dict[str, str]] | None = None,
        search_limit: int = 20,
        rerank_limit: int = 5,
    ) -> tuple[RoutedQuestion, Sequence[RetrievalCandidate], SelectedContext, GroundedAnswer]:
        """Runs the complete chat generation pipeline: retrieve -> context select -> generate.

        This is the backward-compatible method that returns RoutedQuestion.
        """
        # 1. Run retrieval flow to fetch candidates
        routed_question, candidates = self.retrieval_pipeline.retrieve(
            workspace_id=workspace_id,
            question=question,
            search_limit=search_limit,
            rerank_limit=rerank_limit,
        )

        # 2. Select compact context fitting within token bounds
        selected_context = self.context_selector.select(candidates)

        # 3. Generate grounded answer passing history context
        grounded_answer = self.generator.generate(
            question=question,
            context=selected_context,
            chat_history=chat_history,
        )

        return routed_question, candidates, selected_context, grounded_answer

    def generate_response_multi_source(
        self,
        workspace_id: str,
        question: str,
        available_branches: Sequence[tuple[str, ...]] | None = None,
        chat_history: Sequence[dict[str, str]] | None = None,
        search_limit: int = 20,
        rerank_limit: int = 5,
    ) -> tuple[QueryRoute, Sequence[RetrievalCandidate], SelectedContext, GroundedAnswer, KGQueryResult | None]:
        """Runs the multi-source chat generation pipeline.

        Uses MultiSourceRetrievalPipeline for intelligent query routing
        across KB, Web, and Knowledge Graph sources.

        Falls back to legacy generate_response() if multi_source_pipeline
        is not configured.

        Returns:
            (QueryRoute, candidates, selected_context, grounded_answer, kg_result)
        """
        if not self.multi_source_pipeline:
            # Fallback to legacy pipeline
            routed_question, candidates, selected_context, grounded_answer = self.generate_response(
                workspace_id=workspace_id,
                question=question,
                chat_history=chat_history,
                search_limit=search_limit,
                rerank_limit=rerank_limit,
            )
            # Wrap RoutedQuestion into QueryRoute for consistent return type
            query_route = QueryRoute(
                question=routed_question.question,
                workspace_id=routed_question.workspace_id,
                strategy=RetrievalStrategy.KB_ONLY,
                branch_path=routed_question.branch_path,
                confidence=routed_question.confidence,
                reasoning="Legacy pipeline (multi-source not configured)",
            )
            return query_route, candidates, selected_context, grounded_answer, None

        # 1. Run multi-source retrieval
        query_route, candidates, kg_result = self.multi_source_pipeline.retrieve(
            workspace_id=workspace_id,
            question=question,
            available_branches=available_branches,
            search_limit=search_limit,
            rerank_limit=rerank_limit,
        )

        # 2. Select compact context fitting within token bounds
        selected_context = self.context_selector.select(candidates)

        # 3. Generate grounded answer
        grounded_answer = self.generator.generate(
            question=question,
            context=selected_context,
            chat_history=chat_history,
        )

        return query_route, candidates, selected_context, grounded_answer, kg_result
