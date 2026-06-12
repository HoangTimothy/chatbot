from functools import lru_cache

from app.core.config import Settings, get_settings
from app.generation.llm import RefusalGenerator
from app.reranking.reranker import ScorePreservingReranker
from app.retrieval.context_selector import TokenBudgetContextSelector
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.interfaces import EmptyKeywordSearchPort, EmptyVectorSearchPort
from app.routing.domain_router import TreeDomainRouter, build_default_company_tree
from app.services.rag_service import RagService


@lru_cache
def get_rag_service() -> RagService:
    settings: Settings = get_settings()

    router = TreeDomainRouter(root=build_default_company_tree())
    retriever = HybridRetriever(
        keyword_search=EmptyKeywordSearchPort(),
        vector_search=EmptyVectorSearchPort(),
        max_candidates=settings.max_retrieval_candidates,
    )
    reranker = ScorePreservingReranker()
    selector = TokenBudgetContextSelector(
        max_chunks=settings.max_context_chunks,
        max_tokens=settings.max_context_tokens,
    )
    generator = RefusalGenerator()

    return RagService(
        router=router,
        retriever=retriever,
        reranker=reranker,
        context_selector=selector,
        generator=generator,
    )

