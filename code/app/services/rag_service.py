from app.generation.llm import GeneratorPort
from app.generation.prompt import REFUSAL_MESSAGE
from app.reranking.reranker import RerankerPort
from app.retrieval.context_selector import TokenBudgetContextSelector
from app.retrieval.hybrid_search import HybridRetriever
from app.routing.domain_router import DomainRouter
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.retrieval import RetrievedChunk


class RagService:
    def __init__(
        self,
        router: DomainRouter,
        retriever: HybridRetriever,
        reranker: RerankerPort,
        context_selector: TokenBudgetContextSelector,
        generator: GeneratorPort,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.reranker = reranker
        self.context_selector = context_selector
        self.generator = generator

    async def answer(self, request: ChatRequest) -> ChatResponse:
        routed = self.router.route(request.question)
        candidates = await self.retriever.retrieve(request.question, routed.branch_path)
        reranked = await self.reranker.rerank(request.question, candidates, top_k=request.top_k)
        selected = self.context_selector.select(reranked)

        if not selected:
            return ChatResponse(
                answer=REFUSAL_MESSAGE,
                routed_branch=routed.branch_path,
                route_confidence=routed.confidence,
                insufficient_context=True,
            )

        answer = await self.generator.generate(request.question, selected)
        insufficient = answer.strip() == REFUSAL_MESSAGE

        return ChatResponse(
            answer=answer,
            citations=[] if insufficient else self._citations(selected),
            routed_branch=routed.branch_path,
            route_confidence=routed.confidence,
            insufficient_context=insufficient,
        )

    @staticmethod
    def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        return [
            Citation(
                chunk_id=chunk.chunk_id,
                document_title=chunk.metadata.title,
                section=chunk.metadata.section,
                page_number=chunk.metadata.page_number,
                source_uri=chunk.metadata.source_uri,
                hierarchy_path=chunk.metadata.hierarchy_path,
            )
            for chunk in chunks
        ]

