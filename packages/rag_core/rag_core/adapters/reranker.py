import logging
import re
from typing import Sequence
from openai import OpenAI

from rag_core.contracts.types import RetrievalCandidate
from rag_core.ports.interfaces import RerankerPort

logger = logging.getLogger("rag_core.reranker")


class RerankerAdapter(RerankerPort):
    """Adapter for reranking retrieval candidates using a cross-encoder model or similarity fallback."""

    def __init__(self, provider: str = "local", model_name: str = "", api_key: str = ""):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.client = None

        if self.provider == "cohere" and api_key:
            # Lazy import cohere if available, otherwise fallback
            try:
                import cohere
                self.client = cohere.Client(api_key=self.api_key)
            except ImportError:
                logger.warning("Cohere package not installed. Reranking will use fallback similarity.")

    def rerank(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Rerank candidates and select top N items."""
        if not candidates:
            return []

        # If Cohere client is successfully initialized, use Cohere Rerank API
        if self.provider == "cohere" and self.client:
            try:
                return self._cohere_rerank(question, candidates, limit)
            except Exception as e:
                logger.warning(f"Cohere rerank failed: {e}. Falling back to Jaccard overlap similarity.")

        # Default fallback: Jaccard word-overlap similarity reranker
        return self._fallback_rerank(question, candidates, limit)

    def _cohere_rerank(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Call Cohere rerank API endpoint."""
        texts = [c.chunk.text for c in candidates]
        response = self.client.rerank(
            model=self.model_name or "rerank-english-v3.0",
            query=question,
            documents=texts,
            top_n=limit
        )

        ranked_candidates = []
        for result in response.results:
            orig_candidate = candidates[result.index]
            # Replace score with Cohere relevance score
            ranked_candidates.append(
                RetrievalCandidate(
                    chunk=orig_candidate.chunk,
                    score=result.relevance_score,
                    source=f"reranker_{self.provider}"
                )
            )
        return ranked_candidates

    def _fallback_rerank(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate],
        limit: int,
    ) -> Sequence[RetrievalCandidate]:
        """Compute string overlap Jaccard similarity score to rank candidates."""
        query_words = set(re.findall(r"\w+", question.lower()))
        if not query_words:
            return candidates[:limit]

        scored_candidates = []
        for candidate in candidates:
            text_words = set(re.findall(r"\w+", candidate.chunk.text.lower()))
            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)
            
            # Jaccard overlap score
            jaccard_score = len(intersection) / len(union) if union else 0.0

            scored_candidates.append(
                RetrievalCandidate(
                    chunk=candidate.chunk,
                    score=jaccard_score,
                    source="reranker_fallback_jaccard"
                )
            )

        # Sort descending by Jaccard score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:limit]
