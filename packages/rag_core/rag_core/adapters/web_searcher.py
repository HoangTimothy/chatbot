"""Web search adapter using Tavily API.

Tavily is optimized for RAG/AI use-cases and returns clean, structured
search results with extracted page content — no HTML parsing needed.

Falls back to a mock searcher when the API key is missing (dev/testing mode).
"""

import logging
import uuid
from typing import Sequence

from rag_core.contracts.types import (
    Chunk,
    DocumentRef,
    RetrievalCandidate,
    WebSearchResult,
)
from rag_core.ports.interfaces import WebSearchPort

logger = logging.getLogger("rag_core.web_searcher")


class TavilyWebSearcher(WebSearchPort):
    """Web search adapter backed by the Tavily Search API.

    Tavily docs: https://docs.tavily.com
    """

    def __init__(
        self,
        api_key: str = "",
        search_depth: str = "basic",
        max_results: int = 5,
    ):
        self.api_key = api_key
        self.search_depth = search_depth
        self.max_results = max_results
        self.client = None
        self._is_enabled = False

        if api_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]
                self.client = TavilyClient(api_key=api_key)
                self._is_enabled = True
                logger.info("Tavily web search adapter initialized successfully.")
            except ImportError:
                logger.warning(
                    "tavily-python package not installed. "
                    "Run: pip install tavily-python"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily client: {e}")
        else:
            logger.info("Tavily API key not provided. Web search is disabled.")

    def search(self, query: str, limit: int = 5) -> Sequence[WebSearchResult]:
        """Execute web search via Tavily and return structured results."""
        if not self._is_enabled or not self.client:
            logger.info("Tavily is disabled. Returning empty web search results.")
            return []

        try:
            effective_limit = min(limit, self.max_results)
            response = self.client.search(
                query=query,
                search_depth=self.search_depth,
                max_results=effective_limit,
                include_answer=False,
            )

            results: list[WebSearchResult] = []
            for item in response.get("results", []):
                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", "")[:300],
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                    )
                )

            logger.info(f"Tavily returned {len(results)} web search results for: {query[:80]}")
            return results

        except Exception as e:
            logger.error(f"Tavily web search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_retrieval_candidates(
        results: Sequence[WebSearchResult],
        workspace_id: str = "",
    ) -> Sequence[RetrievalCandidate]:
        """Convert web search results to RetrievalCandidate for unified fusion.

        Creates synthetic Chunk objects wrapping web content so they can be
        processed by the same reranker and context selector as KB chunks.
        """
        candidates: list[RetrievalCandidate] = []
        for result in results:
            # Create a synthetic document ref for web results
            doc_ref = DocumentRef(
                workspace_id=workspace_id,
                document_id=f"web:{result.url}",
                document_version_id="web_live",
                file_name=result.title or result.url,
                file_hash="",
            )

            chunk = Chunk(
                chunk_id=f"web_{uuid.uuid4().hex[:12]}",
                document=doc_ref,
                text=result.content,
                token_count=len(result.content.split()),
                knowledge_branch_path=("web",),
                features={
                    "source_type": "web_search",
                    "url": result.url,
                    "title": result.title,
                },
            )

            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    score=result.score,
                    source="web_search",
                )
            )

        return candidates
