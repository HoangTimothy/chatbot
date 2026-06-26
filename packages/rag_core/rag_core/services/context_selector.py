from typing import Sequence
from rag_core.contracts.types import RetrievalCandidate, SelectedContext
from rag_core.ports.interfaces import ContextSelectorPort


class TokenAwareContextSelector(ContextSelectorPort):
    """Context selector that gathers candidates up to a specific token budget."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def select(self, candidates: Sequence[RetrievalCandidate]) -> SelectedContext:
        """Select candidates such that their token counts don't exceed the token budget."""
        selected_chunks = []
        total_tokens = 0
        
        for candidate in candidates:
            chunk = candidate.chunk
            if total_tokens + chunk.token_count <= self.max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk.token_count
            else:
                # Stop when the next chunk would exceed the token budget
                break
                
        return SelectedContext(
            chunks=tuple(selected_chunks),
            total_tokens=total_tokens
        )
