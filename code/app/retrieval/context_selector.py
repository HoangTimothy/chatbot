from app.schemas.retrieval import RetrievedChunk


class TokenBudgetContextSelector:
    def __init__(
        self,
        max_tokens: int = 3200,
        max_chunks: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
        self.min_score = min_score

    def select(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        used_tokens = 0

        for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
            if len(selected) >= self.max_chunks or chunk.score < self.min_score:
                break

            token_count = chunk.metadata.token_count or self._estimate_tokens(chunk.text)
            if selected and used_tokens + token_count > self.max_tokens:
                continue

            selected.append(chunk)
            used_tokens += token_count

        return selected

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

