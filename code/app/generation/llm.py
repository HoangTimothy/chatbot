from typing import Protocol

from app.generation.prompt import REFUSAL_MESSAGE
from app.schemas.retrieval import RetrievedChunk


class GeneratorPort(Protocol):
    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate an answer grounded in selected chunks."""
        ...


class RefusalGenerator:
    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        del question, chunks
        return REFUSAL_MESSAGE
