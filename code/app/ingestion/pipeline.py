from pathlib import Path
from typing import Protocol

from app.ingestion.chunker import DocumentChunk, SemanticChunker
from app.ingestion.document_loader import LoaderRegistry
from app.ingestion.hierarchy import HierarchyResolver


class ChunkIndexPort(Protocol):
    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Persist chunks into keyword and vector indexes."""
        ...


class IngestionPipeline:
    def __init__(
        self,
        loader_registry: LoaderRegistry,
        hierarchy_resolver: HierarchyResolver,
        chunker: SemanticChunker,
        index: ChunkIndexPort,
    ) -> None:
        self.loader_registry = loader_registry
        self.hierarchy_resolver = hierarchy_resolver
        self.chunker = chunker
        self.index = index

    async def ingest_path(self, path: Path) -> list[DocumentChunk]:
        document = self.loader_registry.load(path)
        hierarchy_path = self.hierarchy_resolver.resolve(document.title)
        chunks = self.chunker.chunk(document, hierarchy_path)
        await self.index.upsert(chunks)
        return chunks
