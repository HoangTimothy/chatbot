from dataclasses import dataclass
from re import Match
import re

from app.ingestion.document_loader import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    section: str | None
    hierarchy_path: list[str]
    token_count: int


class SemanticChunker:
    heading_pattern = re.compile(r"^(#{1,6}\s+.+|[A-Z0-9][A-Z0-9\s\-&/]{6,})$", re.MULTILINE)

    def __init__(self, min_tokens: int = 300, max_tokens: int = 800) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def chunk(self, document: LoadedDocument, hierarchy_path: list[str]) -> list[DocumentChunk]:
        sections = self._split_sections(document.text)
        chunks: list[DocumentChunk] = []

        for section_title, section_text in sections:
            paragraphs = [item.strip() for item in section_text.split("\n\n") if item.strip()]
            buffer: list[str] = []
            token_count = 0

            for paragraph in paragraphs:
                paragraph_tokens = self._estimate_tokens(paragraph)
                if buffer and token_count + paragraph_tokens > self.max_tokens:
                    chunks.append(
                        self._build_chunk(document, hierarchy_path, section_title, buffer, len(chunks))
                    )
                    buffer = []
                    token_count = 0

                buffer.append(paragraph)
                token_count += paragraph_tokens

            if buffer:
                chunks.append(self._build_chunk(document, hierarchy_path, section_title, buffer, len(chunks)))

        return chunks

    def _split_sections(self, text: str) -> list[tuple[str | None, str]]:
        matches = list(self.heading_pattern.finditer(text))
        if not matches:
            return [(None, text)]

        sections: list[tuple[str | None, str]] = []
        for index, match in enumerate(matches):
            title = self._clean_heading(match)
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((title, body))
        return sections or [(None, text)]

    @staticmethod
    def _build_chunk(
        document: LoadedDocument,
        hierarchy_path: list[str],
        section_title: str | None,
        paragraphs: list[str],
        index: int,
    ) -> DocumentChunk:
        text = "\n\n".join(paragraphs)
        return DocumentChunk(
            chunk_id=f"{document.document_id}:{index}",
            document_id=document.document_id,
            text=text,
            section=section_title,
            hierarchy_path=hierarchy_path,
            token_count=SemanticChunker._estimate_tokens(text),
        )

    @staticmethod
    def _clean_heading(match: Match[str]) -> str:
        return match.group(0).lstrip("#").strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

