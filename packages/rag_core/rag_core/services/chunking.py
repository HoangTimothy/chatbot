import uuid
from typing import Sequence
import re
import tiktoken

from rag_core.contracts.types import Chunk, DocumentRef, ParsedBlock
from rag_core.ports.interfaces import ChunkerPort


class SemanticChunker(ChunkerPort):
    """Semantic chunker implementation that groups parsed text blocks by headings and section boundaries."""

    def __init__(self, document_ref: DocumentRef, target_chunk_size: int = 400, max_chunk_size: int = 600):
        self.document_ref = document_ref
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def _get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken (OpenAI standard) with fallback to whitespace split."""
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text.split())

    def _extract_quality_features(self, text: str) -> dict:
        """Extract content features and analyze chunk quality signals."""
        lower_text = text.lower()

        # Quality scoring
        char_count = len(text)
        word_count = len(text.split())
        
        # Heuristic quality flags
        contains_policy = bool(re.search(r"\b(policy|shall|must|required|prohibited|regulation|agreement)\b", lower_text))
        contains_spec = bool(re.search(r"\b(spec|specification|version|dimensions|parameters|model|features)\b", lower_text))
        contains_procedure = bool(re.search(r"\b(step \d+|\d+\.|first,|then,|next,|finally|how to|guide)\b", lower_text))

        # Basic markdown table count
        table_count = text.count(" | ") // 2

        # Basic quality score metrics: degrade if too short, or lacks semantic words
        quality_score = 1.0
        if word_count < 10:
            quality_score = 0.4
        elif word_count < 30:
            quality_score = 0.8

        return {
            "char_count": char_count,
            "table_count": table_count,
            "image_count": 0,
            "contains_policy_language": contains_policy,
            "contains_product_spec": contains_spec,
            "contains_procedure_steps": contains_procedure,
            "chunk_quality_score": quality_score,
            "chunking_strategy": "semantic_heading",
            "chunk_version": 1,
            "source_file_name": self.document_ref.file_name,
            "source_file_hash": self.document_ref.file_hash,
        }

    def chunk(self, blocks: Sequence[ParsedBlock], branch_path: tuple[str, ...]) -> Sequence[Chunk]:
        """Group blocks by heading path and merge them within token limit boundaries."""
        chunks = []
        current_group = []
        current_group_tokens = 0
        current_group_headings = ()
        current_group_section = None
        current_group_page = None

        for block in blocks:
            block_text = block.text.strip()
            if not block_text:
                continue

            block_tokens = self._get_token_count(block_text)

            # If the single block is larger than max_chunk_size, we split it by sentences
            if block_tokens > self.max_chunk_size:
                # Flush existing group if any
                if current_group:
                    chunks.append(self._create_chunk(current_group, current_group_headings, branch_path, current_group_page))
                    current_group = []
                    current_group_tokens = 0

                sentences = re.split(r"(?<=[.!?])\s+", block_text)
                sub_group = []
                sub_tokens = 0
                for sentence in sentences:
                    sent_tokens = self._get_token_count(sentence)
                    if sub_tokens + sent_tokens > self.target_chunk_size and sub_group:
                        chunks.append(self._create_chunk(sub_group, block.heading_path, branch_path, block.page_number))
                        sub_group = [sentence]
                        sub_tokens = sent_tokens
                    else:
                        sub_group.append(sentence)
                        sub_tokens += sent_tokens
                if sub_group:
                    chunks.append(self._create_chunk(sub_group, block.heading_path, branch_path, block.page_number))
                continue

            # Check if we should flush the group:
            # 1. Heading path changes drastically
            # 2. Section changes
            # 3. Adding the block exceeds the target/max chunk size
            heading_changed = block.heading_path != current_group_headings and current_group_headings
            section_changed = block.section_title != current_group_section and current_group_section
            size_exceeded = (current_group_tokens + block_tokens) > self.target_chunk_size

            if (heading_changed or section_changed or size_exceeded) and current_group:
                chunks.append(self._create_chunk(current_group, current_group_headings, branch_path, current_group_page))
                current_group = []
                current_group_tokens = 0

            # Start or continue the group
            if not current_group:
                current_group_headings = block.heading_path
                current_group_section = block.section_title
                current_group_page = block.page_number

            current_group.append(block_text)
            current_group_tokens += block_tokens

        # Flush final group
        if current_group:
            chunks.append(self._create_chunk(current_group, current_group_headings, branch_path, current_group_page))

        return chunks

    def _create_chunk(self, text_list: list[str], heading_path: tuple[str, ...], branch_path: tuple[str, ...], page_number: int | None) -> Chunk:
        """Helper to construct a Chunk data object."""
        combined_text = "\n".join(text_list)
        tokens = self._get_token_count(combined_text)
        features = self._extract_quality_features(combined_text)
        
        # Add heading features if available
        features["heading_path"] = list(heading_path)
        if page_number:
            features["page_number"] = page_number

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document=self.document_ref,
            text=combined_text,
            token_count=tokens,
            knowledge_branch_path=branch_path,
            features=features,
        )
