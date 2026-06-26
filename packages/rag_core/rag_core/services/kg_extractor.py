"""Knowledge graph entity and relation extractor using LLM.

Extracts structured (entity, relation, entity) triples from text chunks
and feeds them into the KnowledgeGraphPort during ingestion.
"""

import json
import logging
import os
import uuid
from typing import Sequence

from rag_core.contracts.types import DocumentRef, KGEntity, KGRelation

logger = logging.getLogger("rag_core.kg_extractor")


class KGExtractor:
    """Extract entities and relations from text using LLM.

    Supports both OpenAI and Google Gemini providers.
    """

    def __init__(
        self,
        openai_api_key: str = "",
        google_api_key: str = "",
        model_name: str = "gpt-4o-mini",
        prompt_path: str = "",
    ):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key

        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if self.provider == "openai" and not openai_api_key and google_api_key:
            self.provider = "google"

        self.openai_client = None
        if self.provider == "openai" and openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)

        self.prompt_template = ""
        self._load_prompt_template(prompt_path)

    def _load_prompt_template(self, prompt_path: str = "") -> None:
        paths = []
        if prompt_path:
            paths.append(prompt_path)

        paths.extend([
            "prompts/kg_extract.md",
            "../prompts/kg_extract.md",
            "../../prompts/kg_extract.md",
        ])

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        while curr_dir and curr_dir != os.path.dirname(curr_dir):
            candidate = os.path.join(curr_dir, "prompts", "kg_extract.md")
            if candidate not in paths:
                paths.append(candidate)
            candidate2 = os.path.join(curr_dir, "rag_project", "prompts", "kg_extract.md")
            if candidate2 not in paths:
                paths.append(candidate2)
            curr_dir = os.path.dirname(curr_dir)

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.prompt_template = f.read()
                        logger.info(f"Loaded KG extraction prompt from: {path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to read KG prompt path {path}: {e}")

        self.prompt_template = (
            "Extract entities and relations from the text. "
            "Output JSON with keys: entities, relations."
        )

    def extract(
        self,
        text: str,
        document_ref: DocumentRef | None = None,
    ) -> tuple[list[KGEntity], list[KGRelation]]:
        """Extract entities and relations from a text chunk.

        Returns:
            A tuple of (entities, relations) lists.
        """
        if not self.openai_client and self.provider != "google":
            logger.info("KGExtractor: No LLM client available. Skipping extraction.")
            return [], []

        try:
            content = self._call_llm(text)
            if not content:
                return [], []

            parsed = json.loads(content)
            entities = self._parse_entities(parsed.get("entities", []), document_ref)
            relations = self._parse_relations(parsed.get("relations", []))

            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations.")
            return entities, relations

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse KG extraction JSON: {e}")
            return [], []
        except Exception as e:
            logger.error(f"KG extraction failed: {e}")
            return [], []

    def _call_llm(self, text: str) -> str | None:
        """Send text to LLM for entity/relation extraction."""
        system_prompt = self.prompt_template

        if self.provider == "google":
            import google.generativeai as genai

            api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                return None

            genai.configure(api_key=api_key)

            gemini_model = "gemini-2.5-flash"
            if "gemini" in self.model_name:
                gemini_model = self.model_name

            model = genai.GenerativeModel(
                model_name=gemini_model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                text,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 30.0},
            )
            return response.text
        else:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content

    @staticmethod
    def _parse_entities(
        raw_entities: list[dict],
        document_ref: DocumentRef | None,
    ) -> list[KGEntity]:
        """Parse raw LLM entity output into KGEntity dataclasses."""
        entities: list[KGEntity] = []
        for item in raw_entities[:20]:  # Cap at 20 entities per chunk
            name = item.get("name", "").strip()
            if not name:
                continue

            props = dict(item.get("properties", {}))
            if document_ref:
                props["source_document"] = document_ref.file_name

            entities.append(
                KGEntity(
                    entity_id=f"ent_{uuid.uuid4().hex[:10]}",
                    name=name,
                    entity_type=item.get("entity_type", "unknown"),
                    properties=props,
                )
            )
        return entities

    @staticmethod
    def _parse_relations(raw_relations: list[dict]) -> list[KGRelation]:
        """Parse raw LLM relation output into KGRelation dataclasses."""
        relations: list[KGRelation] = []
        for item in raw_relations[:30]:  # Cap at 30 relations per chunk
            source = item.get("source", "").strip()
            target = item.get("target", "").strip()
            relation_type = item.get("relation_type", "").strip()

            if not source or not target or not relation_type:
                continue

            relations.append(
                KGRelation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    properties=dict(item.get("properties", {})),
                )
            )
        return relations
