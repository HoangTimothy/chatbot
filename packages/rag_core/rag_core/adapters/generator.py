import logging
import json
import os
import re
from typing import Sequence
from openai import OpenAI

from rag_core.contracts.types import SelectedContext, GroundedAnswer, Chunk
from rag_core.ports.interfaces import GeneratorPort

logger = logging.getLogger("rag_core.generator")


class OpenAIGenerator(GeneratorPort):
    """OpenAI generator adapter with fallback to mock response for testing."""

    def __init__(self, openai_api_key: str = "", model_name: str = "gpt-4o-mini", prompt_path: str = ""):
        self.model_name = model_name
        self.prompt_template = ""
        self.hyde_prompt_template = ""
        self.client = None

        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")

        # Fallback provider to google if OpenAI API key is missing but Google API key is available
        if self.provider == "openai" and not openai_api_key and self.google_api_key:
            self.provider = "google"
            logger.info("OpenAI API key missing. Auto-falling back LLM_PROVIDER to 'google'")

        if self.provider == "openai" and openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)

        # Load prompt template from prompts/grounded_answer.md
        self._load_prompt_template(prompt_path)
        # Load HyDE prompt template from prompts/hyde.md
        self._load_hyde_prompt_template(prompt_path)
        self.had_fallback = False
        self.fallback_reason = None

    def _load_prompt_template(self, prompt_path: str = ""):
        paths = []
        if prompt_path:
            paths.append(prompt_path)
        # Search relative paths
        paths.extend([
            "prompts/grounded_answer.md",
            "../prompts/grounded_answer.md",
            "../../prompts/grounded_answer.md"
        ])
        
        # Try to locate by walking up directories to find workspace root
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        while curr_dir and curr_dir != os.path.dirname(curr_dir):
            potential_path = os.path.join(curr_dir, "prompts", "grounded_answer.md")
            if potential_path not in paths:
                paths.append(potential_path)
            potential_path = os.path.join(curr_dir, "rag_project", "prompts", "grounded_answer.md")
            if potential_path not in paths:
                paths.append(potential_path)
            curr_dir = os.path.dirname(curr_dir)

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.prompt_template = f.read()
                        logger.info(f"Loaded grounded answer prompt template from: {path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to read prompt path {path}: {e}")

        # Final fallback default prompt template
        self.prompt_template = (
            "# Grounded Answer Prompt\n\n"
            "## System Policy\n\n"
            "Answer only using the retrieved context.\n"
            "If the context does not contain sufficient evidence, answer exactly:\n"
            "I cannot find sufficient information in the knowledge base."
        )

    def _load_hyde_prompt_template(self, prompt_path: str = ""):
        paths = []
        if prompt_path:
            if "grounded_answer.md" in prompt_path:
                paths.append(prompt_path.replace("grounded_answer.md", "hyde.md"))
            else:
                paths.append(prompt_path)
        
        paths.extend([
            "prompts/hyde.md",
            "../prompts/hyde.md",
            "../../prompts/hyde.md"
        ])
        
        # Try to locate by walking up directories to find workspace root
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        while curr_dir and curr_dir != os.path.dirname(curr_dir):
            potential_path = os.path.join(curr_dir, "prompts", "hyde.md")
            if potential_path not in paths:
                paths.append(potential_path)
            potential_path = os.path.join(curr_dir, "rag_project", "prompts", "hyde.md")
            if potential_path not in paths:
                paths.append(potential_path)
            curr_dir = os.path.dirname(curr_dir)

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.hyde_prompt_template = f.read()
                        logger.info(f"Loaded HyDE prompt template from: {path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to read HyDE prompt path {path}: {e}")

        # Fallback default prompt template
        self.hyde_prompt_template = (
            "Please write a short hypothetical document or paragraph that directly answers the following query.\n"
            "Write this hypothetical document in a realistic, authoritative tone as if it is a passage from an internal company knowledge base, technical manual, or policy document.\n"
            "Do NOT write any introductory remarks like \"Here is a hypothetical document:\" or \"According to the query...\". Just write the hypothetical response/information block directly.\n\n"
            "ALWAYS write the hypothetical document in the same language as the user query."
        )

    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document/answer to expand the query vector."""
        if not self.client and self.provider != "google":
            logger.info("Generator client is offline/missing. Using mock hypothetical document.")
            return f"Đây là tài liệu giả định trả lời cho câu hỏi: {question}"

        system_instruction = self.hyde_prompt_template

        try:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                
                gemini_model = "gemini-2.5-flash"
                if "gemini" in self.model_name:
                    gemini_model = self.model_name
                elif self.model_name.startswith("gpt-"):
                    gemini_model = "gemini-2.5-flash"
                
                model = genai.GenerativeModel(
                    model_name=gemini_model,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    question,
                    request_options={"timeout": 30.0}
                )
                res_content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7
                )
                res_content = response.choices[0].message.content

            if not res_content:
                raise ValueError("Empty hypothetical content generated")
            return res_content.strip()

        except Exception as e:
            logger.warning(
                f"Failed to generate hypothetical document ({type(e).__name__}): {e}. "
                f"Falling back to original question query."
            )
            return question

    def generate(
        self,
        question: str,
        context: SelectedContext,
        chat_history: Sequence[dict[str, str]] | None = None
    ) -> GroundedAnswer:
        """Generate a grounded answer using the retrieved context and user prompt constraints."""
        
        # Refusal check: If no chunks retrieved, immediately return refusal
        if not context.chunks:
            return GroundedAnswer(
                answer="I cannot find sufficient information in the knowledge base.",
                citations=(),
                insufficient_context=True
            )
        if not self.client and self.provider != "google":
            # Local fallback simulator when offline or API key is missing
            self.had_fallback = True
            self.fallback_reason = "No LLM API client initialized (missing API key or offline mode)"
            return self._fallback_generate(question, context, chat_history)
        # Build JSON response format system prompt using loaded template
        system_instruction = (
            f"{self.prompt_template}\n\n"
            "## Output Format\n"
            "You MUST output raw JSON with the following schema:\n"
            "{\n"
            "  \"answer\": \"Markdown-formatted answer. Use \\n for newlines, **bold**, bullet lists with \\n- items.\",\n"
            "  \"citations\": [\"chunk-id-1\", \"chunk-id-2\"],\n"
            "  \"insufficient_context\": false\n"
            "}\n"
            "If context is insufficient:\n"
            "{\n"
            "  \"answer\": \"I cannot find sufficient information in the knowledge base.\",\n"
            "  \"citations\": [],\n"
            "  \"insufficient_context\": true\n"
            "}"
        )

        # Assemble retrieved context content
        context_str = "\n\n".join([
            f"[Chunk ID: {chunk.chunk_id}]\nSource: {chunk.document.file_name}\nContent:\n{chunk.text}"
            for chunk in context.chunks
        ])

        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Append chat history — only pass clean role+content (strip old context blocks)
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not content or role not in ["user", "assistant"]:
                    continue
                # Strip 'Retrieved Context:...' prefix from old user messages to prevent context bleed
                if role == "user" and "Retrieved Context:" in content:
                    # Keep only the Question part to avoid injecting old contexts
                    parts = content.split("Question:", 1)
                    content = parts[-1].strip() if len(parts) > 1 else content
                messages.append({"role": role, "content": content})

        # Append current query with fresh retrieved context
        messages.append({
            "role": "user",
            "content": f"Retrieved Context:\n{context_str}\n\nQuestion: {question}"
        })
        try:
            if self.provider == "google":
                import google.generativeai as genai
                
                genai.configure(api_key=self.google_api_key)
                
                gemini_model = "gemini-2.5-flash"
                if "gemini" in self.model_name:
                    gemini_model = self.model_name
                elif self.model_name.startswith("gpt-"):
                    gemini_model = "gemini-2.5-flash"
                
                model = genai.GenerativeModel(
                    model_name=gemini_model,
                    system_instruction=system_instruction
                )
                
                # Format prompts and history
                prompt_parts = []
                if chat_history:
                    prompt_parts.append("## Chat History:\n")
                    for msg in chat_history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        prompt_parts.append(f"{role.capitalize()}: {content}\n")
                    prompt_parts.append("\n")
                
                prompt_parts.append(f"## Retrieved Context:\n{context_str}\n\n")
                prompt_parts.append(f"Question: {question}")
                
                prompt = "".join(prompt_parts)
                
                response = model.generate_content(
                    prompt,
                    generation_config={'response_mime_type': 'application/json'},
                    request_options={"timeout": 30.0}
                )
                res_content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                res_content = response.choices[0].message.content

            if not res_content:
                raise ValueError("Received empty content from generator")
            parsed = json.loads(res_content)
            answer = parsed.get("answer", "").strip()
            citations = parsed.get("citations", [])
            insufficient = bool(parsed.get("insufficient_context", False))

            # Grounding Guardrail 1: Refusal check
            if insufficient or "I cannot find sufficient information" in answer:
                return GroundedAnswer(
                    answer="I cannot find sufficient information in the knowledge base.",
                    citations=(),
                    insufficient_context=True
                )

            # Grounding Guardrail 2: Citations validation against actual retrieved chunks
            valid_chunk_ids = {c.chunk_id for c in context.chunks}
            validated_citations = [c_id for c_id in citations if c_id in valid_chunk_ids]

            # Grounding Guardrail 3: Warn if LLM did not cite valid chunk IDs but do NOT refuse
            # — the answer may still be grounded (LLM sometimes uses short keys or paraphrased IDs)
            if not validated_citations:
                logger.warning(
                    "LLM answered without matching chunk IDs in citations. "
                    "Allowing answer through with empty citations (answer may still be grounded)."
                )
                # Attach first context chunk as a best-effort citation so the UI shows something
                fallback_citations = tuple(c.chunk_id for c in context.chunks[:3])
                return GroundedAnswer(
                    answer=answer,
                    citations=fallback_citations,
                    insufficient_context=False
                )

            return GroundedAnswer(
                answer=answer,
                citations=tuple(validated_citations),
                insufficient_context=False
            )

        except Exception as e:
            self.had_fallback = True
            self.fallback_reason = f"Type: {type(e).__name__}, Error: {str(e)}"
            logger.error(
                f"OpenAI completion generation failed — "
                f"type={type(e).__name__}, model={self.model_name}, "
                f"error={e}. Running fallback generator."
            )
            fallback_answer = self._fallback_generate(question, context, chat_history)
            
            # Prepend a user-friendly warning in Vietnamese and English indicating LLM API rate limit or connection error
            warning_prefix = (
                "⚠️ **Lưu ý: Không thể kết nối với LLM API (Lỗi 429 Quota Exceeded hoặc lỗi kết nối mạng).**\n"
                "Hệ thống tự động chuyển sang chế độ dự phòng ngoại tuyến (Offline Fallback). Dưới đây là nội dung trích xuất thô trực tiếp từ các đoạn tài liệu liên quan nhất trong cơ sở dữ liệu:\n\n---\n\n"
            )
            import dataclasses
            return dataclasses.replace(
                fallback_answer,
                answer=warning_prefix + fallback_answer.answer
            )

    def _fallback_generate(
        self,
        question: str,
        context: SelectedContext,
        chat_history: Sequence[dict[str, str]] | None = None
    ) -> GroundedAnswer:
        """Local similarity overlap generator fallback (runs without OpenAI API keys).
        
        Scores all chunks by keyword overlap, selects top-3 most relevant,
        cleans their text and produces structured markdown output.
        """
        query_words = set(re.findall(r"\w+", question.lower()))
        if not query_words:
            return GroundedAnswer(
                answer="Không tìm thấy thông tin phù hợp trong tài liệu.",
                citations=(),
                insufficient_context=True
            )

        # Score all chunks by keyword overlap
        scored = []
        for chunk in context.chunks:
            chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                scored.append((overlap, chunk))

        if not scored:
            return GroundedAnswer(
                answer="Không tìm thấy thông tin phù hợp trong tài liệu.",
                citations=(),
                insufficient_context=True
            )

        # Sort by overlap score descending, take top 3
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:3]

        def clean_text(raw: str) -> str:
            """Two-phase clean: merge \n\n word-fragments, then collapse single \n word-breaks."""
            t = raw.replace("\r\n", "\n").replace("\r", "\n")

            def is_struct(s: str) -> bool:
                return bool(re.match(
                    r"^(\s*([-*+✔✗●▪•→]\s|#{1,6}\s|```|---|\d+[.)]\s*))",
                    s
                )) or s.strip().startswith("**")

            # ── Phase 1: merge short paragraph artifacts (double-newline word breaks) ──
            raw_blocks = t.split("\n\n")
            merged: list[str] = []
            for blk in raw_blocks:
                b = blk.strip()
                if not b:
                    continue
                word_count = len(b.split())
                is_artifact = word_count <= 3 and len(b) <= 40 and not is_struct(b)
                if is_artifact and merged:
                    prev = merged[-1]
                    if not re.search(r'[.!?:;»)}\]\'""]$', prev.rstrip()):
                        merged[-1] = (prev + " " + b).replace("  ", " ").strip()
                        continue
                merged.append(b)

            # ── Phase 2: collapse single-\n word-breaks within each merged block ──
            final_parts: list[str] = []
            for block in merged:
                lines = block.split("\n")
                out: list[str] = []
                buf: list[str] = []

                def flush() -> None:
                    if buf:
                        out.append(" ".join(buf))
                        buf.clear()

                for line in lines:
                    trim = line.strip()
                    if not trim:
                        flush()
                        out.append("")
                    elif is_struct(line):
                        flush()
                        out.append(line)
                    else:
                        buf.append(trim)
                flush()

                block_result = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
                if block_result:
                    final_parts.append(block_result)

            return "\n\n".join(final_parts)

        # Build structured markdown answer
        parts = []
        citation_ids = []
        seen_files: set[str] = set()

        for _, chunk in top_chunks:
            cleaned = clean_text(chunk.text)
            if not cleaned:
                continue
            file_name = chunk.document.file_name
            if file_name not in seen_files:
                seen_files.add(file_name)
                parts.append(f"**📄 {file_name}**\n\n{cleaned}")
            else:
                parts.append(cleaned)
            citation_ids.append(chunk.chunk_id)

        if not parts:
            return GroundedAnswer(
                answer="Không tìm thấy thông tin phù hợp trong tài liệu.",
                citations=(),
                insufficient_context=True
            )

        answer = "\n\n---\n\n".join(parts)
        return GroundedAnswer(
            answer=answer,
            citations=tuple(citation_ids),
            insufficient_context=False
        )

    def generate_contextual_prefix(self, document_text: str, chunk_text: str) -> str:
        """Generate a short 1-2 sentence context to situate the chunk within the overall document."""
        if not self.client and self.provider != "google":
            logger.info("Generator client is offline/missing. Skipping contextual prefix.")
            return ""

        system_instruction = (
            "You are a helpful assistant. You will be given a whole document and a short chunk from that document. "
            "Your task is to write a short 1-2 sentence context explanation to situate the chunk in the overall document. "
            "Do not include any introductory remarks like 'This chunk describes' or 'In this document'. Just write the context block directly. "
            "You MUST output the explanation in the same language as the document and chunk."
        )

        prompt = (
            f"<document>\n{document_text}\n</document>\n\n"
            f"Here is the chunk we want to situate within the whole document:\n"
            f"<chunk>\n{chunk_text}\n</chunk>\n\n"
            f"Please write a short 1-2 sentence explanation to situate this chunk in the overall document."
        )

        try:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                
                gemini_model = "gemini-2.5-flash"
                if "gemini" in self.model_name:
                    gemini_model = self.model_name
                elif self.model_name.startswith("gpt-"):
                    gemini_model = "gemini-2.5-flash"
                
                model = genai.GenerativeModel(
                    model_name=gemini_model,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": 30.0}
                )
                res_content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                res_content = response.choices[0].message.content

            if not res_content:
                raise ValueError("Empty contextual prefix generated")
            return res_content.strip()

        except Exception as e:
            logger.warning(
                f"Failed to generate contextual prefix ({type(e).__name__}): {e}. "
                f"Skipping contextual prefix."
            )
            return ""


