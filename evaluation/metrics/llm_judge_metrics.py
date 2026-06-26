"""Tier 3 — Custom LLM-as-Judge metrics.

Uses a configurable LLM (OpenAI or Google Gemini) to evaluate
answer quality on dimensions that require semantic understanding:

- Faithfulness Score (0–1): Are all claims grounded in context?
- Completeness Score (0–1): Does the answer cover the question fully?
- Language Consistency (bool): Does the answer language match the question?
- Formatting Quality (1–5): Is the markdown formatting clean and readable?
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

from evaluation.dataset import EvalResult

logger = logging.getLogger("evaluation.metrics.llm_judge")


# ──────────────────────────────────────────────────────────────────────
# Prompt loader
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_JUDGE_PROMPT = """\
You are an expert evaluation judge for a RAG (Retrieval-Augmented Generation) chatbot.
You will evaluate the quality of an AI-generated answer given the user question and the retrieved context.

## Evaluation Criteria

Score each dimension independently:

1. **faithfulness** (0.0 – 1.0): Are ALL factual claims in the answer directly supported by the retrieved context? 
   - 1.0 = every claim is grounded
   - 0.5 = some claims are grounded, some are unsupported
   - 0.0 = answer contradicts or fabricates information

2. **completeness** (0.0 – 1.0): Does the answer fully address all aspects of the question using available context?
   - 1.0 = comprehensive, all relevant information from context is used
   - 0.5 = partially answers the question
   - 0.0 = does not address the question at all

3. **language_consistent** (true/false): Is the answer written in the SAME language as the question?
   - true = answer language matches question language
   - false = language mismatch (e.g., question in Vietnamese but answer in English)

4. **formatting_quality** (1 – 5): Is the answer well-formatted using markdown?
   - 5 = excellent: proper headings, lists, bold, tables where needed
   - 3 = adequate: readable but could be better structured
   - 1 = poor: wall of text, no structure

## Input

**Question:** {question}

**Retrieved Context:**
{context}

**Generated Answer:**
{answer}

## Output Format

Return ONLY a raw JSON object with no explanation:
{{"faithfulness": 0.85, "completeness": 0.9, "language_consistent": true, "formatting_quality": 4}}
"""


def _load_judge_prompt(prompt_path: str) -> str:
    """Load the judge prompt template from file, falling back to built-in default."""
    path = Path(prompt_path)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load judge prompt from {path}: {e}. Using default.")
    return _DEFAULT_JUDGE_PROMPT


# ──────────────────────────────────────────────────────────────────────
# LLM Judge implementation
# ──────────────────────────────────────────────────────────────────────


class LLMJudge:
    """LLM-based evaluation judge supporting OpenAI and Google Gemini providers."""

    def __init__(
        self,
        provider: str = "google",
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        prompt_path: str = "evaluation/prompts/judge_faithfulness.md",
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.prompt_template = _load_judge_prompt(prompt_path)
        self._client = None

        if self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI package not installed for LLM judge.")
        elif self.provider == "google" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                logger.warning("google-generativeai package not installed for LLM judge.")

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the configured LLM and return raw response text."""
        if self.provider == "google" and self._client:
            model = self._client.GenerativeModel(model_name=self.model)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": 60.0},
            )
            return response.text or ""

        elif self.provider == "openai" and self._client:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        else:
            raise RuntimeError(
                f"LLM Judge is not configured. Provider={self.provider}, "
                f"API key={'set' if self.api_key else 'missing'}"
            )

    def judge_single(self, result: EvalResult) -> dict[str, float | bool]:
        """Evaluate a single EvalResult using the LLM judge.

        Returns:
            Dictionary with judge scores.  On failure, returns default low scores.
        """
        default_scores: dict[str, float | bool] = {
            "judge_faithfulness": 0.0,
            "judge_completeness": 0.0,
            "judge_language_consistent": False,
            "judge_formatting_quality": 1.0,
        }

        try:
            prompt = self.prompt_template.format(
                question=result.sample.question,
                context=result.context_text or "\n".join(result.retrieved_texts),
                answer=result.generated_answer,
            )

            raw_response = self._call_llm(prompt)

            # Parse JSON from response (strip markdown code fences if present)
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                # Remove code fences
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            parsed = json.loads(cleaned)

            scores: dict[str, float | bool] = {
                "judge_faithfulness": float(parsed.get("faithfulness", 0.0)),
                "judge_completeness": float(parsed.get("completeness", 0.0)),
                "judge_language_consistent": bool(parsed.get("language_consistent", False)),
                "judge_formatting_quality": float(parsed.get("formatting_quality", 1)),
            }

            result.scores.update(scores)
            return scores

        except Exception as e:
            logger.error(f"LLM judge failed for question '{result.sample.question[:50]}...': {e}")
            result.scores.update(default_scores)
            return default_scores


def compute_judge_metrics(
    results: Sequence[EvalResult],
    provider: str = "google",
    model: str = "gemini-2.5-flash",
    api_key: str = "",
    prompt_path: str = "evaluation/prompts/judge_faithfulness.md",
) -> dict[str, float]:
    """Run LLM judge evaluation on all results.

    Returns:
        Dictionary with averaged judge metric scores.
    """
    empty_scores = {
        "avg_judge_faithfulness": 0.0,
        "avg_judge_completeness": 0.0,
        "avg_judge_language_consistency": 0.0,
        "avg_judge_formatting_quality": 0.0,
    }

    if not results:
        return empty_scores

    judge = LLMJudge(
        provider=provider,
        model=model,
        api_key=api_key,
        prompt_path=prompt_path,
    )

    for i, result in enumerate(results):
        logger.info(f"Judging sample {i + 1}/{len(results)}: {result.sample.question[:60]}...")
        judge.judge_single(result)

    n = len(results)
    return {
        "avg_judge_faithfulness": sum(
            r.scores.get("judge_faithfulness", 0.0) for r in results
        ) / n,
        "avg_judge_completeness": sum(
            r.scores.get("judge_completeness", 0.0) for r in results
        ) / n,
        "avg_judge_language_consistency": sum(
            1 for r in results if r.scores.get("judge_language_consistent", False)
        ) / n,
        "avg_judge_formatting_quality": sum(
            r.scores.get("judge_formatting_quality", 0.0) for r in results
        ) / n,
    }
