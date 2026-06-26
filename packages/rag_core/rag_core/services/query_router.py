"""Intelligent query router that classifies questions and decides retrieval strategy.

Extends the existing DomainRouter by adding multi-source strategy decisions:
which sources to query (KB, Web, KG) in addition to which knowledge branch.
"""

import json
import logging
import os
from typing import Sequence

from rag_core.contracts.types import QueryRoute, RetrievalStrategy, RoutedQuestion
from rag_core.ports.interfaces import QueryRouterPort
from rag_core.services.router import DomainRouter

logger = logging.getLogger("rag_core.query_router")


class QueryRouter(QueryRouterPort):
    """LLM-based query router that determines retrieval strategy and knowledge branch.

    Falls back to DomainRouter (kb_only strategy) when LLM is unavailable or
    when multi-source routing is disabled.
    """

    def __init__(
        self,
        available_branches: Sequence[tuple[str, ...]],
        openai_api_key: str = "",
        google_api_key: str = "",
        model_name: str = "gpt-4o-mini",
        prompt_path: str = "",
    ):
        self.available_branches = available_branches
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key

        # Determine LLM provider
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if self.provider == "openai" and not openai_api_key and google_api_key:
            self.provider = "google"

        # Initialise OpenAI client
        self.openai_client = None
        if self.provider == "openai" and openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)

        # Keep a DomainRouter for fallback
        self._domain_router = DomainRouter(
            available_branches=available_branches,
            openai_api_key=openai_api_key,
            model_name=model_name,
        )

        # Load prompt template
        self.prompt_template = ""
        self._load_prompt_template(prompt_path)

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_prompt_template(self, prompt_path: str = "") -> None:
        paths = []
        if prompt_path:
            paths.append(prompt_path)

        paths.extend([
            "prompts/query_router.md",
            "../prompts/query_router.md",
            "../../prompts/query_router.md",
        ])

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        while curr_dir and curr_dir != os.path.dirname(curr_dir):
            candidate = os.path.join(curr_dir, "prompts", "query_router.md")
            if candidate not in paths:
                paths.append(candidate)
            candidate2 = os.path.join(curr_dir, "rag_project", "prompts", "query_router.md")
            if candidate2 not in paths:
                paths.append(candidate2)
            curr_dir = os.path.dirname(curr_dir)

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.prompt_template = f.read()
                        logger.info(f"Loaded query router prompt template from: {path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to read prompt path {path}: {e}")

        # Fallback minimal prompt
        self.prompt_template = (
            "Classify the user question and decide which retrieval strategy to use.\n"
            "Output JSON with keys: strategy, branch_path, confidence, reasoning, sub_queries."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        workspace_id: str,
        question: str,
        available_branches: Sequence[tuple[str, ...]] | None = None,
        web_search_enabled: bool = False,
        kg_available: bool = False,
    ) -> QueryRoute:
        """Classify question and decide retrieval strategy + knowledge branch."""

        branches = available_branches if available_branches is not None else self.available_branches

        # If neither web nor KG is enabled, skip LLM routing and use old DomainRouter
        if not web_search_enabled and not kg_available:
            return self._fallback_route(workspace_id, question)

        # Attempt LLM routing
        if self.openai_client or self.provider == "google":
            try:
                result = self._llm_route(
                    workspace_id, question, branches,
                    web_search_enabled, kg_available,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"QueryRouter LLM call failed: {e}. Falling back to DomainRouter.")

        return self._fallback_route(workspace_id, question)

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    def _llm_route(
        self,
        workspace_id: str,
        question: str,
        branches: Sequence[tuple[str, ...]],
        web_search_enabled: bool,
        kg_available: bool,
    ) -> QueryRoute | None:
        """Use LLM to classify question into a retrieval strategy."""

        branches_str = "\n".join([str(list(b)) for b in branches]) if branches else "(no branches)"

        # Build available strategies list based on enabled features
        available_strategies = ["kb_only"]
        if web_search_enabled:
            available_strategies.extend(["web_search", "kb_and_web"])
        if kg_available:
            available_strategies.extend(["knowledge_graph", "kb_and_kg"])
        if web_search_enabled and kg_available:
            available_strategies.append("all")

        strategies_str = ", ".join(available_strategies)

        system_prompt = (
            f"{self.prompt_template}\n\n"
            f"## Available Knowledge Branches\n{branches_str}\n\n"
            f"## Enabled Strategies\nOnly choose from: {strategies_str}\n"
        )

        try:
            if self.provider == "google":
                content = self._call_google(system_prompt, question)
            else:
                content = self._call_openai(system_prompt, question)

            if not content:
                return None

            parsed = json.loads(content)
            strategy_str = parsed.get("strategy", "kb_only")

            # Validate strategy is one of the enabled ones
            try:
                strategy = RetrievalStrategy(strategy_str)
            except ValueError:
                logger.warning(f"LLM returned unknown strategy '{strategy_str}'. Defaulting to kb_only.")
                strategy = RetrievalStrategy.KB_ONLY

            if strategy_str not in available_strategies:
                logger.warning(f"LLM chose disabled strategy '{strategy_str}'. Defaulting to kb_only.")
                strategy = RetrievalStrategy.KB_ONLY

            branch_list = parsed.get("branch_path", [])
            branch_path = tuple(branch_list)
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", ""))
            sub_queries = tuple(parsed.get("sub_queries", []))

            # Validate branch exists
            if branch_path and branch_path not in branches and branch_path != ():
                logger.warning(f"LLM returned invalid branch: {branch_path}. Defaulting to root.")
                branch_path = ()

            return QueryRoute(
                question=question,
                workspace_id=workspace_id,
                strategy=strategy,
                branch_path=branch_path,
                confidence=confidence,
                reasoning=reasoning,
                sub_queries=sub_queries,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse QueryRouter LLM response as JSON: {e}")
            return None

    def _call_openai(self, system_prompt: str, question: str) -> str | None:
        """Call OpenAI chat completion."""
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: '{question}'"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _call_google(self, system_prompt: str, question: str) -> str | None:
        """Call Google Gemini for routing."""
        import google.generativeai as genai

        api_key = self.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set for QueryRouter.")
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
            f"Question: '{question}'",
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 30.0},
        )
        return response.text

    # ------------------------------------------------------------------
    # Fallback: wrap existing DomainRouter result into QueryRoute
    # ------------------------------------------------------------------

    def _fallback_route(self, workspace_id: str, question: str) -> QueryRoute:
        """Use existing DomainRouter and wrap result as kb_only QueryRoute."""
        routed: RoutedQuestion = self._domain_router.route(workspace_id, question)
        return QueryRoute(
            question=routed.question,
            workspace_id=routed.workspace_id,
            strategy=RetrievalStrategy.KB_ONLY,
            branch_path=routed.branch_path,
            confidence=routed.confidence,
            reasoning="Fallback to domain-only routing (multi-source routing disabled or LLM unavailable)",
        )

    # ------------------------------------------------------------------
    # Utility: convert QueryRoute → RoutedQuestion for backward compat
    # ------------------------------------------------------------------

    @staticmethod
    def to_routed_question(route: QueryRoute) -> RoutedQuestion:
        """Convert a QueryRoute to a legacy RoutedQuestion for existing pipeline compat."""
        return RoutedQuestion(
            question=route.question,
            workspace_id=route.workspace_id,
            branch_path=route.branch_path,
            confidence=route.confidence,
        )
