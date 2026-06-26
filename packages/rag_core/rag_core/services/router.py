import json
import logging
from typing import Sequence
from openai import OpenAI

from rag_core.contracts.types import RoutedQuestion
from rag_core.ports.interfaces import RouterPort

logger = logging.getLogger("rag_core.router")


class DomainRouter(RouterPort):
    """Hierarchical domain router that categorizes queries into knowledge branches."""

    def __init__(self, available_branches: Sequence[tuple[str, ...]], openai_api_key: str = "", model_name: str = "gpt-4o-mini"):
        self.available_branches = available_branches
        self.model_name = model_name
        self.openai_client = None

        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

    def route(self, workspace_id: str, question: str) -> RoutedQuestion:
        """Route the question to the most relevant knowledge branch."""
        if not self.available_branches:
            logger.info("No branches available to route to. Defaulting to ROOT branch.")
            return RoutedQuestion(
                question=question,
                workspace_id=workspace_id,
                branch_path=(),
                confidence=1.0
            )

        # Attempt LLM routing if client is active
        if self.openai_client:
            try:
                routed = self._llm_route(workspace_id, question)
                if routed:
                    return routed
            except Exception as e:
                logger.warning(f"LLM routing failed: {e}. Falling back to keyword routing.")

        # Fallback to local heuristic keyword-based routing
        return self._keyword_route(workspace_id, question)

    def _llm_route(self, workspace_id: str, question: str) -> RoutedQuestion | None:
        """Use OpenAI completion to select the best domain branch path."""
        branches_str = "\n".join([str(list(b)) for b in self.available_branches])
        
        system_prompt = (
            "You are an enterprise search query router. Your task is to route the user's question "
            "to the most specific knowledge branch path from the available list below.\n\n"
            f"Available Branches:\n{branches_str}\n\n"
            "If the question does not clearly fit into any of the available branches, select an empty list [] "
            "which represents the root search domain.\n\n"
            "Return the output as raw JSON with the following structure:\n"
            '{\n  "branch_path": ["parent", "child"],\n  "confidence": 0.95,\n  "rationale": "Explanation here"\n}'
        )

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: '{question}'"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        content = response.choices[0].message.content
        if not content:
            return None

        result = json.loads(content)
        path_list = result.get("branch_path", [])
        branch_path = tuple(path_list)
        confidence = float(result.get("confidence", 0.0))

        # Ensure routed branch exists in available branches list
        if branch_path not in self.available_branches and branch_path != ():
            logger.warning(f"LLM returned invalid branch: {branch_path}. Defaulting to ROOT.")
            branch_path = ()
            confidence = 0.5

        return RoutedQuestion(
            question=question,
            workspace_id=workspace_id,
            branch_path=branch_path,
            confidence=confidence
        )

    def _keyword_route(self, workspace_id: str, question: str) -> RoutedQuestion:
        """Route queries by simple text overlap matching when LLM is unavailable."""
        import re
        lower_q = question.lower()
        q_tokens = re.findall(r'\w+', lower_q)
        best_branch = ()
        max_score = 0

        # Score based on token matches and stem overlap in branch names
        for branch in self.available_branches:
            score = 0
            for segment in branch:
                seg_lower = segment.lower()
                # 1. Direct substring check
                if seg_lower in lower_q:
                    score += len(seg_lower) * 2
                else:
                    # 2. Check for token prefix/stem overlap
                    for token in q_tokens:
                        common_len = 0
                        for i in range(min(len(seg_lower), len(token))):
                            if seg_lower[i] == token[i]:
                                common_len += 1
                            else:
                                break
                        # If common prefix length is >= 4, or matches the full short word (min length >= 3)
                        if common_len >= 4:
                            score += common_len
                            break
                        elif common_len >= 3 and common_len >= min(len(seg_lower), len(token)) - 1:
                            score += common_len
                            break

            if score > max_score:
                max_score = score
                best_branch = branch

        confidence = 0.8 if max_score > 0 else 1.0  # 1.0 confidence for root fallback

        return RoutedQuestion(
            question=question,
            workspace_id=workspace_id,
            branch_path=best_branch,
            confidence=confidence
        )

