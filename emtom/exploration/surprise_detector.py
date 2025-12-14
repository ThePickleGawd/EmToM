"""
LLM-based surprise detection for EMTOM benchmark.

Uses LLM self-report to identify when agents encounter unexpected behaviors
that might indicate interesting mechanics worth investigating.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SurpriseAssessment:
    """Result of surprise detection."""

    is_surprised: bool
    level: int  # 1-5 scale
    explanation: str
    hypothesis: str  # Agent's guess about why this happened
    raw_response: str = ""


class SurpriseDetector:
    """
    LLM-based surprise detection through self-report.

    After each action, asks the LLM to evaluate whether the outcome
    was surprising and why. This identifies moments where the agent
    has discovered unexpected behaviors in the world.
    """

    PROMPT_TEMPLATE = """You just performed an action and observed the result.

Action: {action}
Target: {target}
Expected outcome: {expected}
Actual outcome: {actual}
{trigger_note}

Were you surprised by this outcome? Consider:
- Did the result match your expectations?
- Did anything unusual or unexpected happen?
- Does this reveal something about how this world works?

Respond in JSON format:
{{
    "is_surprised": true/false,
    "level": 1-5 (1=slightly unexpected, 5=completely unexpected),
    "explanation": "<why you were or weren't surprised>",
    "hypothesis": "<if surprised, what might explain this behavior?>"
}}"""

    def __init__(
        self,
        llm_client: Any,
        surprise_threshold: float = 0.5,
        require_hypothesis: bool = True,
    ):
        """
        Initialize the surprise detector.

        Args:
            llm_client: LLM client with generate(prompt) method
            surprise_threshold: Minimum surprise level to report (0-1)
            require_hypothesis: Whether to require hypothesis in response
        """
        self.llm = llm_client
        self.surprise_threshold = surprise_threshold
        self.require_hypothesis = require_hypothesis

    def assess_surprise(
        self,
        agent_id: str,
        action: str,
        target: Optional[str],
        expected: str,
        actual: str,
        trigger: Optional[str] = None,
    ) -> SurpriseAssessment:
        """
        Assess whether an action outcome was surprising.

        Args:
            agent_id: ID of the agent
            action: Action that was performed
            target: Target of the action
            expected: What the agent expected to happen
            actual: What actually happened
            trigger: Optional system hint about surprise

        Returns:
            SurpriseAssessment with surprise level and explanation
        """
        trigger_note = ""
        if trigger:
            trigger_note = f"\nSystem note: {trigger}"

        prompt = self.PROMPT_TEMPLATE.format(
            action=action,
            target=target or "none",
            expected=expected or "default behavior",
            actual=actual,
            trigger_note=trigger_note,
        )

        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> SurpriseAssessment:
        """Parse LLM response into SurpriseAssessment."""
        try:
            # Find JSON in response
            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return SurpriseAssessment(
                    is_surprised=bool(data.get("is_surprised", False)),
                    level=int(data.get("level", 1)),
                    explanation=data.get("explanation", ""),
                    hypothesis=data.get("hypothesis", ""),
                    raw_response=response,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to infer from text
        is_surprised = any(
            word in response.lower()
            for word in ["surprised", "unexpected", "strange", "unusual"]
        )

        return SurpriseAssessment(
            is_surprised=is_surprised,
            level=3 if is_surprised else 1,
            explanation="Could not parse structured response",
            hypothesis="",
            raw_response=response,
        )

    def is_significant_surprise(self, assessment: SurpriseAssessment) -> bool:
        """Check if a surprise meets the significance threshold."""
        if not assessment.is_surprised:
            return False
        # Normalize level to 0-1 scale
        normalized_level = (assessment.level - 1) / 4.0
        return normalized_level >= self.surprise_threshold


