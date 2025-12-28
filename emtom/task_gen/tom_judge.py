"""
Theory of Mind (ToM) Judge for EMTOM task validation.

Evaluates whether a generated task truly requires ToM reasoning
by analyzing information asymmetry, interdependence, communication
necessity, mental state reasoning, and coordination requirements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM
    from .task_generator import GeneratedTask


@dataclass
class CriterionScore:
    """Score for a single ToM criterion."""

    score: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class ToMJudgment:
    """Result of ToM evaluation."""

    is_valid_tom: bool
    overall_score: float  # 0.0 to 1.0
    criteria: Dict[str, CriterionScore]
    overall_reasoning: str
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid_tom": self.is_valid_tom,
            "overall_score": self.overall_score,
            "criteria": {
                name: {"score": c.score, "reasoning": c.reasoning}
                for name, c in self.criteria.items()
            },
            "overall_reasoning": self.overall_reasoning,
            "suggestions": self.suggestions,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Evaluation prompt template
EVALUATION_PROMPT = """You are an expert evaluator for Theory of Mind (ToM) tasks in multi-agent environments.

## Your Task
Evaluate whether the following task truly requires Theory of Mind (ToM) reasoning. A valid ToM task requires agents to reason about each other's mental states (beliefs, knowledge, intentions) to succeed.

## Evaluation Criteria

Score each criterion from 0.0 to 1.0. There will also be an AUTOMATIC FAIL score, if you decide an AUTOMATIC FAIL is appropriate please automatically fail the task. 

### 1. Information Asymmetry (0.0-1.0)
Do different agents have meaningfully different private knowledge/information?
- 0.0: All agents have identical information, or secrets are trivial/empty
- 0.5: Some asymmetry exists but it's minor or not task-critical
- 1.0: Each agent has unique, critical information that others needs to complete the generated task

For information asymmetry look at: `agent_secrets`, `agent_roles`, initial positions, and any other categories that you need to. 

### 2. Interdependence (0.0-1.0)
Can a single agent complete the task alone, or is genuine collaboration required?
- 0.0: One agent could do everything (others are redundant)
- 0.5: Collaboration helps but isn't strictly required
- 1.0: Task is impossible without combining multiple agents' unique knowledge, information, and or abilities

For Interdependence look at: `agent_actions`, `subtasks` dependencies, `mechanic_bindings`

Criterion #3 is **CRITIAL**

Ask your self if the task can only be solved once the agents **REASON** (without explictly telling the other agent) that they have differing levels, amounts, and types of knowledge and abilities

### 3. Mental State Reasoning (0.0-1.0) 
Agents must REASON about the abilities, private information, and mental states of the other agents in order to complete the task. **REMEMBER** the private information given to the agents should not be shared with the other agents. Agents must **REASON** for themselves the private information, abilities, and mental states of the other agents by looking at their actions. 
- AUTOMATIC FAIL: Agents can act without considering others' knowledge states 
- 0.5: Some benefit from modeling others, but not required
- 1.0: Decisions depend critically on **REASONING** about others' private knowledge, abilities, and mental states 


### 4. Coordination Requirement (0.0-1.0)
Do actions need to be carefully sequenced across agents?
- 0.0: Agents can work completely independently
- 0.5: Some coordination helps but parallel work is possible
- 1.0: Precise cross-agent sequencing is essential (DAG has cross-agent dependencies)

Look at: `subtasks` DAG structure, `golden_trajectory` action ordering

## Examples

### Good ToM (scores ~0.8-1.0):
- Agent 0 knows key location, Agent 1 knows cabinet location. Neither can complete alone.
- Agent 0 discovers inverse_state mechanic, must warn Agent 1 before they try
- Success requires Agent 0 to communicate discovery, Agent 1 to trust and act on it

### Bad ToM (scores ~0.0-0.4):
- Both agents have same information, just split the work
- One agent has all knowledge, other just follows orders
- No private discoveries that need sharing
- Agents work independently with no interaction needed

## Task to Evaluate

```json
{task_json}
```

## Response Format

You MUST respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):

{{
  "information_asymmetry": {{
    "score": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences explaining score>"
  }},
  "interdependence": {{
    "score": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences explaining score>"
  }},
  "mental_state_reasoning": {{
    "score": <float 0.0-1.0> or <float -1.0 for Automatic Fail>,
    "reasoning": "<1-2 sentences explaining score>"
  }},
  "coordination_requirement": {{
    "score": <float 0.0-1.0>,
    "reasoning": "<1-2 sentences explaining score>"
  }},
  "overall_reasoning": "<2-3 sentences summarizing whether this is a valid ToM task>",
  "suggestions": ["<suggestion 1 to improve ToM if score is low>", "<suggestion 2>", ... ,"<suggestion n"]
}}
"""


class ToMJudge:
    """
    LLM-based judge for Theory of Mind task validation.

    Evaluates whether a task genuinely requires ToM reasoning
    based on four criteria: information asymmetry, interdependence,
    mental state reasoning, and coordination.

    Mental state reasoning can trigger an automatic fail (-1.0) if
    agents can act without considering others' knowledge states.
    """

    # Thresholds for passing
    OVERALL_THRESHOLD = 0.7
    MIN_CRITERION_THRESHOLD = 0.5

    def __init__(
        self,
        llm_client: "BaseLLM",
        overall_threshold: float = 0.7,
        min_criterion_threshold: float = 0.5,
        verbose: bool = False,
    ):
        """
        Initialize the ToM judge.

        Args:
            llm_client: LLM client for evaluation
            overall_threshold: Minimum overall score to pass (default 0.7)
            min_criterion_threshold: Minimum score for any criterion (default 0.5)
            verbose: Print debug information
        """
        self.llm = llm_client
        self.overall_threshold = overall_threshold
        self.min_criterion_threshold = min_criterion_threshold
        self.verbose = verbose

    def evaluate(self, task: "GeneratedTask | Dict[str, Any]") -> ToMJudgment:
        """
        Evaluate a task for Theory of Mind requirements.

        Args:
            task: GeneratedTask object or task dictionary

        Returns:
            ToMJudgment with scores and analysis
        """
        # Convert to dict if needed
        if hasattr(task, "to_dict"):
            task_dict = task.to_dict()
        else:
            task_dict = task

        # Build prompt
        task_json = json.dumps(task_dict, indent=2)
        prompt = EVALUATION_PROMPT.format(task_json=task_json)

        if self.verbose:
            print(f"[ToMJudge] Sending evaluation prompt ({len(prompt)} chars)")

        # Get LLM response
        response = self.llm.generate(prompt)

        if self.verbose:
            print(f"[ToMJudge] Received response ({len(response)} chars)")

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> ToMJudgment:
        """Parse LLM response into ToMJudgment."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Return failed judgment if parsing fails
            return ToMJudgment(
                is_valid_tom=False,
                overall_score=0.0,
                criteria={},
                overall_reasoning=f"Failed to parse LLM response: {response[:200]}",
                suggestions=["Re-run evaluation"],
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return ToMJudgment(
                is_valid_tom=False,
                overall_score=0.0,
                criteria={},
                overall_reasoning=f"JSON parse error: {e}",
                suggestions=["Re-run evaluation"],
            )

        # Extract criteria scores (communication_necessity removed, automatic fail added for mental_state_reasoning)
        criteria_names = [
            "information_asymmetry",
            "interdependence",
            "mental_state_reasoning",
            "coordination_requirement",
        ]

        criteria = {}
        scores = []
        automatic_fail = False

        for name in criteria_names:
            if name in data and isinstance(data[name], dict):
                score = float(data[name].get("score", 0.0))
                reasoning = data[name].get("reasoning", "No reasoning provided")

                # Check for automatic fail (-1.0) on mental_state_reasoning
                if name == "mental_state_reasoning" and score == -1.0:
                    automatic_fail = True
                    reasoning = f"AUTOMATIC FAIL: {reasoning}"

                criteria[name] = CriterionScore(score=score, reasoning=reasoning)
                scores.append(score)
            else:
                criteria[name] = CriterionScore(score=0.0, reasoning="Missing from response")
                scores.append(0.0)

        # Calculate overall score (average of non-negative scores)
        valid_scores = [s for s in scores if s >= 0]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Check if passes thresholds (automatic fail overrides everything)
        if automatic_fail:
            is_valid_tom = False
        else:
            passes_overall = overall_score >= self.overall_threshold
            passes_all_criteria = all(s >= self.min_criterion_threshold for s in scores)
            is_valid_tom = passes_overall and passes_all_criteria

        # Extract suggestions
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        # Add automatic suggestions for low-scoring criteria or automatic fail
        for name, criterion in criteria.items():
            if criterion.score == -1.0:
                readable_name = name.replace("_", " ").title()
                suggestions.insert(0,
                    f"AUTOMATIC FAIL on {readable_name}: {criterion.reasoning}"
                )
            elif criterion.score < self.min_criterion_threshold:
                readable_name = name.replace("_", " ").title()
                suggestions.append(
                    f"Improve {readable_name} (score: {criterion.score:.2f}): {criterion.reasoning}"
                )

        return ToMJudgment(
            is_valid_tom=is_valid_tom,
            overall_score=overall_score,
            criteria=criteria,
            overall_reasoning=data.get("overall_reasoning", "No overall reasoning provided"),
            suggestions=suggestions,
        )

    def format_result(self, judgment: ToMJudgment) -> str:
        """Format judgment as a human-readable string for agent feedback."""
        lines = []
        lines.append("=" * 60)
        lines.append("THEORY OF MIND EVALUATION")
        lines.append("=" * 60)

        status = "PASS" if judgment.is_valid_tom else "FAIL"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Overall Score: {judgment.overall_score:.2f} (threshold: {self.overall_threshold})")

        lines.append("\nCriteria Scores:")
        lines.append("-" * 40)

        for name, criterion in judgment.criteria.items():
            readable_name = name.replace("_", " ").title()
            # Handle automatic fail (-1.0) for mental_state_reasoning
            if criterion.score == -1.0:
                status_icon = "X"
                score_display = "AUTO-FAIL"
            elif criterion.score >= self.min_criterion_threshold:
                status_icon = "+"
                score_display = f"{criterion.score:.2f}"
            else:
                status_icon = "!"
                score_display = f"{criterion.score:.2f}"
            lines.append(f"  [{status_icon}] {readable_name}: {score_display}")
            lines.append(f"      {criterion.reasoning}")

        lines.append("\nOverall Assessment:")
        lines.append(f"  {judgment.overall_reasoning}")

        if judgment.suggestions:
            lines.append("\nSuggestions for Improvement:")
            for i, suggestion in enumerate(judgment.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        lines.append("=" * 60)
        return "\n".join(lines)
