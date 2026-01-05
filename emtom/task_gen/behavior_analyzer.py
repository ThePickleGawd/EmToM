"""
Behavior analyzer for benchmark agent traces.

Analyzes whether benchmark agents exhibited Theory of Mind reasoning
during task execution, aligned with the judge_tom criteria.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


@dataclass
class BehaviorAnalysis:
    """Result of analyzing agent behavior during benchmark."""

    # Observations aligned with judge_tom criteria
    information_sharing: str  # How agents shared/used asymmetric information
    interdependence_observed: str  # Whether agents actually depended on each other
    mental_state_reasoning: str  # Whether agents reasoned about each other's knowledge/beliefs
    coordination_behavior: str  # How agents coordinated their actions

    # Overall observations
    communication_patterns: str  # What agents communicated and when
    unexpected_behaviors: str  # Anything surprising or noteworthy
    tom_utilized: bool  # Whether ToM reasoning appeared to be used
    summary: str  # Brief overall summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "information_sharing": self.information_sharing,
            "interdependence_observed": self.interdependence_observed,
            "mental_state_reasoning": self.mental_state_reasoning,
            "coordination_behavior": self.coordination_behavior,
            "communication_patterns": self.communication_patterns,
            "unexpected_behaviors": self.unexpected_behaviors,
            "tom_utilized": self.tom_utilized,
            "summary": self.summary,
        }


ANALYSIS_PROMPT = """Analyze the behavior of two AI agents attempting to complete a collaborative task.

## Task Description
{task_description}

## Agent Secrets (Private Information)
{agent_secrets}

## Agent 0 Trace
{agent_0_trace}

## Agent 1 Trace
{agent_1_trace}

## Subtask Progress
{subtask_progress}

---

Analyze the agent behavior with respect to Theory of Mind. Consider:

1. **Information Sharing**: Did agents share their private information? Was it necessary? Did the receiving agent use it?

2. **Interdependence**: Did agents actually need each other to complete the task, or could one have done it alone?

3. **Mental State Reasoning**: Did agents appear to reason about what the other agent knows, believes, or needs? Or did they act independently?

4. **Coordination**: How did agents coordinate? Did they wait for each other, communicate plans, or work in parallel?

5. **Communication Patterns**: What did agents communicate? Was it helpful, redundant, or missing critical information?

6. **Unexpected Behaviors**: Note any surprising strategies, failures to coordinate, or interesting patterns.

Respond with JSON:
{{
  "information_sharing": "<observation about how agents handled asymmetric information>",
  "interdependence_observed": "<observation about whether agents actually depended on each other>",
  "mental_state_reasoning": "<observation about whether agents reasoned about each other's mental states>",
  "coordination_behavior": "<observation about how agents coordinated>",
  "communication_patterns": "<observation about agent communication>",
  "unexpected_behaviors": "<any surprising or noteworthy behaviors, or 'None observed'>",
  "tom_utilized": <true if agents appeared to use ToM reasoning, false otherwise>,
  "summary": "<1-2 sentence overall summary of agent behavior>"
}}

Be observational and descriptive. It's fine if agents didn't use ToM reasoning - that's an interesting finding about LLM capabilities."""


class BehaviorAnalyzer:
    """Analyzes benchmark agent behavior for ToM-relevant patterns."""

    def __init__(self, llm_client: "BaseLLM", verbose: bool = False):
        self.llm = llm_client
        self.verbose = verbose

    def analyze(
        self,
        task_data: Dict[str, Any],
        agent_traces: Dict[str, str],
        subtask_progress: Dict[str, bool],
    ) -> BehaviorAnalysis:
        """
        Analyze agent behavior from benchmark traces.

        Args:
            task_data: The task JSON
            agent_traces: Dict mapping agent_id to their reasoning trace
            subtask_progress: Dict mapping subtask_id to completion status

        Returns:
            BehaviorAnalysis with observations
        """
        # Format task info
        task_description = f"{task_data.get('title', 'Untitled')}\n\n[Task]: {task_data.get('task', '')}"

        # Format agent secrets
        secrets = task_data.get("agent_secrets", {})
        secrets_text = "\n".join(
            f"{agent}: {', '.join(s_list)}"
            for agent, s_list in secrets.items()
        )

        # Format traces (truncate if too long)
        max_trace_len = 4000
        agent_0_trace = agent_traces.get("agent_0", "No trace available")
        agent_1_trace = agent_traces.get("agent_1", "No trace available")

        if len(agent_0_trace) > max_trace_len:
            agent_0_trace = agent_0_trace[:max_trace_len] + "\n... [truncated]"
        if len(agent_1_trace) > max_trace_len:
            agent_1_trace = agent_1_trace[:max_trace_len] + "\n... [truncated]"

        # Format subtask progress
        progress_text = "\n".join(
            f"  {subtask}: {'COMPLETE' if status else 'INCOMPLETE'}"
            for subtask, status in subtask_progress.items()
        ) or "No subtask data"

        # Build prompt
        prompt = ANALYSIS_PROMPT.format(
            task_description=task_description,
            agent_secrets=secrets_text or "No secrets defined",
            agent_0_trace=agent_0_trace,
            agent_1_trace=agent_1_trace,
            subtask_progress=progress_text,
        )

        # Get LLM analysis
        try:
            response = self.llm(prompt)

            if self.verbose:
                print(f"[BehaviorAnalyzer] Response: {response[:500]}...")

            # Parse JSON response
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return BehaviorAnalysis(
                information_sharing=data.get("information_sharing", "Unable to analyze"),
                interdependence_observed=data.get("interdependence_observed", "Unable to analyze"),
                mental_state_reasoning=data.get("mental_state_reasoning", "Unable to analyze"),
                coordination_behavior=data.get("coordination_behavior", "Unable to analyze"),
                communication_patterns=data.get("communication_patterns", "Unable to analyze"),
                unexpected_behaviors=data.get("unexpected_behaviors", "None observed"),
                tom_utilized=data.get("tom_utilized", False),
                summary=data.get("summary", "Analysis incomplete"),
            )

        except Exception as e:
            if self.verbose:
                print(f"[BehaviorAnalyzer] Error: {e}")

            return BehaviorAnalysis(
                information_sharing="Analysis failed",
                interdependence_observed="Analysis failed",
                mental_state_reasoning="Analysis failed",
                coordination_behavior="Analysis failed",
                communication_patterns="Analysis failed",
                unexpected_behaviors="Analysis failed",
                tom_utilized=False,
                summary=f"Behavior analysis failed: {e}",
            )
