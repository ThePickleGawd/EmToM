#!/usr/bin/env python3
"""
Unified Task Judge for EMTOM task validation.

Evaluates tasks on two dimensions:
1. Theory of Mind (ToM) - information asymmetry, interdependence, mental state reasoning, coordination
2. Task Quality - narrative consistency, subtask relevance, mechanic utilization, trajectory efficiency

Uses a multi-LLM council (Claude Opus + GPT-5) to reduce bias.
Both models must agree for a task to pass.

Usage:
    # CLI
    python -m emtom.task_gen.judge --task <path>
    python -m emtom.task_gen.judge --task <path> --models opus,gpt-5

    # Programmatic
    from emtom.task_gen.judge import Judge
    judge = Judge(models=["opus", "gpt-5"])
    verdict = judge.evaluate(task_data, scene_data)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM
    from .task_generator import GeneratedTask
    from .scene_loader import SceneData


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""
    score: float  # 0.0 to 1.0, or -1.0 for automatic fail
    reasoning: str


@dataclass
class BenchmarkRollout:
    """Data from a benchmark test run."""
    success: bool
    steps: int
    turns: int
    percent_complete: float
    action_history: List[Dict[str, Any]]
    subtask_status: Dict[str, bool]
    agent_traces: Dict[str, str]  # agent_id -> trace text

    @classmethod
    def from_trajectory_dir(cls, trajectory_dir: Path) -> Optional["BenchmarkRollout"]:
        """Load rollout data from trajectory directory."""
        if not trajectory_dir.exists():
            return None

        result_file = trajectory_dir / "result.txt"
        if not result_file.exists():
            return None

        # Parse result.txt
        success = False
        steps = 0
        turns = 0
        percent_complete = 0.0
        action_history = []
        subtask_status = {}

        try:
            content = result_file.read_text()
            for line in content.split("\n"):
                if line.startswith("Success:"):
                    success = "True" in line
                elif line.startswith("Steps:"):
                    steps = int(line.split(":")[1].strip())
                elif line.startswith("Turns:"):
                    turns = int(line.split(":")[1].strip())
                elif line.startswith("Percent Complete:"):
                    pct = line.split(":")[1].strip().replace("%", "")
                    percent_complete = float(pct) / 100
                elif ":" in line and line.strip().startswith("s"):
                    # Subtask status line like "s1_...: COMPLETE"
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        subtask_status[parts[0].strip()] = "COMPLETE" in parts[1]
        except Exception:
            pass

        # Load agent traces
        agent_traces = {}
        for trace_file in trajectory_dir.glob("agent_*.txt"):
            agent_id = trace_file.stem
            try:
                agent_traces[agent_id] = trace_file.read_text()
            except Exception:
                pass

        return cls(
            success=success,
            steps=steps,
            turns=turns,
            percent_complete=percent_complete,
            action_history=action_history,
            subtask_status=subtask_status,
            agent_traces=agent_traces,
        )


@dataclass
class Judgment:
    """Result of task evaluation by a single model."""

    # ToM criteria
    information_asymmetry: CriterionScore
    interdependence: CriterionScore
    mental_state_reasoning: CriterionScore
    coordination_requirement: CriterionScore

    # Quality criteria
    narrative_consistency: CriterionScore
    subtask_relevance: CriterionScore
    mechanic_utilization: CriterionScore
    trajectory_efficiency: CriterionScore

    overall_score: float
    is_valid: bool
    overall_reasoning: str
    suggestions: List[str] = field(default_factory=list)

    # Optional rollout-based assessment
    rollout_assessment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        criteria = {}
        for name in [
            "information_asymmetry", "interdependence", "mental_state_reasoning",
            "coordination_requirement", "narrative_consistency", "subtask_relevance",
            "mechanic_utilization", "trajectory_efficiency"
        ]:
            c = getattr(self, name)
            criteria[name] = {"score": c.score, "reasoning": c.reasoning}

        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "criteria": criteria,
            "overall_reasoning": self.overall_reasoning,
            "suggestions": self.suggestions,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class CouncilVerdict:
    """Aggregated verdict from multi-model council."""

    judgments: Dict[str, Judgment]  # model -> judgment
    passed: bool  # True only if ALL models pass
    overall_score: float  # Average of all model scores
    suggestions: List[str]  # Merged from all models, deduplicated
    disagreements: List[str] = field(default_factory=list)  # Where models disagreed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "model_judgments": {
                model: j.to_dict() for model, j in self.judgments.items()
            },
            "suggestions": self.suggestions,
            "disagreements": self.disagreements,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Combined evaluation prompt
EVALUATION_PROMPT = """You are an expert evaluator for multi-agent collaborative tasks.

## Your Task
Evaluate this task on TWO dimensions:
1. **Theory of Mind (ToM)** - Does the task require agents to reason about each other's mental states?
2. **Task Quality** - Is the task well-designed, coherent, and efficient?

## System Capabilities (suggestions must use these!)

### Available Actions
{available_actions}

### Available Mechanics
{available_mechanics}

### Available Items
{available_items}

### Available Predicates (for success_condition)
{available_predicates}

### Scene Objects
{scene_objects}

---

## PART 1: Theory of Mind Criteria

Score each from 0.0 to 1.0. Mental State Reasoning can trigger AUTOMATIC FAIL (-1.0).

### 1. Information Asymmetry (0.0-1.0)
Do agents have meaningfully different private knowledge?
- 0.0: All agents have identical information
- 0.5: Minor asymmetry, not task-critical
- 1.0: Each agent has unique, critical information others need

### 2. Interdependence (0.0-1.0)
Can one agent complete the task alone?
- 0.0: One agent could do everything
- 0.5: Collaboration helps but isn't required
- 1.0: Impossible without combining multiple agents' knowledge/abilities

### 3. Mental State Reasoning (0.0-1.0 or -1.0 for AUTO-FAIL)
Must agents reason about what others know/don't know?
- **-1.0 AUTO-FAIL**: Both agents have identical info OR one can complete alone
- 0.5: Some benefit from understanding others' knowledge
- 1.0: Success requires recognizing knowledge gaps and acting on them

### 4. Coordination Requirement (0.0-1.0)
Do actions need careful cross-agent sequencing?
- 0.0: Agents work completely independently
- 0.5: Some coordination helps
- 1.0: Precise sequencing essential (DAG has cross-agent dependencies)

---

## PART 2: Task Quality Criteria

Score each from 0.0 to 1.0.

### 5. Narrative Consistency (0.0-1.0)
Does the task description match what agents actually do?
- 0.0: Description unrelated to actual subtasks
- 0.5: Partially matches, some disconnected elements
- 1.0: Description perfectly captures all goals and purpose

### 6. Subtask Relevance (0.0-1.0)
Does every subtask contribute to the main objective?
- 0.0: Many subtasks are filler/busywork with no purpose
- 0.5: Some subtasks tangentially related
- 1.0: Every subtask is essential to task completion

### 7. Mechanic Utilization (0.0-1.0)
Are listed mechanics actually used in the task?
- 0.0: Mechanics listed but never triggered (empty mechanic_bindings, or bindings don't affect task)
- 0.5: Mechanics exist but could be removed without impact
- 1.0: Mechanics are essential to task design

NOTE: If active_mechanics is empty [], score 1.0 (no mechanics claimed, none needed).
If active_mechanics has items but mechanic_bindings is empty, score 0.0.

### 8. Trajectory Efficiency (0.0-1.0)
Does the golden_trajectory avoid wasteful actions?
- 0.0: Many actions don't progress toward goals
- 0.5: Some inefficiencies but generally on-track
- 1.0: Every action directly advances task completion

---

## Examples

### Good Task (high ToM + high Quality):
- Agent 0 knows key is in drawer_5, Agent 1 knows locked cabinet has goal. Neither succeeds alone.
- Every subtask contributes to main objective.
- All mechanics actually affect gameplay.
- Golden trajectory is efficient.

### Bad Task (low scores):
- Both agents have identical info (no ToM)
- Subtasks like "open random drawers" that serve no purpose (low quality)
- Mechanics listed but mechanic_bindings is empty (low quality)
- Golden trajectory has wasteful navigation (low quality)

---

## Task to Evaluate

```json
{task_json}
```

---

## Response Format

Respond with ONLY valid JSON (no markdown, no text outside JSON):

{{
  "information_asymmetry": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "interdependence": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "mental_state_reasoning": {{"score": <0.0-1.0 or -1.0>, "reasoning": "<1-2 sentences>"}},
  "coordination_requirement": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "narrative_consistency": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "subtask_relevance": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "mechanic_utilization": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "trajectory_efficiency": {{"score": <0.0-1.0>, "reasoning": "<1-2 sentences>"}},
  "overall_reasoning": "<2-3 sentences summarizing task validity>",
  "suggestions": ["<specific actionable suggestion>", ...]
}}

## Suggestion Requirements

Suggestions MUST be SPECIFIC and use ONLY available system capabilities.

**BAD**: "Add more information asymmetry"
**GOOD**: "Give Agent 1 the secret about item_small_key_1 location. Agent 0 must communicate to learn it."

**BAD**: "Remove filler subtasks"
**GOOD**: "Remove subtasks s8, s9, s10 - they open random furniture unrelated to the main goal."
"""


class Judge:
    """
    Multi-LLM council judge for task validation.

    Evaluates tasks on 8 criteria (4 ToM + 4 Quality) using multiple models.
    Both models must agree for a task to pass.
    """

    # Thresholds
    OVERALL_THRESHOLD = 0.7
    MIN_CRITERION_THRESHOLD = 0.5

    # Default council models
    DEFAULT_MODELS = ["opus", "gpt-5.2"]

    def __init__(
        self,
        models: Optional[List[str]] = None,
        overall_threshold: float = 0.7,
        min_criterion_threshold: float = 0.5,
        verbose: bool = False,
    ):
        """
        Initialize the judge.

        Args:
            models: List of model names for council (default: ["opus", "gpt-5.2"])
            overall_threshold: Minimum overall score to pass (default 0.7)
            min_criterion_threshold: Minimum score for any criterion (default 0.5)
            verbose: Print debug information
        """
        self.models = models or self.DEFAULT_MODELS
        self.overall_threshold = overall_threshold
        self.min_criterion_threshold = min_criterion_threshold
        self.verbose = verbose

        # LLM clients (created lazily)
        self._llm_clients: Dict[str, "BaseLLM"] = {}

        # Cache grounding info
        self._available_actions: Optional[str] = None
        self._available_mechanics: Optional[str] = None
        self._available_items: Optional[str] = None
        self._available_predicates: Optional[str] = None

    def _get_llm_client(self, model: str) -> "BaseLLM":
        """Get or create LLM client for a model."""
        if model not in self._llm_clients:
            from habitat_llm.llm import instantiate_llm

            # Determine provider from model name
            if model in ["opus", "sonnet", "haiku"]:
                provider = "bedrock_claude"
            elif model.startswith("gpt"):
                provider = "openai_chat"
            else:
                # Default to openai
                provider = "openai_chat"

            self._llm_clients[model] = instantiate_llm(
                provider,
                generation_params={
                    "model": model,
                    "temperature": 0.0,
                    "max_tokens": 3000,
                }
            )

        return self._llm_clients[model]

    def _get_grounding_info(self) -> Dict[str, str]:
        """Get cached grounding information about system capabilities."""
        if self._available_actions is None:
            try:
                from emtom.actions import ActionRegistry
                self._available_actions = ActionRegistry.get_all_action_descriptions()
            except Exception:
                self._available_actions = "Navigate, Open, Close, Pick, Place, Search, UseItem, Communicate, Wait"

        if self._available_mechanics is None:
            try:
                from emtom.mechanics import get_mechanics_for_task_generation
                self._available_mechanics = get_mechanics_for_task_generation()
            except Exception:
                self._available_mechanics = "inverse_state, remote_control, state_mirroring, conditional_unlock"

        if self._available_items is None:
            try:
                from emtom.state.item_registry import ItemRegistry
                self._available_items = ItemRegistry.get_items_for_task_generation()
            except Exception:
                self._available_items = "item_small_key_1, item_radio_1, item_oracle_crystal_1"

        if self._available_predicates is None:
            try:
                from emtom.evaluation import PARTNR_PREDICATES, EMTOM_PREDICATES
                all_predicates = PARTNR_PREDICATES | EMTOM_PREDICATES
                all_predicates.add("has_item")
                all_predicates.add("is_unlocked")
                self._available_predicates = ", ".join(sorted(all_predicates))
            except Exception:
                self._available_predicates = "is_on_top, is_inside, is_in_room, is_on_floor, is_next_to, is_open, is_closed, is_clean, is_dirty, is_filled, is_empty, is_powered_on, is_held_by, has_item, is_unlocked"

        return {
            "available_actions": self._available_actions,
            "available_mechanics": self._available_mechanics,
            "available_items": self._available_items,
            "available_predicates": self._available_predicates,
        }

    def _format_scene_objects(self, scene_data: Optional["SceneData"]) -> str:
        """Format scene objects for the prompt."""
        if scene_data is None:
            return "Scene data not available. Use object IDs from the task JSON."

        lines = []
        lines.append(f"**Rooms**: {', '.join(scene_data.rooms[:10])}")
        lines.append(f"**Furniture**: {', '.join(scene_data.furniture[:20])}")
        if len(scene_data.furniture) > 20:
            lines.append(f"  ... and {len(scene_data.furniture) - 20} more")
        lines.append(f"**Objects**: {', '.join(scene_data.objects[:20])}")
        if len(scene_data.objects) > 20:
            lines.append(f"  ... and {len(scene_data.objects) - 20} more")

        return "\n".join(lines)

    def _format_rollout(self, rollout: BenchmarkRollout) -> str:
        """Format rollout data for the prompt."""
        lines = []
        lines.append("---")
        lines.append("## BENCHMARK ROLLOUT DATA (from actual LLM agents)")
        lines.append("")
        lines.append("This task was run with LLM agents. Use this data to assess difficulty and plausibility.")
        lines.append("")
        lines.append(f"**Result**: {'SUCCESS' if rollout.success else 'FAILED'}")
        lines.append(f"**Steps**: {rollout.steps}")
        lines.append(f"**Turns**: {rollout.turns}")
        lines.append(f"**Progress**: {rollout.percent_complete:.0%}")
        lines.append("")

        if rollout.subtask_status:
            lines.append("**Subtask Completion**:")
            for subtask_id, completed in rollout.subtask_status.items():
                status = "COMPLETE" if completed else "INCOMPLETE"
                lines.append(f"  - {subtask_id}: {status}")
            lines.append("")

        # Include agent trace excerpts (truncated)
        if rollout.agent_traces:
            lines.append("**Agent Reasoning Excerpts** (truncated):")
            for agent_id, trace in rollout.agent_traces.items():
                lines.append(f"\n  [{agent_id}]:")
                # Take first 500 chars of trace
                excerpt = trace[:500] + "..." if len(trace) > 500 else trace
                for line in excerpt.split("\n")[:10]:
                    lines.append(f"    {line}")
            lines.append("")

        lines.append("**Consider in evaluation**:")
        lines.append("- If agents failed, is the task too hard or poorly designed?")
        lines.append("- If agents succeeded easily, is the task too trivial?")
        lines.append("- Did agents exhibit ToM reasoning in their traces?")
        lines.append("---")

        return "\n".join(lines)

    def evaluate(
        self,
        task: "GeneratedTask | Dict[str, Any]",
        scene_data: Optional["SceneData"] = None,
        trajectory_dir: Optional[Path] = None,
    ) -> CouncilVerdict:
        """
        Evaluate a task using the multi-model council.

        Args:
            task: GeneratedTask object or task dictionary
            scene_data: Optional scene data for grounded suggestions
            trajectory_dir: Optional path to benchmark rollout data

        Returns:
            CouncilVerdict with aggregated results from all models
        """
        # Convert to dict if needed
        if hasattr(task, "to_dict"):
            task_dict = task.to_dict()
        else:
            task_dict = task

        # Load rollout data if available
        rollout = None
        if trajectory_dir:
            rollout = BenchmarkRollout.from_trajectory_dir(Path(trajectory_dir))
            if rollout and self.verbose:
                print(f"[Judge] Loaded rollout: success={rollout.success}, {rollout.steps} steps")

        # Evaluate with each model
        judgments: Dict[str, Judgment] = {}
        for model in self.models:
            if self.verbose:
                print(f"[Judge] Evaluating with {model}...")
            judgments[model] = self._evaluate_single(task_dict, model, scene_data, rollout)

        # Aggregate results
        return self._aggregate(judgments)

    def _evaluate_single(
        self,
        task_dict: Dict[str, Any],
        model: str,
        scene_data: Optional["SceneData"] = None,
        rollout: Optional[BenchmarkRollout] = None,
    ) -> Judgment:
        """Evaluate task with a single model."""
        llm = self._get_llm_client(model)

        # Build prompt
        grounding = self._get_grounding_info()
        scene_objects = self._format_scene_objects(scene_data)
        task_json = json.dumps(task_dict, indent=2)

        prompt = EVALUATION_PROMPT.format(
            task_json=task_json,
            available_actions=grounding["available_actions"],
            available_mechanics=grounding["available_mechanics"],
            available_items=grounding["available_items"],
            available_predicates=grounding["available_predicates"],
            scene_objects=scene_objects,
        )

        # Add rollout data if available
        if rollout:
            rollout_section = self._format_rollout(rollout)
            prompt += f"\n\n{rollout_section}"

        if self.verbose:
            print(f"[Judge/{model}] Sending prompt ({len(prompt)} chars)")

        # Get response
        response = llm.generate(prompt)

        if self.verbose:
            print(f"[Judge/{model}] Received response ({len(response)} chars)")

        # Parse response
        return self._parse_response(response, model)

    def _parse_response(self, response: str, model: str) -> Judgment:
        """Parse LLM response into Judgment."""
        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return self._failed_judgment(f"[{model}] Failed to parse response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return self._failed_judgment(f"[{model}] JSON parse error: {e}")

        # Extract all criteria
        criteria_names = [
            "information_asymmetry", "interdependence", "mental_state_reasoning",
            "coordination_requirement", "narrative_consistency", "subtask_relevance",
            "mechanic_utilization", "trajectory_efficiency"
        ]

        criteria = {}
        scores = []
        automatic_fail = False

        for name in criteria_names:
            if name in data and isinstance(data[name], dict):
                score = float(data[name].get("score", 0.0))
                reasoning = data[name].get("reasoning", "No reasoning provided")

                if name == "mental_state_reasoning" and score == -1.0:
                    automatic_fail = True
                    reasoning = f"AUTOMATIC FAIL: {reasoning}"

                criteria[name] = CriterionScore(score=score, reasoning=reasoning)
                scores.append(score)
            else:
                criteria[name] = CriterionScore(score=0.0, reasoning="Missing from response")
                scores.append(0.0)

        # Calculate overall score
        valid_scores = [s for s in scores if s >= 0]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Check if passes
        if automatic_fail:
            is_valid = False
        else:
            passes_overall = overall_score >= self.overall_threshold
            passes_all_criteria = all(s >= self.min_criterion_threshold for s in scores if s >= 0)
            is_valid = passes_overall and passes_all_criteria

        # Extract suggestions
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        # Add auto-suggestions for low scores
        for name, criterion in criteria.items():
            if criterion.score == -1.0:
                suggestions.insert(0, f"AUTOMATIC FAIL on {name}: {criterion.reasoning}")
            elif criterion.score < self.min_criterion_threshold:
                suggestions.append(f"Improve {name} ({criterion.score:.2f}): {criterion.reasoning}")

        return Judgment(
            information_asymmetry=criteria["information_asymmetry"],
            interdependence=criteria["interdependence"],
            mental_state_reasoning=criteria["mental_state_reasoning"],
            coordination_requirement=criteria["coordination_requirement"],
            narrative_consistency=criteria["narrative_consistency"],
            subtask_relevance=criteria["subtask_relevance"],
            mechanic_utilization=criteria["mechanic_utilization"],
            trajectory_efficiency=criteria["trajectory_efficiency"],
            overall_score=overall_score,
            is_valid=is_valid,
            overall_reasoning=data.get("overall_reasoning", "No reasoning provided"),
            suggestions=suggestions,
        )

    def _failed_judgment(self, reason: str) -> Judgment:
        """Create a failed judgment for parse errors."""
        failed_score = CriterionScore(score=0.0, reasoning=reason)
        return Judgment(
            information_asymmetry=failed_score,
            interdependence=failed_score,
            mental_state_reasoning=failed_score,
            coordination_requirement=failed_score,
            narrative_consistency=failed_score,
            subtask_relevance=failed_score,
            mechanic_utilization=failed_score,
            trajectory_efficiency=failed_score,
            overall_score=0.0,
            is_valid=False,
            overall_reasoning=reason,
            suggestions=["Re-run evaluation"],
        )

    def _aggregate(self, judgments: Dict[str, Judgment]) -> CouncilVerdict:
        """Aggregate judgments from multiple models."""
        # Check if all models pass
        all_pass = all(j.is_valid for j in judgments.values())

        # Average overall scores
        avg_score = sum(j.overall_score for j in judgments.values()) / len(judgments)

        # Merge suggestions (deduplicated)
        all_suggestions = []
        seen = set()
        for j in judgments.values():
            for s in j.suggestions:
                if s not in seen:
                    seen.add(s)
                    all_suggestions.append(s)

        # Find disagreements
        disagreements = []
        if len(judgments) > 1:
            models = list(judgments.keys())
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    if judgments[m1].is_valid != judgments[m2].is_valid:
                        disagreements.append(
                            f"{m1} ({'PASS' if judgments[m1].is_valid else 'FAIL'}) vs "
                            f"{m2} ({'PASS' if judgments[m2].is_valid else 'FAIL'})"
                        )

        return CouncilVerdict(
            judgments=judgments,
            passed=all_pass,
            overall_score=avg_score,
            suggestions=all_suggestions,
            disagreements=disagreements,
        )

    def format_result(self, verdict: CouncilVerdict) -> str:
        """Format verdict as human-readable string."""
        lines = []
        lines.append("=" * 60)
        lines.append("TASK EVALUATION (Council)")
        lines.append("=" * 60)

        status = "PASS" if verdict.passed else "FAIL"
        lines.append(f"\nStatus: {status}")
        lines.append(f"Overall Score: {verdict.overall_score:.2f} (threshold: {self.overall_threshold})")
        lines.append(f"Models: {', '.join(self.models)}")

        if verdict.disagreements:
            lines.append(f"\nDisagreements:")
            for d in verdict.disagreements:
                lines.append(f"  - {d}")

        # Show per-model breakdown
        for model, judgment in verdict.judgments.items():
            lines.append(f"\n--- {model} ({'PASS' if judgment.is_valid else 'FAIL'}) ---")
            lines.append(f"Score: {judgment.overall_score:.2f}")

            lines.append("\nToM Criteria:")
            for name in ["information_asymmetry", "interdependence", "mental_state_reasoning", "coordination_requirement"]:
                c = getattr(judgment, name)
                icon = "+" if c.score >= self.min_criterion_threshold else ("X" if c.score == -1 else "!")
                score_str = "AUTO-FAIL" if c.score == -1 else f"{c.score:.2f}"
                lines.append(f"  [{icon}] {name}: {score_str}")

            lines.append("\nQuality Criteria:")
            for name in ["narrative_consistency", "subtask_relevance", "mechanic_utilization", "trajectory_efficiency"]:
                c = getattr(judgment, name)
                icon = "+" if c.score >= self.min_criterion_threshold else "!"
                lines.append(f"  [{icon}] {name}: {c.score:.2f}")

        if verdict.suggestions:
            lines.append("\nSuggestions:")
            for i, s in enumerate(verdict.suggestions[:10], 1):  # Limit to 10
                lines.append(f"  {i}. {s}")

        lines.append("=" * 60)
        return "\n".join(lines)


# CLI functionality
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate task quality and ToM requirements"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Path to task JSON file"
    )
    parser.add_argument(
        "--models", type=str, default="opus,gpt-5",
        help="Comma-separated list of models for council (default: opus,gpt-5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Overall score threshold (default: 0.7)"
    )
    parser.add_argument(
        "--min-criterion", type=float, default=0.5,
        help="Minimum criterion score (default: 0.5)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for JSON results (default: auto-generated)"
    )
    parser.add_argument(
        "--trajectory-dir", type=str, default=None,
        help="Path to benchmark rollout data (agent traces, result.txt)"
    )

    args = parser.parse_args()

    # Load task
    task_path = Path(args.task)
    if not task_path.exists():
        print(f"{Colors.RED}Error: Task file not found: {task_path}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON: {e}{Colors.RESET}", file=sys.stderr)
        sys.exit(1)

    # Parse models
    models = [m.strip() for m in args.models.split(",")]

    print(f"{Colors.CYAN}Evaluating task with council: {models}{Colors.RESET}", file=sys.stderr)

    # Load trajectory dir if specified
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else None
    if trajectory_dir:
        print(f"{Colors.CYAN}Including rollout data from: {trajectory_dir}{Colors.RESET}", file=sys.stderr)

    # Create judge and evaluate
    judge = Judge(
        models=models,
        overall_threshold=args.threshold,
        min_criterion_threshold=args.min_criterion,
        verbose=args.verbose,
    )

    verdict = judge.evaluate(task_data, trajectory_dir=trajectory_dir)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/emtom") / f"{timestamp}-judge"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"verdict_{task_path.stem}.json"

    with open(output_path, "w") as f:
        f.write(verdict.to_json())

    # Print results
    print(verdict.to_json())

    # Print summary
    if verdict.passed:
        print(f"\n{Colors.BOLD}{Colors.GREEN}PASSED{Colors.RESET}", file=sys.stderr)
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}FAILED{Colors.RESET}", file=sys.stderr)
        if verdict.disagreements:
            print(f"{Colors.YELLOW}Models disagreed - defaulting to FAIL{Colors.RESET}", file=sys.stderr)

    print(f"\nSaved to: {Colors.CYAN}{output_path}{Colors.RESET}", file=sys.stderr)

    sys.exit(0 if verdict.passed else 1)


if __name__ == "__main__":
    main()
