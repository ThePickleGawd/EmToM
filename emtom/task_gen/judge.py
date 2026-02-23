#!/usr/bin/env python3
"""
Category-aware Task Judge for EMTOM task validation.

Evaluates tasks using category-specific criteria:
- Cooperative: agent necessity, secrets, interdependence
- Competitive: agent necessity, secrets, goal opposition
- Mixed: agent necessity, secrets, subgoal tension

Priority criteria (evaluated first, suggestions prioritized):
- agent_necessity: Every agent must be indispensable
- secret_relevance: Secrets must be required for task completion

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
    from .diversity import DiversityTracker


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


# =============================================================================
# Category Configuration
# =============================================================================

# Shared quality criteria (apply to ALL categories)
# Ordered by importance - agent design criteria first
SHARED_CRITERIA = [
    "agent_necessity",       # Every agent must be essential
    "secret_quality",        # Secrets are actionable, natural, and non-leaking
    "task_naturalness",      # Task description uses natural language, not object IDs
    "narrative_consistency",
    "goal_relevance",        # Renamed from subtask_relevance for PDDL
    "mechanic_utilization",
    "pddl_solvability",     # PDDL goal solvability and ToM depth
]

# Category-specific criteria
CATEGORY_CRITERIA = {
    "cooperative": SHARED_CRITERIA + ["task_interdependence"],
    "competitive": SHARED_CRITERIA + ["goal_opposition"],
    "mixed": SHARED_CRITERIA + ["subgoal_tension"],
}

# Criteria descriptions for prompts
CRITERIA_DESCRIPTIONS = {
    # Core agent design criteria (most important)
    "agent_necessity": {
        "name": "Agent Necessity",
        "description": "Is EVERY agent essential? Could the task work with fewer agents?",
        "rubric": """0.0: One or more agents are completely idle or removable
0.3: Some agents do trivial work that others could handle
0.5: All agents participate but some are only marginally needed
0.7: Most agents essential, one could theoretically be merged
1.0: Every agent is indispensable - removing any would make task impossible""",
    },
    "secret_quality": {
        "name": "Secret Quality",
        "description": "Are secrets actionable, natural-language, and non-leaking? They must be required to solve the task.",
        "rubric": """0.0: Secrets leak targets OR use IDs OR are useless/vague
0.3: Secrets are too vague, too prescriptive, or partially leak key info
0.5: Mostly OK but still broad, redundant, or weakly necessary
0.7: Actionable, natural, and mostly non-leaking with minor redundancy
1.0: Each secret is essential, actionable, natural-language, and does not leak public targets""",
    },
    "task_naturalness": {
        "name": "Task Description Natural Language",
        "description": "Does the `task` field and `agent_secrets` use natural language instead of object IDs? Agents use FindObjectTool to resolve descriptions. Check ONLY `task` and `agent_secrets` text — NOT `pddl_goal`, `golden_trajectory`, or `pddl_ordering` (these are machine-readable fields that use IDs by design).",
        "rubric": """0.0: `task` or `agent_secrets` contain many object IDs (toy_airplane_0, microwave_29) or a 'Grounding note' with IDs
0.3: `task` or `agent_secrets` have some object IDs mixed with natural descriptions
0.5: Mostly natural language in task/secrets but a few IDs slip through
0.7: Natural language throughout task/secrets, minor specificity issues
1.0: Pure natural language in `task` and `agent_secrets` — agents discover IDs via FindObjectTool. (IDs in pddl_goal/trajectory are expected and should not lower this score.)""",
    },
    # Task quality criteria
    "narrative_consistency": {
        "name": "Narrative Consistency",
        "description": "Does the task description accurately describe what agents must do?",
        "rubric": """0.0: Description misleading or unrelated to actual subtasks
0.3: Major discrepancies (mentions goals not in subtasks, or misses key objectives)
0.5: Partial match (captures main idea but omits or misrepresents details)
0.7: Good match with minor omissions
1.0: Perfect - description precisely matches subtask objectives""",
    },
    "goal_relevance": {
        "name": "Goal Relevance",
        "description": "Does every PDDL goal conjunct directly contribute to the main objective?",
        "rubric": """0.0: Many filler goals (opening random furniture, navigating pointlessly)
0.3: Several goal conjuncts unrelated to main objective
0.5: Some tangential goals that could be removed
0.7: Most goals essential, 1-2 questionable
1.0: All goal conjuncts directly advance the objective - removing any would break the task""",
    },
    "pddl_solvability": {
        "name": "PDDL Solvability & Epistemic Coherence",
        "description": "Is the task structurally solvable? Are K() goals backed by CONCRETE information asymmetry? Check: does a mechanic (room_restriction, restricted_communication, unreliable_communication) actually prevent the agent from directly observing the fact? If the agent could just walk to the room and see for themselves, the K() goal is trivially satisfied and should score LOW.",
        "rubric": """0.0: Goal references nonexistent objects or uses invalid predicates
0.3: Goal is technically valid but trivially satisfied or impossible; K() goals with NO backing mechanic (agent can directly observe the fact — no room_restriction, no restricted_communication blocking them)
0.5: Goal is valid but ordering is empty with multiple conjuncts; K() goals where the agent has indirect access (could reach the location with extra steps)
0.7: Goal is well-formed, K() goals each backed by a specific mechanic that prevents direct observation, minor ordering issues
1.0: Goal is well-formed, all predicates valid, every K() goal has a concrete blocking mechanic (room_restriction, restricted_communication, etc.), ordering defines meaningful dependencies, structurally solvable""",
    },
    "mechanic_utilization": {
        "name": "Mechanic Utilization & Balance",
        "description": "Are listed mechanics essential AND is the count appropriate? (Empty list = auto-pass at 1.0). Too many mechanics (4+) overwhelm agents — penalize overstacking.",
        "rubric": """0.0: Mechanics listed but never triggered or used; OR 4+ mechanics creating unmanageable complexity
0.3: Mechanics present but could be removed; OR 3+ mechanics where some are redundant or don't serve distinct purposes
0.5: Mechanics add flavor but aren't essential; OR mechanics are essential but there are too many interacting constraints
0.7: 1-3 well-integrated mechanics that each serve a distinct purpose
1.0: 1-2 tightly integrated mechanics that are essential — task wouldn't work without them""",
    },
    # Cooperative-specific
    "task_interdependence": {
        "name": "Task Interdependence",
        "description": "Do agents genuinely NEED each other to succeed?",
        "rubric": """0.0: One agent can complete the entire task alone
0.3: Agents help but aren't required (parallel independent work)
0.5: Some interdependence but key steps can be done solo
0.7: Strong interdependence with minor exceptions
1.0: Impossible for any single agent to succeed - must coordinate and share information""",
    },
    # Competitive-specific
    "goal_opposition": {
        "name": "Goal Opposition",
        "description": "Do teams have truly mutually exclusive win conditions?",
        "rubric": """0.0: Both teams can win simultaneously (no real competition)
0.3: Goals partially conflict but both could achieve objectives
0.5: Conflict exists but not zero-sum
0.7: Strong opposition - one winning significantly hurts the other
1.0: Zero-sum - exactly one team wins, the other loses""",
    },
    "team_balance": {
        "name": "Team Balance",
        "description": "Do both teams have a fair chance of winning?",
        "rubric": """0.0: One team has overwhelming advantage (closer start, easier objective)
0.3: Significant imbalance favoring one team
0.5: Slight advantage but both could win
0.7: Well balanced with minor asymmetries
1.0: Fair contest - either team could win with good play""",
    },
    # Mixed-specific
    "subgoal_tension": {
        "name": "Subgoal Tension",
        "description": "Do hidden subgoals create meaningful conflict with main goal or each other?",
        "rubric": """0.0: Subgoals trivial or don't conflict with anything
0.3: Subgoals exist but easily achieved without tension
0.5: Minor tension that doesn't create real dilemmas
0.7: Meaningful conflicts that require strategic choices
1.0: Real dilemmas - pursuing subgoals risks main goal or conflicts with others' subgoals""",
    },
    # User requirements (added dynamically when query is provided)
    "user_requirements_alignment": {
        "name": "User Requirements Alignment",
        "description": "Does the task align with the user's specific request/query?",
        "rubric": """0.0: Task completely ignores user's request (wrong items, wrong mechanics, wrong theme)
0.3: Task vaguely relates but misses the key elements requested
0.5: Task partially addresses the request but missing important aspects
0.7: Task mostly aligns with minor omissions
1.0: Task fully incorporates what the user requested""",
    },
    # Task novelty (added dynamically when diversity tracker is provided)
    "task_novelty": {
        "name": "Task Novelty",
        "description": "Is this task structurally different from existing tasks in the dataset?",
        "rubric": """0.0: Nearly identical to an existing task (same structure, just different items/names)
0.3: Very similar to existing task(s), feels like a reskin
0.5: Shares significant structural elements with existing tasks
0.7: Mostly novel with some minor similarities to existing patterns
1.0: Completely novel structure, nothing similar exists in the dataset""",
    },
}


@dataclass
class Judgment:
    """Result of task evaluation by a single model."""

    category: str  # Task category that was evaluated
    criteria_scores: Dict[str, CriterionScore]  # Dynamic based on category
    overall_score: float
    is_valid: bool
    overall_reasoning: str
    suggestions: List[str] = field(default_factory=list)

    # Optional rollout-based assessment
    rollout_assessment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        criteria = {}
        for name, score in self.criteria_scores.items():
            criteria[name] = {"score": score.score, "reasoning": score.reasoning}

        return {
            "category": self.category,
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


# Category-aware evaluation prompt template
EVALUATION_PROMPT = """You are an expert evaluator for multi-agent tasks.

## Task Category: {category}
{category_description}
{difficulty_section}
{user_requirements_section}

## System Capabilities (use these in suggestions)
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

## Checks
- `task` is GLOBAL; for competitive/mixed it must not leak secret targets or team-specific objectives
- Secrets must be actionable (room/furniture/key/constraint) and required
- Secrets must be natural language (no IDs) and not step-by-step
- `required` semantics: true(shared), false(optional), "team_X"(win), "agent_X"(subgoal)
- Category must match `required` usage
- **Mechanic consistency**: Every mechanic referenced in `task` or `agent_secrets` (e.g., "the handle is reversed", "you have limited messages") MUST have a corresponding entry in `mechanic_bindings`. If secrets describe constraints that aren't in bindings, the simulator won't enforce them.
- **K() goal backing**: Every `K()` goal in `pddl_goal` must be backed by a mechanic that prevents the agent from directly observing the fact (e.g., `room_restriction` blocks navigation, `restricted_communication` blocks direct messaging). If the agent could just walk there and see, the K() goal is fake.

## Evaluation Criteria
Score each criterion from 0.0 to 1.0.
{criteria_section}

## Task to Evaluate
```json
{task_json}
```

## Response Format
Respond with ONLY valid JSON. Keep reasoning brief (under 15 words each).
{{
{response_format}
  "overall_reasoning": "<1 sentence>",
  "suggestions": ["<high-impact fix>", "..."]
}}

## Suggestion Requirements
Be specific and only use available system capabilities. Prioritize the most important fixes.
"""

# Category descriptions for the prompt
CATEGORY_PROMPT_DESCRIPTIONS = {
    "cooperative": """**COOPERATIVE** - All agents united toward shared goals
- Every agent contributes unique knowledge, skills, or access that others lack
- Information is distributed: one agent might know key locations, another knows which locks need which keys
- Success requires piecing together distributed information through communication
- Complex tasks can have parallel workstreams that converge
- Uses `agent_secrets` to distribute knowledge, `required: true` for shared goals""",
    "competitive": """**COMPETITIVE** - Teams with opposing objectives
- Divide agents into teams (any split: 1v1, 2v1, 2v2, 3v2, etc.)
- Teams compete for contested resources OR race to complete opposing objectives
- Each team member should contribute - divide responsibilities within teams
- Balance matters: if teams are uneven in size, give smaller team easier objectives
- Uses `teams` mapping, `required: "team_X"` for each team's win conditions
- Public `task` must be symmetric; team-specific targets belong in secrets""",
    "mixed": """**MIXED** - Cooperation with hidden conflicts
- All agents share a main goal they must complete together (`required: true`)
- Each agent also has a SECRET personal subgoal (`required: "agent_X"`) that may conflict with others
- Tension: agents must cooperate on the main task while secretly pursuing conflicting interests
- Subgoals should create interesting dilemmas, not make main goal impossible
- Public `task` must not reveal secret subgoals or targets""",
}


DIFFICULTY_DESCRIPTIONS = {
    "easy": """## Intended Difficulty: EASY
This task is designed for WEAKER models. Calibrate your evaluation accordingly:
- **Agent necessity**: 2-3 agents with clear, distinct roles is sufficient. Simple role division (e.g., one agent fetches, another places) counts as high agent necessity.
- **Task interdependence / goal opposition / subgoal tension**: Simple dependencies are fine. One clear handoff or information exchange between agents is enough.
- **Secret quality**: Secrets should be straightforward and actionable. Crucially, secrets MUST include hints about active mechanics (e.g., "the cabinet handle is reversed — opening closes it" for inverse_state, "operating the office cabinet seems to affect the kitchen fridge" for remote_control). Without mechanic hints, agents cannot discover mechanics through trial-and-error. Score HIGH when mechanic hints are present.
- **Mechanic utilization**: Using 0-1 mechanics is sufficient. Prefer simple, observable mechanics. Avoid stacking multiple mechanics.
- **Overall**: A well-structured simple task with clear agent roles, mechanic hints in secrets, and basic ToM should score HIGH. Do NOT penalize simplicity.""",
    "medium": """## Intended Difficulty: MEDIUM
This task targets mid-tier models. Standard evaluation applies:
- Agents should have meaningful distinct roles with some interdependence.
- Secrets should require reasoning to use effectively.
- Tasks should use 2-3 mechanics appropriately.
- Moderate complexity in coordination is expected.""",
    "hard": """## Intended Difficulty: HARD
This task is designed for the STRONGEST models. Expect high complexity:
- **Agent necessity**: Each agent should be truly indispensable with unique capabilities or knowledge.
- **Task interdependence / goal opposition / subgoal tension**: Complex multi-step dependencies, cascading information needs, or deep strategic considerations.
- **Secret quality**: Secrets should create genuine reasoning challenges — layered information, indirect clues, or strategic deception opportunities.
- **Mechanic utilization**: 2-3 well-integrated mechanics maximum. Complexity should come from deeper interactions between fewer mechanics, NOT from stacking 4+ mechanics. Penalize overloaded tasks.
- **Overall**: Reward tasks that would genuinely challenge top-tier AI models through depth, not breadth of mechanics.""",
}


def _get_criteria_for_category(category: str, user_query: Optional[str] = None) -> List[str]:
    """Get the list of criteria for a category, optionally including user_requirements_alignment."""
    criteria = list(CATEGORY_CRITERIA.get(category, SHARED_CRITERIA))
    if user_query:
        criteria.append("user_requirements_alignment")
    return criteria


def _build_criteria_section(category: str, user_query: Optional[str] = None) -> str:
    """Build the criteria section for a given category."""
    criteria = _get_criteria_for_category(category, user_query)
    lines = []
    for i, criterion in enumerate(criteria, 1):
        info = CRITERIA_DESCRIPTIONS.get(criterion, {})
        lines.append(f"### {i}. {info.get('name', criterion)} (0.0-1.0)")
        lines.append(info.get('description', ''))
        lines.append(f"- {info.get('rubric', '')}")
        lines.append("")
    return "\n".join(lines)


def _build_response_format(category: str, user_query: Optional[str] = None) -> str:
    """Build the JSON response format for a given category."""
    criteria = _get_criteria_for_category(category, user_query)
    lines = []
    for criterion in criteria:
        lines.append(f'  "{criterion}": {{"score": <0.0-1.0>, "reasoning": "<brief>"}},')
    return "\n".join(lines)


class Judge:
    """
    Multi-LLM council judge for task validation.

    Evaluates tasks using category-specific criteria:
    - Cooperative: 6 criteria (5 shared + task_interdependence)
    - Competitive: 6 criteria (5 shared + goal_opposition)
    - Mixed: 6 criteria (5 shared + subgoal_tension)

    Priority criteria (suggestions appear first):
    - agent_necessity: Every agent must be essential
    - secret_quality: Secrets must be actionable, natural, and non-leaking

    Both models must agree for a task to pass.
    """

    # Priority criteria - suggestions for these appear first
    PRIORITY_CRITERIA = ["agent_necessity", "secret_quality"]

    # Default council models
    DEFAULT_MODELS = ["opus", "gpt-5.2"]

    def __init__(
        self,
        models: Optional[List[str]] = None,
        overall_threshold: float = 0.65,
        min_criterion_threshold: float = 0.5,
        verbose: bool = False,
        user_query: Optional[str] = None,
        diversity_tracker: Optional["DiversityTracker"] = None,
        difficulty: Optional[str] = None,
    ):
        """
        Initialize the judge.

        Args:
            models: List of model names for council (default: ["opus", "gpt-5.2"])
            overall_threshold: Minimum overall score to pass (default 0.65)
            min_criterion_threshold: Minimum score for any criterion (default 0.5)
            verbose: Print debug information
            user_query: Optional user query that the task should align with
            diversity_tracker: Optional tracker to check task novelty against existing tasks
            difficulty: Intended difficulty level ("easy", "medium", "hard") for calibrated evaluation
        """
        self.models = models or self.DEFAULT_MODELS
        self.overall_threshold = overall_threshold
        self.min_criterion_threshold = min_criterion_threshold
        self.verbose = verbose
        self.user_query = user_query
        self.diversity_tracker = diversity_tracker
        self.difficulty = difficulty

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
                self._available_actions = "Navigate, Open, Close, Pick, Place, UseItem, Communicate, Wait"

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

        # Evaluate with all models in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.verbose:
            print(f"[Judge] Evaluating with {len(self.models)} models in parallel: {', '.join(self.models)}")

        judgments: Dict[str, Judgment] = {}
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._evaluate_single, task_dict, model, scene_data, rollout): model
                for model in self.models
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    judgments[model] = future.result()
                    if self.verbose:
                        print(f"[Judge] {model} completed")
                except Exception as e:
                    print(f"[Judge] {model} failed: {e}")
                    judgments[model] = self._failed_judgment(f"[{model}] Error: {e}")

        # Check novelty if diversity tracker is available
        if self.diversity_tracker:
            novelty_result = self.diversity_tracker.check_novelty(task_dict)
            novelty_score = CriterionScore(
                score=novelty_result["score"],
                reasoning=novelty_result["reason"],
            )
            if self.verbose:
                print(f"[Judge] Novelty check: {novelty_result['score']:.2f} - {novelty_result['reason']}")

            # Inject novelty score into each judgment
            for model, judgment in judgments.items():
                judgment.criteria_scores["task_novelty"] = novelty_score
                # Add suggestion if novelty is low
                if novelty_result["score"] < self.min_criterion_threshold:
                    similar_str = ", ".join(novelty_result.get("similar_to", [])[:3])
                    suggestion = f"[Task Novelty] Task is too similar to existing patterns"
                    if similar_str:
                        suggestion += f" (similar to: {similar_str})"
                    suggestion += ". Try a different win condition, mechanics, or dependency structure."
                    judgment.suggestions.insert(0, suggestion)
                # Recalculate overall score to include novelty
                all_scores = [c.score for c in judgment.criteria_scores.values()]
                judgment.overall_score = sum(all_scores) / len(all_scores)
                # Recalculate validity — novelty is a soft criterion (affects
                # the average but cannot single-handedly veto a task)
                hard_scores = [
                    c.score for k, c in judgment.criteria_scores.items()
                    if k != "task_novelty"
                ]
                judgment.is_valid = (
                    judgment.overall_score >= self.overall_threshold
                    and all(s >= self.min_criterion_threshold for s in hard_scores)
                )

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

        # Get task category (default to cooperative for backwards compatibility)
        category = task_dict.get("category", "cooperative")
        if category not in CATEGORY_CRITERIA:
            category = "cooperative"

        # Build user requirements section if query was provided
        user_requirements_section = ""
        if self.user_query:
            user_requirements_section = f"""
## User Requirements

The user specifically requested:
> {self.user_query}

**IMPORTANT**: The task MUST align with this request. Evaluate whether the task incorporates the requested elements (items, mechanics, themes, etc.).
"""

        # Build difficulty section
        difficulty_section = ""
        if self.difficulty and self.difficulty in DIFFICULTY_DESCRIPTIONS:
            difficulty_section = DIFFICULTY_DESCRIPTIONS[self.difficulty]

        # Build category-aware prompt
        grounding = self._get_grounding_info()
        scene_objects = self._format_scene_objects(scene_data)
        task_json = json.dumps(task_dict, indent=2)

        prompt = EVALUATION_PROMPT.format(
            category=category.upper(),
            category_description=CATEGORY_PROMPT_DESCRIPTIONS.get(category, ""),
            difficulty_section=difficulty_section,
            user_requirements_section=user_requirements_section,
            criteria_section=_build_criteria_section(category, self.user_query),
            response_format=_build_response_format(category, self.user_query),
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
            print(f"[Judge/{model}] Evaluating {category} task ({len(prompt)} chars)")

        # Get response
        response = llm.generate(prompt)

        if self.verbose:
            print(f"[Judge/{model}] Received response ({len(response)} chars)")

        # Parse response (pass user_query so it knows which criteria to expect)
        return self._parse_response(response, model, category, self.user_query)

    def _parse_response(self, response: str, model: str, category: str = "cooperative", user_query: Optional[str] = None) -> Judgment:
        """Parse LLM response into Judgment."""
        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return self._failed_judgment(f"[{model}] Failed to parse response", category)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            return self._failed_judgment(f"[{model}] JSON parse error: {e}", category)

        # Get criteria for this category (includes user_requirements_alignment if query provided)
        criteria_names = _get_criteria_for_category(category, user_query)

        criteria_scores = {}
        scores = []

        for name in criteria_names:
            if name in data and isinstance(data[name], dict):
                score = float(data[name].get("score", 0.0))
                reasoning = data[name].get("reasoning", "No reasoning provided")
                criteria_scores[name] = CriterionScore(score=score, reasoning=reasoning)
                scores.append(score)
            else:
                criteria_scores[name] = CriterionScore(score=0.0, reasoning="Missing from response")
                scores.append(0.0)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0.0

        is_valid = (
            overall_score >= self.overall_threshold
            and all(s >= self.min_criterion_threshold for s in scores)
        )

        # Extract suggestions
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]

        # Add auto-suggestions for low scores
        for name, criterion in criteria_scores.items():
            if criterion.score < self.min_criterion_threshold:
                suggestions.append(f"Improve {name} ({criterion.score:.2f}): {criterion.reasoning}")

        return Judgment(
            category=category,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            is_valid=is_valid,
            overall_reasoning=data.get("overall_reasoning", "No reasoning provided"),
            suggestions=suggestions,
        )

    def _failed_judgment(self, reason: str, category: str = "cooperative") -> Judgment:
        """Create a failed judgment for parse errors."""
        criteria_names = CATEGORY_CRITERIA.get(category, SHARED_CRITERIA)
        failed_scores = {
            name: CriterionScore(score=0.0, reasoning=reason)
            for name in criteria_names
        }
        return Judgment(
            category=category,
            criteria_scores=failed_scores,
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

        # Merge suggestions (deduplicated), prioritizing agent_necessity and secret_relevance
        priority_suggestions = []
        other_suggestions = []
        seen = set()
        for j in judgments.values():
            for s in j.suggestions:
                if s not in seen:
                    seen.add(s)
                    # Check if this suggestion relates to priority criteria
                    is_priority = any(crit in s.lower() for crit in self.PRIORITY_CRITERIA)
                    if is_priority:
                        priority_suggestions.append(s)
                    else:
                        other_suggestions.append(s)
        all_suggestions = priority_suggestions + other_suggestions

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
            lines.append(f"Category: {judgment.category}")
            lines.append(f"Score: {judgment.overall_score:.2f}")

            lines.append("\nCriteria:")
            for name, criterion in judgment.criteria_scores.items():
                icon = "+" if criterion.score >= self.min_criterion_threshold else "!"
                lines.append(f"  [{icon}] {name}: {criterion.score:.2f}")

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
        description="Evaluate task quality with category-specific criteria"
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
        "--threshold", type=float, default=0.65,
        help="Overall score threshold (default: 0.65)"
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
