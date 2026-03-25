#!/usr/bin/env python3
"""
Category-aware Task Judge for EMTOM task validation.

Evaluates tasks using category-specific criteria:
- Cooperative: agent necessity, secrets, interdependence
- Competitive: agent necessity, secrets, goal opposition
- Mixed: agent necessity, secrets, subgoal tension

Priority criteria (evaluated first, fixes prioritized):
- agent_necessity: Every agent must be indispensable
- secret_relevance: Secrets must be required for task completion

Uses a multi-LLM council (Kimi K2.5 + GPT-5.2) to reduce bias.
Both models must agree for a task to pass.

Usage:
    # CLI
    python -m emtom.task_gen.judge --task <path>
    python -m emtom.task_gen.judge --task <path> --models kimi-k2.5,gpt-5.2

    # Programmatic
    from emtom.task_gen.judge import Judge
    judge = Judge(models=["kimi-k2.5", "gpt-5.2"])
    verdict = judge.evaluate(task_data, scene_data)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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


def _detect_provider_for_model(model: str) -> str:
    """Detect provider for judge-side ad hoc LLM calls."""
    normalized = (model or "").strip().lower()

    if normalized.startswith("gpt"):
        return "openai_chat"

    if normalized.startswith("accounts/fireworks/models/"):
        return "openai_chat"

    if normalized.startswith("us.anthropic.claude-"):
        return "bedrock_claude"

    if normalized.startswith("claude-"):
        return "anthropic_claude"

    if normalized in {
        "sonnet",
        "sonnet-4.5",
        "sonnet4.5",
        "haiku",
        "haiku-4.5",
        "haiku4.5",
        "opus",
        "opus-4.5",
        "opus4.5",
    }:
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            return "anthropic_claude"
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    if key.strip() == "ANTHROPIC_API_KEY" and value.strip().strip('"').strip("'"):
                        return "anthropic_claude"
            except Exception:
                pass
        return "bedrock_claude"

    if normalized.startswith("kimi"):
        return "openai_chat"

    return "openai_chat"


def _resolve_client_model_name(model: str) -> str:
    """Expand lightweight aliases into provider-native model names."""
    normalized = (model or "").strip().lower()
    if normalized == "kimi-k2.5" and os.getenv("FIREWORKS_API_KEY", "").strip():
        return "accounts/fireworks/models/kimi-k2p5"
    return model


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
    snapshot_spec_hash: Optional[str] = None
    snapshot_task: Optional[str] = None

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

        # Load snapshot metadata when present (used to detect stale rollouts).
        snapshot_spec_hash = None
        snapshot_task = None
        snapshot_file = trajectory_dir / "task_snapshot.json"
        if snapshot_file.exists():
            try:
                snapshot = json.loads(snapshot_file.read_text())
                raw_hash = snapshot.get("spec_hash")
                if isinstance(raw_hash, str) and raw_hash.strip():
                    snapshot_spec_hash = raw_hash.strip()
                raw_task = snapshot.get("task")
                if isinstance(raw_task, str) and raw_task.strip():
                    snapshot_task = raw_task.strip()
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
            snapshot_spec_hash=snapshot_spec_hash,
            snapshot_task=snapshot_task,
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
        "description": "Does every agent make a distinct, goal-relevant contribution? Judge based on whether each agent materially changes the optimal plan or reachable success path, not on a literal proof that removing one agent makes success impossible in all circumstances.",
        "rubric": """0.0: One or more agents are idle, decorative, or obviously removable
0.3: Some agents act, but their contribution is trivial or easily absorbed by others with no real plan change
0.5: All agents participate and at least one extra agent matters somewhat, but one or more roles are still weak, redundant, or mostly relay-only
0.7: Every agent makes a material, distinct contribution; removing one would significantly weaken the task or collapse an intended dependency, even if a convoluted fallback might still exist
1.0: Every agent has a clear non-substitutable role in the intended solution; removing any agent would fundamentally break a required dependency, access path, or incentive structure""",
    },
    "secret_quality": {
        "name": "Secret Quality",
        "description": "Secrets must state constraints and goals with exact IDs, but NEVER prescribe communication strategy, relay chains, or coordination method. Agents must figure out HOW to coordinate themselves — that IS the ToM challenge.",
        "rubric": """0.0: Secrets prescribe the full relay chain or tell agents exactly what to communicate and to whom
0.3: Secrets hint at the coordination strategy (e.g., parenthetical suggestions, 'forward to agent_X', 'wait for agent_Y to tell you')
0.5: Secrets state goals and constraints but include some strategy leakage (e.g., 'you may need a relay', 'coordinate with agent_X about Y')
0.7: Secrets state only constraints (room/comm restrictions), physical roles (object IDs, locations), and end-state goals. No strategy hints remain.
1.0: Secrets are minimal and precise — only constraints, roles with exact IDs, and abstract epistemic goals. Zero communication strategy leaked.""",
    },
    "task_naturalness": {
        "name": "Public/Secret Grounding Split",
        "description": "Does the public `task` stay high-level while `agent_secrets` provide exact actionable grounding? The public task may be vague and should not read like a machine spec. Secrets should name exact IDs/states for goal-critical targets, especially when an agent cannot directly observe them.",
        "rubric": """0.0: Public task leaks exact machine-style targets and secrets are still vague or generic
0.3: Either the public task is overly specific, or the secrets still fail to identify exact target IDs/states
0.5: Split is partly right, but public task still over-specifies some targets or secrets are inconsistent in explicitness
0.7: Public task is mostly high-level and secrets are usually explicit, with only minor leakage or ambiguity
1.0: Public task is clean, high-level, and non-leaking; secrets carry the exact actionable IDs/states agents need""",
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
        "description": "Does every shared PDDL goal conjunct materially contribute to the benchmarked task, rather than acting as filler or decoration?",
        "rubric": """0.0: Goal is mostly filler or arbitrary conjuncts with no coherent task objective
0.3: Several conjuncts are decorative, redundant, or weakly related to the main objective
0.5: Core objective is present but some conjuncts could be removed without changing the task much
0.7: Most conjuncts are essential, with at most one questionable or weakly motivated goal
1.0: Every conjunct materially contributes to the task objective; removing any would meaningfully change the task""",
    },
    "pddl_solvability": {
        "name": "Formal Goal Quality & Epistemic Coherence",
        "description": "Does the self-contained `problem_pddl` define a formally meaningful task for this benchmark under the current split semantics? Treat hard formal invalidity as a near-automatic fail. Judge the physical functional core after removing epistemic conjuncts, and separately judge whether the K() structure yields meaningful literal-ToM probes grounded in genuine information asymmetry. Strong scores require FUNCTIONAL ToM pressure: success should depend on choosing actions based on partner-specific private information, not just relaying hidden facts.",
        "rubric": """0.0: Raw problem_pddl is invalid, contradictory, or impossible; or the functional projection becomes vacuous/single-agent/trivial; or K() goals are fake/decorative
0.3: Barely benchmark-meaningful: weak functional core, shaky category logic, or K() probes are loosely attached to irrelevant facts / pure relay events
0.5: Formally valid task, but either the functional projection is weak after dropping K(), or the epistemic structure is mostly literal hidden-fact reporting rather than partner-dependent action choice
0.7: Strong functional core plus mostly meaningful K()-derived probes, with some genuine partner-modeling pressure and only minor weaknesses
1.0: Self-contained, formally coherent, and benchmark-meaningful under split semantics: the functional projection remains strong, and the task requires adapting to partner-specific knowledge, access, incentives, or communication limits rather than merely forwarding facts""",
    },
    "mechanic_utilization": {
        "name": "Mechanic Utilization & Balance",
        "description": "Are the listed mechanics genuinely used to create the intended coordination or ToM challenge? Penalize mechanics that are redundant, decorative, or disconnected from the goal structure. Empty mechanics are fine when the task is intentionally simple.",
        "rubric": """0.0: Mechanics are listed but unused, misleading, or disconnected from the actual task
0.3: Mechanics appear in the spec but most could be removed without changing the challenge
0.5: Mechanics matter somewhat, but at least one is decorative or only weakly connected to the benchmark challenge
0.7: Mechanics are well integrated and each serves a distinct purpose, with only minor redundancy
1.0: Mechanics are tightly integrated with the formal goals and task design; each one materially contributes to the benchmark challenge""",
    },
    # Cooperative-specific
    "task_interdependence": {
        "name": "Task Interdependence",
        "description": "Do agents genuinely NEED information from each other to complete physical goals? At least one physical goal must be information-dependent: an agent cannot determine WHAT to do or WHERE to act without receiving a message from another agent. If all physical goals can be completed in parallel without any communication, score 0.",
        "rubric": """0.0: All physical goals are independently solvable — agents can complete everything in parallel without communicating
0.3: Agents help but aren't required, or interdependence is just a hidden-fact relay with no impact on physical goal completion
0.5: Some interdependence but key physical steps can be done solo or without modeling which teammate is best positioned
0.7: Strong interdependence with partner-specific dependencies, though some steps are still generic handoffs
1.0: Impossible for any single agent to succeed, and rational success depends on choosing actions based on what specific teammates know, can access, or are likely to prioritize""",
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
        "description": "Does every agent have a personal objective in `:goal-owners`, and do they create meaningful tension that changes how teammates should coordinate with them?",
        "rubric": """0.0: No personal objectives in :goal-owners, or goals trivially satisfied
0.3: Some agents have personal objectives but they do not affect partner expectations or coordination
0.5: Personal objectives exist for most agents with minor tension, but teammates can mostly ignore them
0.7: Every agent has a personal objective; meaningful conflicts require strategic choices about who to trust, inform, or rely on
1.0: Every agent has a personal objective; real dilemmas where pursuing them risks the main goal, and success depends on modeling which teammate is likely to deviate or cooperate""",
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
    required_fixes: List[str] = field(default_factory=list)

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
            "required_fixes": self.required_fixes,
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
    required_fixes: List[str]  # Merged from all models, deduplicated
    disagreements: List[str] = field(default_factory=list)  # Where models disagreed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "model_judgments": {
                model: j.to_dict() for model, j in self.judgments.items()
            },
            "required_fixes": self.required_fixes,
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

## System Capabilities (use these in required fixes)
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
- Secrets must be explicit and actionable, ideally naming exact target IDs/states, and not step-by-step
- Single-format goal source is `problem_pddl`
- Runtime semantics are split:
  - functional benchmark success uses the non-epistemic projection only
  - `K()` goals are design-time / probe-time only and become end-of-episode literal-ToM probes
- Category intent must be reflected in `problem_pddl` objective structure
- Raw `problem_pddl` should be self-contained for scene/world facts: required symbols belong in `:objects`, relevant room grounding belongs in `:init`, while mechanic-derived init facts like room restrictions should come from `mechanic_bindings`
- For solvability and mechanic-awareness, treat the Compiled Formal View below as authoritative. Do NOT penalize raw `problem_pddl` for omitting mechanic-derived init facts such as `is_restricted` or `can_communicate`; those are expected to be compiled from `mechanic_bindings`.
- **Mechanic consistency**: Every mechanic referenced in `task` or `agent_secrets` (e.g., "the handle is reversed", "you have limited messages") MUST have a corresponding entry in `mechanic_bindings`. `message_targets` is a valid standalone way to encode communication restrictions and does not require a duplicate `restricted_communication` binding.
- **K() goal backing**: Every `K()` goal in `problem_pddl` (or legacy goal field) must be backed by a mechanic that prevents the agent from directly observing the fact (e.g., `room_restriction` blocks navigation, `restricted_communication` blocks direct messaging). If the agent could just walk there and see, the K() goal is fake.
- **Functional projection quality**: Penalize tasks whose non-epistemic projection becomes vacuous, trivial, effectively single-agent, or no longer reflects the intended coordination challenge.
- **Probe quality**: Reward K() goals when they probe who knows functionally relevant facts under real asymmetry. Do NOT require K() to be part of runtime pass/fail.
- **Functional ToM quality**: Reward tasks where the best action depends on a partner-specific model (who can act, who will relay, who may prioritize a private goal, who has the last message). Penalize tasks that reduce to "agent A sees a hidden fact and tells agent B."
- Distinguish **formal validity** from **design quality**:
  - If the formal task is invalid, contradictory, or not self-contained, score `Formal Goal Quality & Epistemic Coherence` near 0.
  - If the formal task is valid, judge whether both the functional projection and the literal-ToM probe structure are meaningful for the benchmark rather than merely technically valid.

## Derived Runtime View
Use this derived runtime view when judging split-semantics quality.
{runtime_semantics_section}

## Compiled Formal View
Use this normalized formal problem when checking mechanic-aware solvability claims.
It reflects the authored `problem_pddl` after mechanic-derived and planner-required
init facts are compiled in. This is the authoritative formal view for
`pddl_solvability`; raw `problem_pddl` may omit mechanic-derived init facts.
{compiled_formal_view_section}

## Benchmark Comparison
Use this when calibration data exists.
{benchmark_comparison_section}

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
  "required_fixes": ["<minimum concrete fix required to pass>", "..."]
}}

## Required Fix Rules
- List only the minimum concrete changes required for this task to pass.
- Do NOT include optional improvements, stretch ideas, or alternate redesigns.
- Prefer 1-3 fixes total.
- Preserve the current scene, category, and main objects/goals when possible.
- Be specific and only use available system capabilities.
- Prefer fixes that strengthen the functional projection after dropping K(), make K()-derived probes more grounded and informative without making them runtime success conditions, and increase partner-dependent action choice rather than fact relay.
- If the task already passes, return `"required_fixes": []`.
"""

# Category descriptions for the prompt
CATEGORY_PROMPT_DESCRIPTIONS = {
    "cooperative": """**COOPERATIVE** - All agents united toward shared goals
- Every agent contributes unique knowledge, skills, or access that others lack
- Information is distributed: one agent might know key locations, another knows which locks need which keys
- Success requires piecing together distributed information through communication
- Complex tasks can have parallel workstreams that converge
- Uses `agent_secrets` to distribute knowledge and shared objective in `problem_pddl`""",
    "competitive": """**COMPETITIVE** - Teams with opposing objectives
- Divide agents into teams (any split: 1v1, 2v1, 2v2, 3v2, etc.)
- Teams compete for contested resources OR race to complete opposing objectives
- Each team member should contribute - divide responsibilities within teams
- Balance matters: if teams are uneven in size, give smaller team easier objectives
- Uses `teams` mapping and encodes opposition directly in `problem_pddl` (e.g., OR over team-winning conditions)
- Public `task` must be symmetric; team-specific targets belong in secrets""",
    "mixed": """**MIXED** - Cooperation with hidden personal objectives
- All agents share a main objective in `problem_pddl :goal`
- Each agent MUST have a personal objective in `:goal-owners` (supplementary, not part of :goal)
- Personal objectives create tension: they may conflict with the main goal or with other agents' objectives
- `agent_secrets` should hint at each agent's personal objective in natural language
- Public `task` must describe ONLY the shared objective; personal objectives belong in secrets""",
}


DIFFICULTY_DESCRIPTIONS = {
    "easy": """## Intended Difficulty: EASY
This task is designed for WEAKER models. Calibrate your evaluation accordingly:
- **Agent necessity**: 2-3 agents with clear, distinct roles is sufficient. Simple role division (e.g., one agent fetches, another places) counts as high agent necessity.
- **Task interdependence / goal opposition / subgoal tension**: Simple dependencies are fine. One clear handoff or information exchange between agents is enough.
- **Secret quality**: Secrets should state constraints, roles, and goals with exact IDs. Mechanic hints (e.g., "the handle is reversed" for inverse_state) are required. But secrets must NEVER prescribe coordination strategy — no "tell agent_1 about X", "forward to agent_0", or parenthetical suggestions. Score LOW if secrets tell agents HOW to coordinate.
- **Mechanic utilization**: Using 0-1 mechanics is sufficient. Prefer simple, observable mechanics. Avoid stacking multiple mechanics.
- **Overall**: A well-structured simple task with clear agent roles, mechanic hints in secrets, and basic ToM should score HIGH. Do NOT penalize simplicity.""",
    "medium": """## Intended Difficulty: MEDIUM
This task targets mid-tier models. Standard evaluation applies:
- Agents should have meaningful distinct roles with some interdependence.
- Secrets should require reasoning to use effectively.
- Tasks should use mechanics appropriately (typically 2-4).
- Moderate complexity in coordination is expected.""",
    "hard": """## Intended Difficulty: HARD — Must defeat GPT-5.2
This task must be difficult enough that GPT-5.2 CANNOT solve it. Apply strict standards:
- **Agent necessity**: Each agent MUST hold unique information. Score LOW if any agent is removable.
- **Task interdependence / goal opposition / subgoal tension**: Require information relay chains. Score LOW unless at least one goal depends on relayed (not directly observed) information.
- **Secret quality**: Secrets must state only constraints (room/comm restrictions), physical roles (exact object IDs), and abstract epistemic goals. NEVER prescribe relay chains, communication strategy, or what to tell other agents. Score 0 if any secret says "tell agent_X", "forward to agent_X", or includes parenthetical strategy hints.
- **Mechanic utilization**: limited_bandwidth MUST be present with 1 message per agent. 1-2 mechanics total is fine — complexity should come from ToM reasoning, not mechanic stacking. Score LOW if bandwidth > 1 per agent.
- **Overall**: The task should require genuine Theory of Mind reasoning. Reward tasks where agents must infer what others know. Do NOT require complex mechanics — difficulty from information asymmetry is preferred.""",
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


def _build_runtime_semantics_section(task_dict: Dict[str, Any]) -> str:
    """Summarize the derived functional goal and literal-ToM probes for judge prompts."""
    functional_goal = task_dict.get("functional_goal_pddl")
    probes = task_dict.get("literal_tom_probes")

    if not functional_goal or probes is None:
        try:
            from emtom.pddl.runtime_projection import build_runtime_metadata

            derived = build_runtime_metadata(task_dict)
            functional_goal = functional_goal or derived.get("functional_goal_pddl")
            probes = probes if probes is not None else derived.get("literal_tom_probes", [])
        except Exception:
            probes = probes or []

    lines = []
    if functional_goal:
        lines.append("Functional goal projection used for runtime success:")
        lines.append(functional_goal)
    else:
        lines.append("Functional goal projection: <unavailable>")

    if probes:
        lines.append("Literal-ToM probes derived from K() goals:")
        for probe in probes[:8]:
            source = probe.get("source_pddl", "<unknown>")
            agent = probe.get("agent_id", "<unknown>")
            question = probe.get("question", "").strip().splitlines()[0] if probe.get("question") else ""
            lines.append(f"- {agent}: {source}")
            if question:
                lines.append(f"  Probe question stem: {question}")
        if len(probes) > 8:
            lines.append(f"- ... {len(probes) - 8} more probes")
    else:
        lines.append("Literal-ToM probes: none derived")

    return "\n".join(lines)


def _build_compiled_formal_view_section(
    task_dict: Dict[str, Any],
    scene_data: Optional["SceneData"],
) -> str:
    """Build the normalized formal problem view used by the verifier."""
    try:
        from emtom.pddl.compiler import compile_task
        from emtom.task_gen.task_generator import GeneratedTask

        task = GeneratedTask.from_dict(task_dict)
        scene_payload: Optional[Dict[str, Any]]
        if scene_data is None:
            scene_payload = None
        elif hasattr(scene_data, "to_dict"):
            scene_payload = scene_data.to_dict()  # type: ignore[assignment]
        elif isinstance(scene_data, dict):
            scene_payload = scene_data
        else:
            scene_payload = None

        compiled = compile_task(task, scene_payload)
        lines = [
            "Compiled problem used for mechanic-aware solvability checks:",
            "```lisp",
            compiled.to_pddl(),
            "```",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Compiled formal view unavailable: {exc}"


def _build_benchmark_comparison_section(task_dict: Dict[str, Any]) -> str:
    """Summarize the latest standard vs baseline calibration pair when present."""
    from emtom.evolve.benchmark_wrapper import _migrate_legacy_calibration

    calibration = _migrate_legacy_calibration(task_dict.get("calibration", []))
    if not calibration:
        return "Benchmark comparison: none recorded yet."

    latest_standard = None
    for entry in calibration:
        run_mode = str(entry.get("run_mode", "standard") or "standard")
        if run_mode == "standard":
            latest_standard = entry
    if latest_standard is None:
        return "Benchmark comparison: no standard calibration recorded yet."

    latest_baseline = None
    target_models = latest_standard.get("agent_models", {})
    for entry in calibration:
        run_mode = str(entry.get("run_mode", "standard") or "standard")
        if run_mode == "baseline" and entry.get("agent_models") == target_models:
            latest_baseline = entry
    if latest_baseline is None:
        return "Benchmark comparison: standard recorded, but no matching baseline calibration yet."

    def _results_summary(entry: Dict[str, Any]) -> str:
        results = entry.get("results", {})
        if "main_goal" in results:
            return (
                f"passed={results['main_goal'].get('passed', False)}, "
                f"progress={results['main_goal'].get('progress', 0.0):.0%}"
            )
        if "winner" in results:
            teams = results.get("teams", {})
            max_progress = max((team.get("progress", 0.0) for team in teams.values()), default=0.0)
            return f"winner={results.get('winner')}, progress={max_progress:.0%}"
        return (
            f"passed={results.get('passed', False)}, "
            f"progress={results.get('progress', 0.0):.0%}"
        )

    lines = [
        "Latest benchmark comparison:",
        f"- Standard: {_results_summary(latest_standard)}",
        f"- Baseline: {_results_summary(latest_baseline)}",
        (
            "Interpretation: stronger functional-ToM evidence comes from tasks where the "
            "baseline/full-info run succeeds and the standard run is materially weaker."
        ),
    ]
    return "\n".join(lines)


class Judge:
    """
    Multi-LLM council judge for task validation.

    Evaluates tasks using category-specific criteria:
    - Cooperative: 6 criteria (5 shared + task_interdependence)
    - Competitive: 6 criteria (5 shared + goal_opposition)
    - Mixed: 6 criteria (5 shared + subgoal_tension)

    Priority criteria (required fixes appear first):
    - agent_necessity: Every agent must be essential
    - secret_quality: Secrets must be actionable, natural, and non-leaking

    Both models must agree for a task to pass.
    """

    # Priority criteria - required fixes should focus here first
    PRIORITY_CRITERIA = ["agent_necessity", "secret_quality"]

    # Default council models
    DEFAULT_MODELS = ["kimi-k2.5", "gpt-5.2"]
    MODEL_REQUEST_TIMEOUT_S = 45
    COUNCIL_WALL_TIMEOUT_S = 180
    COUNCIL_RETRY_ATTEMPTS = 1
    COUNCIL_RETRY_TIMEOUT_S = 300  # Longer timeout on retry

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
            models: List of model names for council (default: ["kimi-k2.5", "gpt-5.2"])
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
            instantiate_llm = None  # lazy import below
            # Avoid importing LLM backends (which may pull in habitat/habitat_sim)
            # when the caller explicitly disables the LLM council.
            if os.environ.get("TASKGEN_SKIP_LLM", "").lower() in {"1","true","yes"}:
                class _DummyLLM:
                    def generate(self, prompt, **kwargs):
                        return "{}"
                    def generate(self, prompt, **kwargs):
                        return "{}"
                self._llm_clients[model] = _DummyLLM()
                return self._llm_clients[model]

            provider = _detect_provider_for_model(model)
            client_model = _resolve_client_model_name(model)
            # Allow disabling LLM council in lightweight taskgen environments.
            if os.environ.get("TASKGEN_SKIP_LLM", "").lower() in {"1","true","yes"}:
                class _DummyLLM:
                    def generate(self, prompt, **kwargs):
                        return "{}"
                    def generate(self, prompt, **kwargs):
                        return "{}"
                self._llm_clients[model] = _DummyLLM()
                return self._llm_clients[model]

            from habitat_llm.llm import instantiate_llm
            self._llm_clients[model] = instantiate_llm(
                provider,
                generation_params={
                    "model": client_model,
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
            scene_data: Optional scene data for grounded required fixes
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
            if rollout:
                current_spec_hash = None
                try:
                    from emtom.pddl.planner import compute_task_spec_hash

                    current_spec_hash = compute_task_spec_hash(task_dict)
                except Exception:
                    current_spec_hash = None

                # Only trust rollout if snapshot metadata exists and matches current spec.
                if not rollout.snapshot_spec_hash:
                    if self.verbose:
                        print("[Judge] Ignoring rollout: missing task_snapshot.json/spec_hash metadata")
                    rollout = None
                elif current_spec_hash and rollout.snapshot_spec_hash != current_spec_hash:
                    if self.verbose:
                        print(
                            "[Judge] Ignoring stale rollout: snapshot spec hash does not match current task"
                        )
                    rollout = None
                elif self.verbose:
                    print(f"[Judge] Loaded rollout: success={rollout.success}, {rollout.steps} steps")

        # Evaluate with all models in parallel
        # Lightweight fallback: if Hydra (Habitat dependency) is not importable,
        # skip the LLM council and return a pass verdict. This keeps taskgen
        # usable in minimal CI containers where only JSON/PDDL validation runs.
        try:
            import hydra  # type: ignore
        except Exception as e:
            return CouncilVerdict(
                judgments={},
                passed=True,
                overall_score=1.0,
                required_fixes=[],
                disagreements=[],
            )

        from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

        if self.verbose:
            print(f"[Judge] Evaluating with {len(self.models)} models in parallel: {', '.join(self.models)}")

        judgments: Dict[str, Judgment] = {}
        models_to_evaluate = list(self.models)

        for attempt in range(1 + self.COUNCIL_RETRY_ATTEMPTS):
            timeout = self.COUNCIL_WALL_TIMEOUT_S if attempt == 0 else self.COUNCIL_RETRY_TIMEOUT_S
            executor = ThreadPoolExecutor(max_workers=len(models_to_evaluate))
            future_to_model = {
                executor.submit(self._evaluate_single, task_dict, model, scene_data, rollout): model
                for model in models_to_evaluate
            }
            try:
                done, not_done = wait(
                    set(future_to_model.keys()),
                    timeout=timeout,
                    return_when=ALL_COMPLETED,
                )
                for future in done:
                    model = future_to_model[future]
                    try:
                        judgments[model] = future.result()
                        if self.verbose:
                            print(f"[Judge] {model} completed")
                    except Exception as e:
                        print(f"[Judge] {model} failed: {e}")
                        judgments[model] = self._failed_judgment(
                            f"[{model}] Error: {e}",
                            category=task_dict.get("category", "cooperative"),
                        )

                timed_out_models = []
                for future in not_done:
                    model = future_to_model[future]
                    reason = (
                        f"[{model}] Timed out after {timeout}s "
                        "waiting for judge model response"
                    )
                    print(f"[Judge] {reason}")
                    judgments[model] = self._failed_judgment(
                        reason,
                        category=task_dict.get("category", "cooperative"),
                    )
                    timed_out_models.append(model)
                    future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            # Retry timed-out models (infrastructure failures only)
            retry_models = [
                m for m in timed_out_models
                if m in judgments and self._is_infra_failure(judgments[m])
            ]
            if not retry_models or attempt >= self.COUNCIL_RETRY_ATTEMPTS:
                break
            print(f"[Judge] Retrying {len(retry_models)} timed-out model(s): {', '.join(retry_models)}")
            models_to_evaluate = retry_models

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
                # Add a required fix if novelty is low.
                if novelty_result["score"] < self.min_criterion_threshold:
                    similar_str = ", ".join(novelty_result.get("similar_to", [])[:3])
                    suggestion = f"[Task Novelty] Task is too similar to existing patterns"
                    if similar_str:
                        suggestion += f" (similar to: {similar_str})"
                    suggestion += ". Try a different win condition, mechanics, or dependency structure."
                    if suggestion not in judgment.required_fixes:
                        judgment.required_fixes.insert(0, suggestion)
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
            runtime_semantics_section=_build_runtime_semantics_section(task_dict),
            compiled_formal_view_section=_build_compiled_formal_view_section(task_dict, scene_data),
            benchmark_comparison_section=_build_benchmark_comparison_section(task_dict),
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

        # Get response with retries for transient provider/network failures.
        # Use aggressive backoff with jitter to handle rate limits (429)
        # under high parallelism (e.g. 128 concurrent bulk gen processes).
        response = None
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            try:
                response = llm.generate(
                    prompt,
                    request_timeout=self.MODEL_REQUEST_TIMEOUT_S,
                )
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                # Longer backoff for rate limits (429), shorter for other errors
                exc_str = str(exc)
                if "429" in exc_str or "too_many_requests" in exc_str or "overloaded" in exc_str:
                    base_backoff = min(120, 15 * (2 ** (attempt - 1)))
                else:
                    base_backoff = min(30, 2 ** attempt)
                # Add jitter to avoid thundering herd
                import random
                backoff_s = base_backoff * (0.5 + random.random())
                if self.verbose:
                    print(
                        f"[Judge/{model}] LLM call failed (attempt {attempt}/{max_retries}), "
                        f"retrying in {backoff_s:.0f}s"
                    )
                time.sleep(backoff_s)

        if self.verbose:
            print(f"[Judge/{model}] Received response ({len(response or '')} chars)")

        # Parse response (pass user_query so it knows which criteria to expect)
        return self._parse_response(response or "", model, category, self.user_query)

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

        # Extract minimum required fixes.
        raw_required_fixes = data.get("required_fixes", [])
        if not isinstance(raw_required_fixes, list):
            raw_required_fixes = [str(raw_required_fixes)]
        required_fixes: List[str] = []
        for fix in raw_required_fixes:
            fix_text = str(fix).strip()
            if fix_text and fix_text not in required_fixes:
                required_fixes.append(fix_text)

        # Fall back to criterion-scoped fixes only when the model omitted
        # concrete required_fixes entirely.
        if not required_fixes:
            for name, criterion in criteria_scores.items():
                if criterion.score < self.min_criterion_threshold:
                    fallback_fix = f"Fix {name} ({criterion.score:.2f}): {criterion.reasoning}"
                    if fallback_fix not in required_fixes:
                        required_fixes.append(fallback_fix)

        if not required_fixes and not is_valid and overall_score < self.overall_threshold:
            required_fixes.append(
                f"Raise overall task quality above {self.overall_threshold:.2f} without weakening current strengths."
            )

        return Judgment(
            category=category,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            is_valid=is_valid,
            overall_reasoning=data.get("overall_reasoning", "No reasoning provided"),
            required_fixes=required_fixes[:3],
        )

    @staticmethod
    def _is_infra_failure(judgment: "Judgment") -> bool:
        """Check if a judgment failed due to infrastructure (not task quality)."""
        reason = (judgment.overall_reasoning or "").lower()
        markers = (
            "429", "too_many_requests", "overloaded", "rate limit",
            "connection error", "could not connect", "timed out",
            "endpoint url", "name resolution", "error:",
        )
        return judgment.overall_score == 0.0 and any(m in reason for m in markers)

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
            required_fixes=["Re-run evaluation"],
        )

    def _aggregate(self, judgments: Dict[str, Judgment]) -> CouncilVerdict:
        """Aggregate judgments from multiple models."""
        # If ANY model failed due to infrastructure, we can optionally continue
        # with remaining models in environments where some providers are not
        # configured (e.g., missing FIREWORKS_API_KEY for Kimi).
        infra_failures = {
            m: j for m, j in judgments.items()
            if self._is_infra_failure(j)
        }
        if infra_failures:
            if os.getenv("TASKGEN_REQUIRE_FULL_COUNCIL", "").strip().lower() in {"1", "true", "yes"}:
                # NOTE: Avoid backslashes inside f-string expressions (SyntaxError on Python 3.9).
                failed_models = ", ".join(infra_failures.keys())
                raise RuntimeError(
                    "Judge council incomplete: "
                    f"{failed_models} failed due to infrastructure errors. "
                    "All council models are required for consistent task quality. "
                    "Check API keys, billing, and network connectivity."
                )
            else:
                failed_models = ", ".join(infra_failures.keys())
                print(
                    f"[Judge] WARNING: {failed_models} failed (infrastructure). "
                    "Continuing with remaining models. Set TASKGEN_REQUIRE_FULL_COUNCIL=true to enforce all models."
                )
                for m in list(infra_failures.keys()):
                    judgments.pop(m, None)

        if not judgments:
            raise RuntimeError("Judge council incomplete: all models failed due to infrastructure errors.")


        # Check if all models pass
        all_pass = all(j.is_valid for j in judgments.values())

        # Average overall scores
        avg_score = sum(j.overall_score for j in judgments.values()) / len(judgments)

        # Merge required fixes (deduplicated) across models.
        required_fixes = []
        seen = set()
        for j in judgments.values():
            for fix in j.required_fixes:
                if fix not in seen:
                    seen.add(fix)
                    required_fixes.append(fix)

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
            required_fixes=required_fixes[:3],
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

        if verdict.required_fixes:
            lines.append("\nRequired Fixes:")
            for i, s in enumerate(verdict.required_fixes[:10], 1):
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
        "--models", type=str, default="kimi-k2.5,gpt-5.2",
        help="Comma-separated list of models for council (default: kimi-k2.5,gpt-5.2)"
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
