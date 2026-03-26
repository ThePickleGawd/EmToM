"""
Agentic task generator for EMTOM benchmark.

A ReAct-style agent with 3 tools:
- bash: Shell commands for file exploration and task editing
- test_task: Run benchmark with current working_task.json
- submit_task: Save current task to output directory
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .judge import Judge, Judgment, CouncilVerdict, Colors
from .authoring_surface import (
    get_authoring_default_actions,
)
from .diversity import DiversityTracker
from .seed_sanitizer import sanitize_task_for_seeding
from .seed_selector import SeedSelectionConfig, build_seed_candidates, select_seed_tasks
from .spec_validator import (
    validate_blocking_spec,
    validate_room_restriction_trajectory,
)
from .task_bootstrap import build_scene_bootstrap_problem_pddl

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


# Markers for the diversity section in the system prompt so it can be found and replaced
_DIVERSITY_START_MARKER = "<!-- DIVERSITY_SECTION_START -->"
_DIVERSITY_END_MARKER = "<!-- DIVERSITY_SECTION_END -->"
_CALIBRATION_TOLERANCE = 0.05

_SYSTEM_PROMPT = """You are a puzzle designer creating multi-agent collaboration benchmark tasks.

## Response Format
Respond in this format:
Thought: [brief reasoning]
Action: tool_name[args]

Exactly one action per turn.

## Tools
- `new_scene[N]`: load a fresh scene with N agents. `N` must be within the run's allowed agent range and never below 2.
- `new_scene[N, keep]`: keep the current scene but change the agent count. `N` must be within the run's allowed agent range and never below 2.
- `bash[cmd]`: inspect files and edit the working task.
- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.
- `test_task[]`: run benchmark calibration when enabled.
- `submit_task[]`: save a passing task.
- `fail[reason]`: stop only for unrecoverable infrastructure issues.

## Workflow
1. Call `new_scene[N]`.
2. Inspect the scene and any sampled seed tasks.
3. Edit `{task_file}`.
4. Run `judge[]`, fix issues, and repeat until it passes.
5. Run `test_task[]` when required.
6. Run `submit_task[]`.

## Hard Rules
- Every command already starts inside `{working_dir}`. Do not prefix every command with `cd {working_dir} &&`.
- Use only valid agent IDs and scene IDs.
- Remove placeholder text. No `TODO`, `TBD`, or generic filler.
- Every essential agent must contribute distinct knowledge, access, or incentive.
- The physical goal must require communication or partner modeling, not just parallel independent work.
- `message_targets` already defines communication restrictions. Do not duplicate it unless you intentionally need an explicit `restricted_communication` mechanic binding.
- Use canonical mechanic schema only:
  `room_restriction` -> `restricted_rooms` + `for_agents`
  `limited_bandwidth` -> `message_limits`
  `restricted_communication` -> `allowed_targets`
- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.
- Do not hand-edit `:objects`, `:init`, or `golden_trajectory`.

## Secret Formatting Rules (judge hard-blocks on violations)
- Secrets must state ONLY facts: room bans, object IDs, goal states, and knowledge gaps.
- NEVER use prescriptive language: 'Tell your partner', 'Ask them', 'Leave it at', 'Coordinate with', 'You should'.
- NEVER describe other agent's knowledge: 'agent_1 knows X'. Instead: 'You do not know X'.
- For K() goals: 'By the end, you must be confident about whether [furniture] in [room] is [state].'
- BUG WARNING: 'agent_X cannot enter room_Y' in agent_Z's secrets is parsed as agent_Z's restriction. Use 'agent_X is barred from room_Y' for other agents' restrictions.

## Category Rules
- Cooperative: shared goals only; no `teams`; no `team_secrets`; no `:goal-owners`.
- Competitive: exactly two teams; public task stays neutral; use incompatible team outcomes.
- Mixed: public task covers the shared objective only; each relevant agent has a hidden personal objective.

## Good ToM
- Good ToM means an agent's correct action depends on another agent's private knowledge, access, or likely behavior.
- Good pattern: agent A cannot determine the right object, room, or target state until agent B observes or communicates it.
- Bad pattern: agents can finish the physical goal independently and communication only reports progress.
- Use `K()` only for facts that matter for planning or coordination.
- The outermost `K()` agent should not be able to directly observe the fact with no blocker.

## References
- Working task: `{task_file}`
- Current scene: `{working_dir}/current_scene.json`
- Sampled seed summary: `{working_dir}/sampled_tasks/SUMMARY.md`
- Sampled seeds: `{working_dir}/sampled_tasks/`
- Template: `{working_dir}/template.json`

## Structural Diversity
{diversity_section}
"""

_USER_PROMPT_TEMPLATE = """Generate {num_tasks} quality benchmark tasks.
{extra_sections}
## Constraints
- Agents: {agents_min}-{agents_max}
- Goal conjuncts: {subtasks_min}-{subtasks_max}

Start with `new_scene[N]`, where `N` must be between {agents_min} and {agents_max} inclusive.
"""


def _strip_pddl_from_guidance(guidance: str) -> str:
    guidance = guidance.replace(
        "- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.\n",
        "- `judge[]`: run non-PDDL validation and quality checks.\n",
    )
    guidance = guidance.replace(
        "- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.\n",
        "- PDDL solvability verification is disabled, but you MUST still write `problem_pddl` as the canonical goal format. Author `:goal` and optional `:goal-owners` normally.\n",
    )
    guidance = guidance.replace(
        "- Do not hand-edit `:objects`, `:init`, or `golden_trajectory`.\n",
        "- Do not hand-author `golden_trajectory`.\n",
    )
    return guidance


def _strip_simulation_from_guidance(guidance: str) -> str:
    return guidance.replace(
        "- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.\n",
        "- `judge[]`: run validation, planner checks when enabled, and quality checks.\n",
    )


def _evaluation_passed(category: str, evaluation: Dict[str, Any]) -> bool:
    """Return whether the benchmark succeeded for the task category."""
    if category == "competitive":
        return evaluation.get("winner") is not None
    if category == "mixed":
        return evaluation.get("main_goal_success", False)
    return evaluation.get("success", False)


def _evaluation_progress(category: str, evaluation: Dict[str, Any]) -> float:
    """Return the benchmark progress value to use for gating/reporting."""
    if category == "mixed":
        return evaluation.get("main_goal_progress", evaluation.get("percent_complete", 0.0))
    return evaluation.get("percent_complete", 0.0)


def _build_results_block(category: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Convert benchmark evaluation into calibration-friendly results."""
    if category == "competitive":
        teams: Dict[str, Any] = {}
        for team_id, prog in evaluation.get("team_progress", {}).items():
            teams[team_id] = {"progress": prog}
        for team_id, status in evaluation.get("team_status", {}).items():
            teams.setdefault(team_id, {})["passed"] = status
        return {"winner": evaluation.get("winner"), "teams": teams}

    if category == "mixed":
        agents = {
            aid: {"subgoal_passed": passed}
            for aid, passed in evaluation.get("agent_subgoal_status", {}).items()
        }
        return {
            "main_goal": {
                "passed": evaluation.get("main_goal_success", False),
                "progress": _evaluation_progress(category, evaluation),
            },
            "agents": agents,
        }

    return {
        "passed": evaluation.get("success", False),
        "progress": _evaluation_progress(category, evaluation),
    }


def _standard_requirement(
    current_rate: Optional[float],
    target_rate: float,
    tolerance: float = _CALIBRATION_TOLERANCE,
    current_passed: Optional[int] = None,
    current_failed: Optional[int] = None,
) -> str:
    """Return how the next standard run should calibrate the dataset."""
    if current_passed is not None and current_failed is not None:
        total = current_passed + current_failed
        next_total = total + 1
        pass_rate_if_pass = (current_passed + 1) / next_total
        pass_rate_if_fail = current_passed / next_total
        pass_delta = abs(pass_rate_if_pass - target_rate)
        fail_delta = abs(pass_rate_if_fail - target_rate)
        if abs(pass_delta - fail_delta) <= 1e-9:
            return "either"
        return "must_pass" if pass_delta < fail_delta else "must_fail"

    if current_rate is None:
        return "either"
    if current_rate > target_rate + tolerance:
        return "must_fail"
    if current_rate < target_rate - tolerance:
        return "must_pass"
    return "either"


def _build_mode_comparison(
    category: str,
    standard: Dict[str, Any],
    baseline: Dict[str, Any],
    current_rate: Optional[float],
    target_rate: float,
    tolerance: float = _CALIBRATION_TOLERANCE,
    current_passed: Optional[int] = None,
    current_failed: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize the dual benchmark run and acceptance gates."""
    std_eval = standard.get("evaluation", {})
    base_eval = baseline.get("evaluation", {})
    standard_passed = _evaluation_passed(category, std_eval)
    baseline_passed = _evaluation_passed(category, base_eval)
    standard_progress = _evaluation_progress(category, std_eval)
    baseline_progress = _evaluation_progress(category, base_eval)
    requirement = _standard_requirement(
        current_rate,
        target_rate,
        tolerance=tolerance,
        current_passed=current_passed,
        current_failed=current_failed,
    )
    next_rate_if_pass = None
    next_rate_if_fail = None
    if current_passed is not None and current_failed is not None:
        next_total = current_passed + current_failed + 1
        next_rate_if_pass = (current_passed + 1) / next_total
        next_rate_if_fail = current_passed / next_total

    gate_passed = baseline_passed
    reasons: List[str] = []
    if not baseline_passed:
        gate_passed = False
        reasons.append(
            "Baseline/full-info run must pass so the task is empirically solvable when information asymmetry is removed."
        )
    if requirement == "must_fail" and standard_passed:
        gate_passed = False
        reasons.append(
            f"Standard run must fail because another pass would move the dataset away from the {target_rate:.0%} target."
        )
    elif requirement == "must_pass" and not standard_passed:
        gate_passed = False
        reasons.append(
            f"Standard run must pass because another fail would move the dataset away from the {target_rate:.0%} target."
        )

    if not reasons:
        reasons.append("Baseline passed and the standard result matches the current calibration target.")

    return {
        "gate_passed": gate_passed,
        "functional_tom_signal": baseline_passed,
        "standard_requirement": requirement,
        "current_standard_pass_rate": current_rate,
        "target_standard_pass_rate": target_rate,
        "next_standard_pass_rate_if_pass": next_rate_if_pass,
        "next_standard_pass_rate_if_fail": next_rate_if_fail,
        "standard_passed": standard_passed,
        "baseline_passed": baseline_passed,
        "standard_progress": standard_progress,
        "baseline_progress": baseline_progress,
        "progress_delta": baseline_progress - standard_progress,
        "standard_turns": standard.get("turns", 0),
        "baseline_turns": baseline.get("turns", 0),
        "turn_delta": standard.get("turns", 0) - baseline.get("turns", 0),
        "standard_steps": standard.get("steps", 0),
        "baseline_steps": baseline.get("steps", 0),
        "step_delta": standard.get("steps", 0) - baseline.get("steps", 0),
        "reasons": reasons,
    }


class TaskGeneratorAgent:
    """
    ReAct-style agent for iterative task generation.

    Uses only 3 tools:
    - bash: For file exploration and task editing
    - test_task: Run benchmark with current task
    - submit_task: Save curated task to output
    """

    def __init__(
        self,
        llm_client: "BaseLLM",
        config: DictConfig,
        working_dir: str = "data/emtom/tasks",
        output_dir: str = "data/emtom/tasks",
        iterations_per_task: int = 100,
        verbose: bool = True,
        subtasks_min: int = 2,
        subtasks_max: int = 5,
        agents_min: int = 2,
        agents_max: int = 2,
        scene_data: Optional[Any] = None,
        log_dir: Optional[str] = None,
        max_context_chars: Optional[int] = None,
        query: Optional[str] = None,
        verification_feedback: Optional[Dict[str, Any]] = None,
        calibration_stats: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        seed_task: Optional[str] = None,
        seed_tasks_dir: Optional[str] = None,
        random_seed_task: bool = False,
        judge_threshold: Optional[float] = None,
        difficulty: Optional[str] = None,
        test_model: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_client: LLM client for agent reasoning
            config: Hydra config for BenchmarkRunner
            working_dir: Directory containing working_task.json
            output_dir: Directory for curated output tasks
            iterations_per_task: Max ReAct iterations allowed per task
            verbose: Print agent thoughts and actions
            subtasks_min: Minimum number of subtasks per task
            subtasks_max: Maximum number of subtasks per task
            agents_min: Minimum number of agents for task generation (2-10)
            agents_max: Maximum number of agents for task generation (2-10)
            scene_data: Live scene data from SceneLoader (required)
            log_dir: Directory for log files (defaults to Hydra output or output_dir/logs)
            max_context_chars: Max context size before summarizing. Auto-detected from model if None.
            query: Optional seed query to guide task generation (e.g., "A task where agents use the radio")
            verification_feedback: Optional dict with required fixes from a failed ToM verification to incorporate
            calibration_stats: Dataset calibration stats (pass rate, target rate) for difficulty guidance
            category: Task category to generate: "cooperative", "competitive", or "mixed" (None = random)
            seed_task: Optional path to existing task JSON to use as seed instead of blank template
            seed_tasks_dir: Task pool used for per-scene seed selection
            random_seed_task: If True, each new_scene[] samples uniformly from the seed pool
        """
        self.llm = llm_client
        self.config = config
        self.working_dir = Path(working_dir)
        self.subtasks_min = subtasks_min
        self.subtasks_max = subtasks_max
        self.agents_min = agents_min
        self.agents_max = agents_max
        self.output_dir = Path(output_dir)
        self.iterations_per_task = iterations_per_task
        self.verbose = verbose
        self.scene_data = scene_data
        self.max_context_chars = max_context_chars or self._get_model_context_limit()
        self.query = query
        self.verification_feedback = verification_feedback
        self.calibration_stats = calibration_stats or {}
        self.category = category  # None means random selection
        self.seed_task = seed_task  # Path to existing task to use as seed
        self.seed_tasks_dir = Path(seed_tasks_dir) if seed_tasks_dir else None
        self.random_seed_task = random_seed_task
        self.difficulty = difficulty  # Difficulty level override for judge guidance
        self.test_model = test_model  # Override model for test_task calibration
        self.skip_steps: List[str] = []  # Pipeline steps to skip (from --remove)

        # K-level enforcement: list of allowed levels, or None = random per task.
        self._allowed_k_levels = self.calibration_stats.get("k_levels")  # e.g. [2,3] or None
        self._current_k_level: Optional[int] = None  # set per-task in run()

        # Task file paths
        self.task_file = self.working_dir / "working_task.json"
        self.template_file = self.working_dir / "template.json"

        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create submitted_tasks directory for multi-task runs
        self.submitted_tasks_dir = self.working_dir / "submitted_tasks"
        self.submitted_tasks_dir.mkdir(parents=True, exist_ok=True)

        # Create agent_trajectories directory for benchmark traces
        self.trajectories_dir = self.working_dir / "agent_trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        # Track run count for trajectory folders (reset per task)
        self._test_run_count = 0

        # Copy fresh template from source to working directory and update for agent range
        # Template shows max agents; LLM can use fewer based on agents_min/agents_max
        source_template = Path(__file__).parent / "template" / "template.json"
        if source_template.exists():
            with open(source_template) as f:
                template = json.load(f)
            # Set num_agents to max (LLM will choose within range)
            template["num_agents"] = self.agents_max
            default_actions = get_authoring_default_actions(include_find_tools=False)
            template["agent_secrets"] = {
                f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"]
                for i in range(self.agents_max)
            }
            template["agent_actions"] = {
                f"agent_{i}": default_actions.copy()
                for i in range(self.agents_max)
            }
            with open(self.template_file, 'w') as f:
                json.dump(template, f, indent=2)

        # Track state
        self.submitted_tasks: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.iteration_count = 0
        self.last_verify_passed = False  # Track if golden trajectory verified
        self.last_verified_spec_hash: Optional[str] = None
        self.last_verified_trajectory_hash: Optional[str] = None
        self.last_judge_passed = False  # Track if Judge passed
        self.last_test_passed = False  # Track if test_task was run (for calibration)
        self.last_judgment: Optional[CouncilVerdict] = None  # Last judgment result
        self.failed = False  # Track if agent called fail[]
        self.fail_reason = ""  # Reason for failure
        self._last_test_comparison: Optional[Dict[str, Any]] = None
        self.task_memories: List[str] = []  # Learnings from completed tasks
        self.consecutive_tom_failures = 0  # Track failures to suggest new_scene
        self.diversity_tracker = DiversityTracker(llm=self.llm)  # Track task patterns for diversity
        # When difficulty is set, treat it as an explicit generation target and
        # don't also inject the raw query into the judge context.
        judge_query = query if not difficulty else None
        judge_kwargs = dict(
            verbose=verbose,
            user_query=judge_query,
            diversity_tracker=self.diversity_tracker,
            difficulty=difficulty,
            skip_steps=self.skip_steps or None,
        )
        if judge_threshold is not None:
            judge_kwargs["overall_threshold"] = judge_threshold
        self.judge = Judge(**judge_kwargs)

        # Setup logging to file
        # Prefer log_dir (Hydra output), fallback to output_dir/logs
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"task_generator_{timestamp}.log"
        self._log_handle = open(self.log_file, "w")

        # Whitelisted bash commands (for safety)
        self.allowed_commands = [
            # File reading/inspection
            "ls", "cat", "head", "tail", "wc", "find",
            # JSON editing
            "jq",
            # Text search/processing
            "grep", "sed", "awk", "sort", "uniq",
            # File operations (needed for jq pattern: > tmp.json && mv tmp.json file.json)
            "mv", "cp", "rm",
            # Comparison
            "diff",
            # Validation
            "python3",
            # Output
            "echo",
            # Shell control structures
            "for", "do", "done", "while", "if", "then", "else", "fi", "in",
            # Patching
            "apply_patch",
        ]

    def _get_model_context_limit(self) -> int:
        """
        Get context limit based on model name.

        Returns 80% of model's context window in chars (~4 chars/token).
        """
        # Model context windows (in tokens)
        context_windows = {
            "gpt-5.2": 128000,
            "gpt-5": 128000,
            "gpt-5-mini": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3": 200000,
            "claude-3.5": 200000,
        }

        # Try to get model name from LLM config
        model_name = ""
        try:
            if hasattr(self.llm, 'llm_conf'):
                params = self.llm.llm_conf.get('generation_params', {})
                model_name = params.get('model', '').lower()
        except Exception:
            pass

        # Find matching context window
        for model_key, tokens in context_windows.items():
            if model_key in model_name:
                # 80% of context, ~4 chars per token
                return int(tokens * 0.8 * 4)

        # Default: assume 128k context (gpt-5 class)
        return int(128000 * 0.8 * 4)

    def run(self, num_tasks_target: int = 1) -> List[str]:
        """
        Run the agent to generate tasks.

        Args:
            num_tasks_target: Number of quality tasks to generate

        Returns:
            List of paths to submitted task files
        """
        # Calculate total max iterations from per-task budget
        self.max_iterations = self.iterations_per_task * num_tasks_target

        self._log(f"\n{'='*60}")
        self._log(f"Starting TaskGeneratorAgent")
        self._log(f"Target: {num_tasks_target} tasks")
        self._log(f"Agents: {self.agents_min} - {self.agents_max}")
        self._log(f"Iterations per task: {self.iterations_per_task}")
        self._log(f"Max iterations: {self.max_iterations} (total budget)")
        self._log(f"Output: {self.output_dir}")
        self._log(f"{'='*60}\n")

        # Scene is loaded lazily via new_scene[num_agents] - don't require it at startup
        if self.scene_data:
            self._log(f"Scene: {self.scene_data.scene_id} (episode {self.scene_data.episode_id})")
            self._log(f"  Rooms: {len(self.scene_data.rooms)}")
            self._log(f"  Furniture: {len(self.scene_data.furniture)}")
            self._log(f"  Objects: {len(self.scene_data.objects)}")
        else:
            self._log("No scene loaded yet. Agent must call new_scene[num_agents] first.")

        # Clean up working_task.json from previous runs
        if self.task_file.exists():
            self.task_file.unlink()
            self._log(f"Cleaned up previous {self.task_file}")

        # Don't create working_task.json yet - agent must call new_scene first
        # self._create_working_task_from_template() is called in _new_scene()

        # Select category (random if not specified)
        task_category = self.category or random.choice(["cooperative", "competitive", "mixed"])

        # Select k-level for this task
        if self._allowed_k_levels:
            self._current_k_level = random.choice(self._allowed_k_levels)
        else:
            # Default: random from 1, 2, 3
            self._current_k_level = random.choice([1, 2, 3])

        # Initialize conversation with action descriptions and paths injected
        replacements = {
            "{task_file}": str(self.task_file),
            "{working_dir}": str(self.working_dir),
            "{category}": task_category.upper(),
            "{diversity_section}": self._wrap_diversity_section(self._build_diversity_section()),
        }
        system_prompt = _SYSTEM_PROMPT

        # Strip sections for removed pipeline components before replacements
        if "pddl" in (self.skip_steps or []):
            system_prompt = _strip_pddl_from_guidance(system_prompt)
        if "simulation" in (self.skip_steps or []):
            system_prompt = _strip_simulation_from_guidance(system_prompt)

        for key, value in replacements.items():
            system_prompt = system_prompt.replace(key, value)

        # Inject removed-steps notice so the agent knows which pipeline stages are disabled
        if self.skip_steps:
            skip_notice = (
                "\n\n## Removed Pipeline Steps\n"
                f"The following pipeline components have been removed for this run via `--remove`: **{', '.join(self.skip_steps)}**.\n"
                "Do NOT attempt to run or rely on these components. They will be automatically skipped.\n"
            )
            if "pddl" in self.skip_steps:
                skip_notice += "- PDDL verification is disabled. You still MUST write `problem_pddl` as the canonical goal format — only the automated PDDL solvability check is skipped.\n"
            if "tom" in self.skip_steps:
                skip_notice += "- ToM level verification is disabled. Do NOT worry about tom_level computation.\n"
            if "simulation" in self.skip_steps:
                skip_notice += "- Simulation verification is disabled. Golden trajectory regeneration and simulator verification are skipped.\n"
            if "llm-council" in self.skip_steps:
                skip_notice += "- LLM council evaluation is disabled. `judge[]` will auto-pass.\n"
            if "task-evolution" in self.skip_steps:
                skip_notice += (
                    "- Task evolution is disabled. No seed tasks are provided. "
                    "Do NOT reference `sampled_tasks/`. Skip `test_task[]` and go directly to `submit_task[]`.\n"
                )
            if "test" in self.skip_steps:
                skip_notice += "- test_task is disabled. You can skip `test_task[]` and go directly to `submit_task[]`.\n"
            system_prompt += skip_notice

        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Build query section if provided
        query_section = ""
        if self.query:
            query_section = f"""
## User Requirements
{self.query}
"""

        # Build verification feedback section if retrying from failed verification
        verification_section = ""
        if self.verification_feedback:
            required_fixes = self.verification_feedback.get("required_fixes", [])
            overall_reasoning = self.verification_feedback.get("overall_reasoning", "")
            criteria = self.verification_feedback.get("criteria", {})

            verification_section = f"""
## IMPORTANT: Previous Task Failed Theory of Mind Verification

Your previous task did not pass the ToM verification. You MUST address these issues in the new task.

**Overall Assessment**: {overall_reasoning}

**Criteria Scores**:
"""
            for criterion, data in criteria.items():
                score = data.get("score", 0)
                reasoning = data.get("reasoning", "")
                status = "✓" if score >= 0.5 else "✗"
                verification_section += f"- {criterion}: {score:.2f} {status} - {reasoning}\n"

            verification_section += "\n**Required Fixes**:\n"
            for i, fix in enumerate(required_fixes, 1):
                verification_section += f"{i}. {fix}\n"

            verification_section += "\nCreate a NEW task that specifically addresses these issues.\n"

        # Build calibration/difficulty guidance section
        calibration_section = ""
        if self.difficulty:
            # When difficulty is explicitly set, use it directly and skip
            # target-model calibration stats to avoid conflicting guidance.
            difficulty_guidance = {
                "easy": (
                    "## Difficulty: EASY\n"
                    "Generate SIMPLE tasks that weaker models can solve:\n"
                    "- Use 0-1 mechanics (prefer limited_bandwidth or room_restriction)\n"
                    "- Avoid inverse_state on easy tasks unless the secret states it plainly\n"
                    "- 2-3 agents with clear roles\n"
                    "- 2-3 subtasks maximum\n"
                    "- Secrets MUST explain any active mechanic in plain language\n"
                    "- All effects should be observable (no remote effects in unseen rooms)\n"
                    "- tom_level 1 only; K=0 tasks are invalid\n"
                    "- limited_bandwidth with generous limits (4-5 messages) works well at this level\n"
                ),
                "medium": (
                    "## Difficulty: MEDIUM\n"
                    "Generate moderately complex tasks:\n"
                    "- 3-4 agents with distinct physical roles\n"
                    "- Use restricted_communication to create directed messaging (not all-to-all)\n"
                    "- Use room_restriction so agents have limited access to rooms\n"
                    "- limited_bandwidth: 2 messages per agent\n"
                    "- Keep the physical core small: 2-4 subtasks and usually one non-trivial K() chain\n"
                    "- Prefer one grounded final-state fact reused by both the physical goal and the K() goal\n"
                    "- Do NOT tell agents exactly what to communicate in secrets\n"
                    "- tom_level 2-3 (required K-level is set separately)\n"
                ),
                "hard": (
                    "## Difficulty: HARD — Target: GPT-5.2 FAILS this task\n"
                    "Generate tasks that top-tier models CANNOT solve. "
                    "test_task[] will REJECT any task GPT-5.2 can solve.\n\n"
                    "### Required properties:\n"
                    "- 3-4 agents, each with a UNIQUE physical role + information\n"
                    "- Use restricted_communication to force RELAY chains (A→B→C, not A→C)\n"
                    "- Use room_restriction to make each agent physically necessary\n"
                    "- limited_bandwidth: 2 messages per agent for K=2, 3 for K=3\n"
                    "- Keep the physical core compact: 2-4 physical conjuncts plus one strict non-trivial K() chain\n"
                    "- Keep mechanics to room_restriction + restricted_communication, and add limited_bandwidth only if the relay still works\n\n"
                    "### How to create K=2/3 tasks that GPT-5.2 CANNOT solve:\n"
                    "- K=2 pattern: Agent A observes fact X. A can only message B.\n"
                    "  B can relay to C, and C cannot directly observe X. Goal: one physical conjunct over X plus (K C (K B X)).\n"
                    "- K=3 pattern: Add a 4th hop, but keep the physical goal itself simple.\n"
                    "- restricted_communication creates the relay chain (agents can't skip hops)\n"
                    "- room_restriction makes the outer K-agents unable to directly observe the fact\n"
                    "- Secrets should say WHAT each agent needs to know, NOT how to communicate it\n"
                    "- Do NOT prescribe the relay chain in secrets — agents must figure it out\n\n"
                    "### Anti-patterns (these break tasks or make them easy):\n"
                    "- 1 message per agent with K≥2 goals (UNSOLVABLE — not enough bandwidth)\n"
                    "- Secrets that say 'tell agent_X: ...' or 'relay to agent_X'\n"
                    "- Plans where 1-2 agents do all meaningful work while others only relay or mirror generic actions (agent_necessity fails)\n"
                    "- K() goals where the outer knower can just walk over and see the fact directly\n"
                    "- Secrets that claim access restrictions not backed by mechanic_bindings\n"
                ),
            }
            calibration_section = difficulty_guidance.get(self.difficulty, "")
        elif self.calibration_stats:
            model = self.calibration_stats.get("model", "unknown")
            target_rate = self.calibration_stats.get("target_rate", 0.10)
            current_rate = self.calibration_stats.get("rate")
            total = self.calibration_stats.get("total", 0)
            passed = self.calibration_stats.get("passed", 0)
            failed = self.calibration_stats.get("failed", 0)
            untested = self.calibration_stats.get("untested", 0)

            calibration_section = f"""
## Dataset Calibration
Target: {target_rate:.0%} of tasks should be passable by {model}

"""
            if current_rate is not None:
                calibration_section += f"""Current dataset statistics for {model}:
- Tested tasks: {total}
- Pass rate: {current_rate:.1%} ({passed} passed, {failed} failed)
- Untested: {untested}

"""
                tolerance = 0.05
                if current_rate > target_rate + tolerance:
                    calibration_section += f"**Guidance**: Current pass rate ({current_rate:.1%}) is ABOVE target ({target_rate:.0%}). Generate HARDER tasks with more complex coordination requirements, deeper dependency chains, or more challenging Theory of Mind requirements.\n"
                elif current_rate < target_rate - tolerance:
                    calibration_section += f"**Guidance**: Current pass rate ({current_rate:.1%}) is BELOW target ({target_rate:.0%}). Generate EASIER tasks with clearer paths to success, simpler coordination, or more direct hints in agent secrets.\n"
                else:
                    calibration_section += f"**Guidance**: Current pass rate is near target. Continue generating varied difficulty tasks.\n"
            else:
                calibration_section += f"No calibration data yet for {model}. Generate tasks of varied difficulty.\n"

        # Build k-level directive for this task.
        tom_calibration_section = ""
        target_k = self._current_k_level
        if target_k is not None:
            tom_calibration_section = (
                f"\n## Required K-Level: {target_k}\n"
                f"This task MUST be Theory-of-Mind level {target_k}.\n"
                "K=0 tasks are invalid and will be rejected.\n"
                f"Design epistemic goals so that `judge[]`'s strict PDDL verification computes tom_level = {target_k}.\n"
                f"submit_task[] will REJECT the task if the computed tom_level is not {target_k}.\n"
            )

        # Build seed task section if using a seed
        seed_section = ""
        if self.seed_task or self.seed_tasks_dir:
            target_model = self.calibration_stats.get("model") or self.test_model or "unknown"
            selection_mode = "uniform random" if self.random_seed_task else "target-model calibrated"
            seed_pool = self.seed_task or (str(self.seed_tasks_dir) if self.seed_tasks_dir else "seed pool")
            seed_section = f"""
## Seed Task
A seed task from `{seed_pool}` will be loaded into working_task.json as your starting point.
`new_scene[N]` re-samples a seed each time so you keep building from the task pool instead of starting blank.
Selection mode: {selection_mode}. Target model: {target_model}.
Use the loaded seed as a foundation and modify it based on the query/requirements above.
After calling `new_scene[N]`, view it with: `bash[cat {self.task_file}]`
The seed task is intentionally structure-only: natural-language fields are scrubbed before loading.
Always rewrite `title`, `task`, `agent_secrets`, and `team_secrets` from scratch for the current scene and current runtime semantics.
"""

        # Build extra sections string and persist for context resets.
        # The k-level section is rebuilt per-task, so store the static parts separately.
        self._static_extra_sections = (
            query_section
            + seed_section
            + verification_section
            + calibration_section
        )
        extra_sections = self._static_extra_sections + tom_calibration_section
        self._extra_sections = extra_sections

        # Initial user message - use template from prompts.py
        user_msg = _USER_PROMPT_TEMPLATE.format(
            num_tasks=num_tasks_target,
            extra_sections=extra_sections,
            agents_min=self.agents_min,
            agents_max=self.agents_max,
            subtasks_min=self.subtasks_min,
            subtasks_max=self.subtasks_max,
        )

        self.messages.append({"role": "user", "content": user_msg})

        # Save initial context window
        self._save_context_window()

        # Main ReAct loop
        llm_error_streak = 0
        max_llm_error_streak = 8
        while len(self.submitted_tasks) < num_tasks_target and not self.failed:
            self.iteration_count += 1

            if self.iteration_count > self.max_iterations:
                self._log(f"\nReached max iterations ({self.max_iterations})")
                break

            self._log(f"\n{'='*40}")
            self._log(f"Iteration {self.iteration_count}/{self.max_iterations} | Submitted: {len(self.submitted_tasks)}/{num_tasks_target}")
            self._log(f"{'='*40}")

            # Check if context needs summarization
            self._maybe_summarize_context()

            # Get LLM response
            try:
                response = self._get_llm_response()
                llm_error_streak = 0
            except Exception as e:
                llm_error_streak += 1
                # Do not burn iteration budget on transient API/network failures.
                self.iteration_count = max(0, self.iteration_count - 1)
                backoff_s = min(30, 2 ** min(llm_error_streak, 5))
                self._log(f"LLM error: {e}")
                self._log(f"Transient LLM failure streak={llm_error_streak}; backing off {backoff_s}s and retrying without consuming iteration.")
                if llm_error_streak >= max_llm_error_streak:
                    self.failed = True
                    self.fail_reason = (
                        "Repeated LLM API/network failures "
                        f"({llm_error_streak} consecutive errors)."
                    )
                    self._log(
                        "Stopping generation due to persistent LLM connectivity failures. "
                        "Please verify API/network availability and retry."
                    )
                    break
                self.messages.append({
                    "role": "user",
                    "content": f"Error getting LLM response: {e}. Please try again."
                })
                time.sleep(backoff_s)
                continue

            # Parse all actions from response
            actions = self._parse_all_actions(response)

            if not actions:
                self._log("No valid action found in response")
                self.messages.append({
                    "role": "assistant",
                    "content": response
                })
                self.messages.append({
                    "role": "user",
                    "content": "I couldn't parse a valid action. Please respond with:\nThought: [reasoning]\nAction: tool_name[args]"
                })
                continue

            # Execute first action only (standard ReAct pattern)
            tool, args = actions[0]
            observation = self._execute_action(tool, args)
            # Keep only content up to first action for clean context
            response_for_history = self._truncate_to_first_action(response)

            # Truncate heredocs in PREVIOUS assistant messages to save context
            self._truncate_old_heredocs()

            # Add current turn to conversation
            self.messages.append({"role": "assistant", "content": response_for_history})
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})

            self._log(f"Observation: {observation}", truncate_terminal=300)

            # Save full context window after each step
            self._save_context_window()

        self._log(f"\n{'='*60}")
        if self.failed:
            self._log(f"Agent FAILED: {self.fail_reason}")
        else:
            self._log(f"Agent finished. Submitted {len(self.submitted_tasks)} tasks:")
            for task_path in self.submitted_tasks:
                self._log(f"  - {task_path}")
        self._log(f"{'='*60}\n")

        # Save working_task.json to log directory for debugging
        self._save_working_task_to_logs()

        return self.submitted_tasks

    def _save_working_task_to_logs(self) -> None:
        """Save a copy of working_task.json to log directory for debugging.

        Re-injects agent_spawns from scene_data since LLM may have removed it during edits.
        """
        if self.task_file.exists():
            try:
                # Load current task
                with open(self.task_file) as f:
                    task_data = json.load(f)

                # Re-inject agent_spawns from scene_data (LLM may have removed it)
                if self.scene_data and self.scene_data.agent_spawns:
                    task_data["agent_spawns"] = self.scene_data.agent_spawns

                # Generate canonical task_id if still a placeholder
                if task_data.get("task_id") == "REPLACE_WITH_UNIQUE_ID":
                    try:
                        import hashlib, re
                        title_slug = re.sub(r"[^a-z0-9]+", "-", str(task_data.get("title", "untitled")).lower()).strip("-")[:40] or "untitled"
                        category = str(task_data.get("category", "cooperative")).lower()
                        scene_id = str(task_data.get("scene_id", "scene"))
                        episode_id = str(task_data.get("episode_id", "episode"))
                        problem_hash = hashlib.sha256(
                            (task_data.get("problem_pddl", "") + "|" + category + "|" + scene_id + "|" + episode_id).encode("utf-8")
                        ).hexdigest()[:8]
                        task_data["task_id"] = f"emtom-{scene_id}-{episode_id}-{category}-{title_slug}-{problem_hash}"
                    except Exception:
                        pass

                # Compute and inject tom_level from PDDL if not already set
                if task_data.get("problem_pddl") and "tom_level" not in task_data:
                    try:
                        from emtom.cli.submit_task import _compute_tom_metadata
                        tom_meta = _compute_tom_metadata(task_data, scene_data=None)
                        task_data["tom_level"] = tom_meta["tom_level"]
                        if "tom_reasoning" in tom_meta:
                            task_data["tom_reasoning"] = tom_meta["tom_reasoning"]
                    except Exception:
                        pass

                # Save to log directory
                dest = self.log_dir / "working_task_final.json"
                with open(dest, "w") as f:
                    json.dump(task_data, f, indent=2)
                self._log(f"Saved working task to: {dest}")
            except Exception as e:
                self._log(f"Warning: Could not save working task: {e}")

    def _save_context_window(self) -> None:
        """Save the entire context window to prompt.txt for debugging."""
        prompt_file = self.log_dir / "prompt.txt"
        try:
            with open(prompt_file, "w") as f:
                for i, msg in enumerate(self.messages):
                    role = msg["role"].upper()
                    content = msg["content"]
                    f.write(f"=== {role} [{i}] ===\n\n")
                    f.write(content)
                    f.write("\n\n")
        except Exception as e:
            self._log(f"Warning: Could not save context window: {e}")

    def _create_working_task_from_template(self, num_agents: Optional[int] = None) -> int:
        """Create working_task.json from template or seed task.

        If self.seed_task or self.seed_tasks_dir is set, loads a seed task instead
        of the blank template.
        Scene fields (scene_id, episode_id, agent_spawns) are always updated from the current scene.

        Args:
            num_agents: Number of agents to use. If None, uses agents_max.

        Returns:
            The number of agents used.
        """
        if num_agents is None:
            num_agents = self.agents_max

        # Load seed task or blank template
        seed_path = self._resolve_seed_task_path()
        if seed_path is not None:
            with open(seed_path) as f:
                task = json.load(f)
            self._log(f"Loaded seed task from {seed_path}")
            # Use seed task's num_agents if not overridden
            if num_agents == self.agents_max and "num_agents" in task:
                num_agents = task["num_agents"]
            task = sanitize_task_for_seeding(task, num_agents=num_agents)
        else:
            with open(self.template_file) as f:
                task = json.load(f)

        # Auto-populate scene fields (always from current scene, even for seed tasks)
        if self.scene_data:
            task["scene_id"] = self.scene_data.scene_id
            task["episode_id"] = self.scene_data.episode_id
            # Include agent spawn positions (calculated once, reused for all runs)
            if self.scene_data.agent_spawns:
                task["agent_spawns"] = self.scene_data.agent_spawns
            task["problem_pddl"] = build_scene_bootstrap_problem_pddl(
                self.scene_data,
                num_agents,
                problem_name=f"scene_{self.scene_data.scene_id}",
            )

        task["num_agents"] = num_agents

        # Only generate placeholder agent fields when not using a seed task
        if seed_path is None:
            # Include Find* tools so agents can discover objects at runtime instead of hardcoded IDs
            default_actions = get_authoring_default_actions(include_find_tools=True)
            task["agent_secrets"] = {
                f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"]
                for i in range(num_agents)
            }
            task["agent_actions"] = {
                f"agent_{i}": default_actions.copy()
                for i in range(num_agents)
            }

        # Clear task_id — canonical ID is generated at submit time.
        task["task_id"] = "REPLACE_WITH_UNIQUE_ID"

        # Write to working_task.json
        with open(self.task_file, 'w') as f:
            json.dump(task, f, indent=2)

        self._log(
            f"Created {self.task_file} with {num_agents} agents"
            f"{' (from seed)' if seed_path is not None else ''}"
        )
        return num_agents

    def _resolve_seed_task_path(self) -> Optional[Path]:
        """Return an explicit seed task or a selected compatible seed from the pool."""
        if self.seed_task:
            return Path(self.seed_task)
        if self.seed_tasks_dir is None:
            return None

        if not self.seed_tasks_dir.exists():
            self._log(
                f"Seed selection requested but seed dir does not exist: "
                f"{self.seed_tasks_dir}"
            )
            return None

        target_model = self.calibration_stats.get("model") or self.test_model or "gpt-5.2"
        selection_config = SeedSelectionConfig(
            tasks_dir=self.seed_tasks_dir,
            target_model=target_model,
            target_pass_rate=self.calibration_stats.get("target_rate", 0.20),
            current_pass_rate=self.calibration_stats.get("rate"),
            category=self.category,
            tom_level=self._current_k_level,
        )
        if self.random_seed_task:
            candidates = build_seed_candidates(selection_config)
        else:
            candidates = select_seed_tasks(selection_config, count=1)
        if not candidates:
            self._log(
                f"Seed selection requested but no compatible task JSONs found in "
                f"{self.seed_tasks_dir}"
            )
            return None

        if self.random_seed_task:
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]
        chosen_path = chosen.path
        chosen_data = chosen.task_data
        self._log(
            "Selected seed task "
            f"{chosen_path} (category={chosen_data.get('category')}, "
            f"tom_level={chosen_data.get('tom_level', 'unknown')}, "
            f"passed_{target_model}={chosen.passed_target_model}, "
            f"progress={chosen.progress if chosen.progress is not None else 'unknown'})"
        )
        return chosen_path

    def _truncate_old_heredocs(self) -> None:
        """Truncate large heredoc content in PREVIOUS assistant messages.

        When the LLM writes a file via heredoc, the entire file content
        appears in the response. For older messages, we truncate to save context.
        The most recent messages are kept full.
        """
        # Pattern: cat > file << 'EOF' ... EOF (or << "EOF" or << EOF)
        heredoc_pattern = re.compile(
            r"(cat\s*>\s*[^\s]+\s*<<\s*['\"]?EOF['\"]?\n)"  # Opening
            r"(.*?)"  # Content (non-greedy)
            r"(\nEOF)",  # Closing
            re.DOTALL
        )

        def truncate_match(match):
            opening = match.group(1)
            content = match.group(2)
            closing = match.group(3)

            # Count lines and chars
            lines = content.count('\n') + 1
            chars = len(content)

            if chars > 500:  # Only truncate if content is large
                # Show first 200 chars and summary
                preview = content[:200].rsplit('\n', 1)[0]  # Clean line break
                return f"{opening}{preview}\n... [truncated: {lines} lines, {chars} chars]{closing}"
            return match.group(0)

        # Truncate all assistant messages except the last 2 (most recent turn)
        for i, msg in enumerate(self.messages[:-2] if len(self.messages) > 2 else []):
            if msg["role"] == "assistant":
                original = msg["content"]
                truncated = heredoc_pattern.sub(truncate_match, original)
                if truncated != original:
                    self.messages[i]["content"] = truncated

    def _get_llm_response(self) -> str:
        """Get response from LLM."""
        prompt = self._format_messages_for_llm()
        # Stop on "Assigned!" - LLM outputs this after each action
        response = self.llm.generate(prompt, stop="Assigned!")
        self._log(f"Agent: {response}", truncate_terminal=300)
        return response

    def _truncate_to_first_action(self, content: str) -> str:
        """Truncate content after the first Action: tool[args] to keep context clean.

        Appends 'Assigned!' since the LLM stops on that token (so it's not in the response).
        """
        action_match = re.search(r'Action:\s*(\w+)\[', content)
        if not action_match:
            return content

        start_idx = action_match.end()
        bracket_content = self._extract_bracket_content(content, start_idx)
        if bracket_content is None:
            return content

        end_idx = start_idx + len(bracket_content) + 1
        # Append "Assigned!" - it was the stop word so not included in response
        return content[:end_idx].rstrip() + "\nAssigned!"

    def _format_messages_for_llm(self) -> str:
        """Format message history for the LLM."""
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System:\n{content}")
            elif role == "user":
                parts.append(f"User:\n{content}")
            elif role == "assistant":
                parts.append(f"Assistant:\n{content}")
        return "\n\n".join(parts)

    def _estimate_context_size(self) -> int:
        """Estimate current context size in characters."""
        return sum(len(msg["content"]) for msg in self.messages)

    def _maybe_summarize_context(self) -> None:
        """
        Summarize older messages if context exceeds threshold.

        Uses the LLM to summarize, preserving intent and important observations.

        Keeps:
        - messages[0]: System prompt (exactly as-is)
        - messages[1]: Initial user message with scene data
        - Last N messages (recent conversation)

        Summarizes middle messages via LLM into a single summary.
        """
        context_size = self._estimate_context_size()
        if context_size < self.max_context_chars:
            return

        self._log(f"Context size ({context_size} chars) exceeds threshold ({self.max_context_chars}). Summarizing...")

        # Keep system prompt (0), initial user msg (1), and last 10 messages
        keep_recent = 10
        if len(self.messages) <= 2 + keep_recent:
            # Not enough messages to summarize
            return

        # Messages to summarize: from index 2 to -(keep_recent)
        to_summarize = self.messages[2:-keep_recent]
        if not to_summarize:
            return

        # Format messages for summarization
        conversation_text = []
        for msg in to_summarize:
            role = msg["role"].capitalize()
            content = msg["content"]
            # Truncate very long observations for the summary request
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_text.append(f"{role}: {content}")

        summarize_prompt = f"""Summarize this conversation history for context continuity.
Preserve:
1. Your current intent/goal and progress toward it
2. Important observations (errors, successful actions, state changes)
3. Key decisions made and why
4. Current state of working_task.json (what fields have been set)

Be concise but complete. This summary replaces the conversation history.

CONVERSATION TO SUMMARIZE:
{chr(10).join(conversation_text)}

SUMMARY:"""

        # Use LLM to generate summary
        try:
            summary = self.llm.generate(summarize_prompt)
            summary = f"[CONTEXT SUMMARY - {len(to_summarize)} messages summarized by LLM]\n\n{summary}\n\n[End of summary - recent conversation follows]"
        except Exception as e:
            self._log(f"LLM summarization failed: {e}, using fallback")
            # Fallback to simple list of actions
            actions = []
            for msg in to_summarize:
                if msg["role"] == "assistant":
                    match = re.search(r'Action:\s*(\w+)\[', msg["content"])
                    if match:
                        actions.append(match.group(1))
            summary = f"[CONTEXT SUMMARY - {len(to_summarize)} messages]\nActions taken: {', '.join(actions[-30:])}\n[End of summary]"

        # Rebuild messages: system + initial + summary + recent
        self.messages = (
            self.messages[:2] +  # System + initial user
            [{"role": "user", "content": summary}] +  # Summary
            self.messages[-keep_recent:]  # Recent messages
        )

        new_size = self._estimate_context_size()
        self._log(f"Context reduced from {context_size} to {new_size} chars")

    def _parse_action(self, response: str) -> Optional[tuple]:
        """
        Parse first action from LLM response using bracket-matching.

        Expected format:
        Thought: [reasoning]
        Action: tool_name[args]

        Returns:
            Tuple of (tool_name, args) or None if parsing fails
        """
        actions = self._parse_all_actions(response)
        return actions[0] if actions else None

    def _parse_all_actions(self, response: str) -> List[tuple]:
        """
        Parse ALL actions from LLM response.

        Supports multiple bash actions:
        Action: bash[cmd1]
        Action: bash[cmd2]

        Returns:
            List of (tool_name, args) tuples
        """
        actions = []
        pos = 0

        while True:
            # Find next "Action:" followed by tool name and opening bracket
            action_match = re.search(r'Action:\s*(\w+)\[', response[pos:])
            if not action_match:
                break

            tool_name = action_match.group(1)
            start_idx = pos + action_match.end()  # Position right after the opening [

            # Use bracket matching to find the closing ]
            args = self._extract_bracket_content(response, start_idx)
            if args is None:
                break

            actions.append((tool_name, args.strip()))

            # Move position past this action
            # Find where the bracket content ended
            end_idx = start_idx + len(args) + 1  # +1 for closing bracket
            pos = end_idx

        return actions

    def _extract_bracket_content(self, text: str, start: int) -> Optional[str]:
        """
        Extract content between brackets using proper bracket matching.

        Args:
            text: The full text
            start: Index right after the opening bracket

        Returns:
            The content between brackets, or None if no matching bracket found
        """
        depth = 1  # We've already seen one [
        in_single_quote = False
        in_double_quote = False
        escape_next = False
        i = start

        while i < len(text) and depth > 0:
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            # Track quote state (only toggle if not in the other quote type)
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            # Count brackets only when not inside quotes
            elif not in_single_quote and not in_double_quote:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1

            i += 1

        if depth == 0:
            # i is now one past the closing bracket
            return text[start:i - 1]

        return None

    def _extract_paths_from_command(self, command: str) -> List[str]:
        """
        Extract file paths from a shell command.

        Returns list of paths that appear as arguments to file-access commands.
        Handles quoted paths and paths with spaces.
        """
        paths = []

        # Commands that take file/directory paths as arguments
        path_commands = {"cat", "head", "tail", "ls", "wc", "find"}

        # Flags that take an argument (not a path)
        flags_with_args = {"-n", "-c", "-name", "-type", "-maxdepth", "-mindepth"}

        # Split command respecting quotes
        tokens = []
        current_token = []
        in_single_quote = False
        in_double_quote = False

        for char in command:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char.isspace() and not in_single_quote and not in_double_quote:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                continue
            current_token.append(char)

        if current_token:
            tokens.append("".join(current_token))

        # Clean quotes from tokens
        tokens = [t.strip("'\"") for t in tokens]

        # Find paths after path commands
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if this is a path command
            if token in path_commands:
                # Collect following tokens that look like paths (not flags)
                j = i + 1
                while j < len(tokens):
                    arg = tokens[j]
                    # Skip flags - if flag takes an argument, skip that too
                    if arg.startswith("-"):
                        if arg in flags_with_args and j + 1 < len(tokens):
                            j += 2  # Skip flag and its argument
                        else:
                            j += 1  # Skip flag only
                        continue
                    # Skip heredoc markers
                    if arg.startswith("<"):
                        j += 1
                        continue
                    # Skip shell operators
                    if arg in ("|", ">", ">>", "&&", "||", ";"):
                        break
                    # Skip jq patterns (start with . but not ./ or .. or just .)
                    if arg.startswith(".") and arg != "." and not arg.startswith("./") and not arg.startswith(".."):
                        break
                    # This looks like a path
                    paths.append(arg)
                    j += 1
                i = j
            else:
                i += 1

        # Also check for redirection targets (> or >>)
        for i, token in enumerate(tokens):
            if token in (">", ">>") and i + 1 < len(tokens):
                paths.append(tokens[i + 1])

        return paths

    def _validate_path_in_working_dir(self, path: str, allowed_paths: List[str]) -> tuple[bool, str]:
        """
        Validate that a path is within the allowed directories.

        Args:
            path: The path to validate
            allowed_paths: List of allowed directory paths

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            # Resolve the path to an absolute path
            # If relative, it's relative to working_dir (since subprocess runs there)
            if not os.path.isabs(path):
                # Resolve relative to the first allowed path (working_dir)
                abs_path = os.path.join(allowed_paths[0], path)
            else:
                abs_path = path

            # Resolve any symlinks and normalize
            real_path = os.path.realpath(abs_path)

            # Check if the path is within any allowed directory
            for allowed in allowed_paths:
                allowed_real = os.path.realpath(allowed)
                # Use os.path.commonpath to check if path is under allowed
                try:
                    common = os.path.commonpath([real_path, allowed_real])
                    if common == allowed_real:
                        return True, ""
                except ValueError:
                    # Different drives on Windows, or other path issues
                    continue

            return False, f"Access denied: '{path}' is outside allowed directories. Only access within working directory is permitted."
        except Exception as e:
            return False, f"Path validation error for '{path}': {e}"

    def _split_by_operators(self, command: str) -> List[str]:
        """
        Split command by shell operators (&&, ||, ;, |) while respecting quotes
        and backslash escapes.

        Returns list of sub-commands.
        """
        sub_commands = []
        current = []
        in_single_quote = False
        in_double_quote = False
        i = 0

        while i < len(command):
            char = command[i]

            # Handle backslash escapes (skip next character)
            # In double quotes, \", \\, \$ etc. are escapes
            # Outside quotes, backslash escapes the next character
            if char == "\\" and not in_single_quote and i + 1 < len(command):
                current.append(char)
                current.append(command[i + 1])
                i += 2
                continue

            # Track quote state
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                current.append(char)
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                current.append(char)
            # Check for operators outside quotes
            elif not in_single_quote and not in_double_quote:
                # Check for && or ||
                if i + 1 < len(command) and command[i:i+2] in ("&&", "||"):
                    sub_commands.append("".join(current))
                    current = []
                    i += 2
                    continue
                # Check for ; or |
                elif char in ";|":
                    sub_commands.append("".join(current))
                    current = []
                else:
                    current.append(char)
            else:
                current.append(char)

            i += 1

        # Add the last command
        if current:
            sub_commands.append("".join(current))

        return sub_commands

    def _execute_action(self, tool: str, args: str) -> str:
        """Execute the specified tool."""
        self._log(f"Executing: {tool}[{args[:100]}...]")

        if tool == "bash":
            # Reset verification on any bash edit to task file.
            # Judge evaluates task *design* quality (not trajectory), so
            # don't reset judge pass — the agent can still submit after
            # adding/fixing a golden trajectory without re-judging.
            if "working_task.json" in args and (">" in args or "cat" in args):
                self.last_verify_passed = False
                self.last_verified_spec_hash = None
                self.last_verified_trajectory_hash = None
                self.last_test_passed = False
                self._last_test_comparison = None
            return self._bash(args)
        elif tool == "test_task":
            return self._test_task()
        elif tool == "judge":
            return self._judge()
        elif tool == "submit_task":
            return self._submit_task()
        elif tool == "new_scene":
            return self._new_scene(args)
        elif tool == "fail":
            return self._fail(args)
        else:
            return f"Unknown tool: {tool}. Available: bash, test_task, judge, submit_task, new_scene, fail"

    def _bash(self, command: str) -> str:
        """
        Execute a shell command with safety limits.

        Allows file exploration and editing within allowed directories only.
        Allows command chaining (&&, ||, ;, |) but validates each sub-command.
        """
        # Fix heredocs where the LLM emitted literal \n instead of real
        # newlines.  Detect: heredoc delimiter immediately followed by \n
        # (two chars) rather than an actual newline, meaning the entire
        # heredoc body is on one line.  Convert literal \n → real newline
        # only in the heredoc body so the shell can parse it.
        heredoc_match = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", command)
        if heredoc_match:
            delim = heredoc_match.group(1)
            after_delim = command[heredoc_match.end():]
            # If the heredoc body starts with literal \n (not a real newline),
            # the LLM serialized the whole thing on one line.
            if after_delim.startswith("\\n") or (
                not after_delim.startswith("\n") and "\\n" in after_delim
            ):
                # Split into command line + heredoc body + closing delimiter.
                # Replace literal \n with real newlines in the body.
                pre = command[:heredoc_match.end()]
                body = after_delim
                # The closing delimiter may be followed by more commands
                body = body.replace("\\n", "\n")
                command = pre + body

        # Block command substitution (dangerous - allows arbitrary code execution)
        dangerous_patterns = ["`", "$(", "${"]
        for pattern in dangerous_patterns:
            if pattern in command:
                return f"Command substitution not allowed: '{pattern}' detected."

        # Allowed directories (temp working dir only - all working files are there)
        allowed_paths = [str(self.working_dir)]

        # Detect heredocs with any delimiter (EOF, PY, END, etc.)
        # Match patterns like: <<EOF, <<'EOF', <<"EOF", <<-EOF, <<-'PY', etc.
        heredoc_match = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", command)
        is_heredoc = heredoc_match is not None

        # Split command by chain operators to validate each part
        # For heredocs, only validate the command part before <<
        if is_heredoc:
            command_part = command.split("<<")[0]
            heredoc_part = "<<" + command.split("<<", 1)[1]
        else:
            command_part = command
            heredoc_part = ""

        # Split by chain operators (but not inside quotes)
        sub_commands = self._split_by_operators(command_part)

        # Validate each sub-command starts with an allowed command
        for sub_cmd in sub_commands:
            sub_cmd = sub_cmd.strip()
            if not sub_cmd:
                continue
            first_word = sub_cmd.split()[0] if sub_cmd.split() else ""
            if first_word not in self.allowed_commands:
                return f"Command not allowed: '{first_word}'. Allowed: {', '.join(self.allowed_commands)}"

        # SECURITY: Validate ALL file paths are within allowed directories
        # Extract paths from the command part (not heredoc content)
        extracted_paths = self._extract_paths_from_command(command_part)

        for path in extracted_paths:
            is_valid, error_msg = self._validate_path_in_working_dir(path, allowed_paths)
            if not is_valid:
                return error_msg

        # For heredoc writes, also validate the target path explicitly
        if is_heredoc and ">" in command:
            path_match = re.search(r'cat\s*>\s*([^\s<]+)', command)
            if path_match:
                target_path = path_match.group(1).strip()
                is_valid, error_msg = self._validate_path_in_working_dir(target_path, allowed_paths)
                if not is_valid:
                    return error_msg

        try:
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",  # Use bash (not /bin/sh/dash) for heredoc support
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.working_dir)  # Run from working directory for safety
            )
            output = result.stdout + result.stderr

            # LLMs often emit python -c payloads with literal "\n" escapes:
            # python3 -c "import json\nx=1". Python treats "\n" as backslash+n,
            # which triggers SyntaxError. Convert those literal escapes to real
            # newlines and retry once.
            if (
                result.returncode != 0
                and "SyntaxError: unexpected character after line continuation character" in output
            ):
                repaired_command = self._repair_python_c_newlines(command)
                if repaired_command != command:
                    self._log(
                        "Retrying bash command after converting literal \\n in python -c payload."
                    )
                    retry = subprocess.run(
                        repaired_command,
                        shell=True,
                        executable="/bin/bash",
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(self.working_dir),
                    )
                    retry_output = retry.stdout + retry.stderr
                    # Prefer retried output if it succeeded or removed the same syntax error.
                    if retry.returncode == 0 or (
                        "SyntaxError: unexpected character after line continuation character"
                        not in retry_output
                    ):
                        result = retry
                        output = retry_output

            if not output.strip():
                output = "(no output)"
            # Log full output to file, but truncate for LLM context
            if len(output) > 20000:
                self._log(f"[Full bash output ({len(output)} chars)]:\n{output}")
                output = output[:20000] + f"\n... (truncated, full output in log file)"

            return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Command failed: {e}"

    def _repair_python_c_newlines(self, command: str) -> str:
        """Convert literal '\\n' escapes inside python -c strings to real newlines."""
        pattern = re.compile(
            r"(\bpython(?:3)?\s+-c\s+)(['\"])((?:\\.|(?!\2).)*)\2",
            flags=re.DOTALL,
        )

        def _repl(match: re.Match[str]) -> str:
            prefix = match.group(1)
            quote = match.group(2)
            code = match.group(3)
            # Only unescape single-backslash \n (keep \\n literals intact).
            repaired = re.sub(r"(?<!\\)\\n", "\n", code)
            if repaired == code:
                return match.group(0)
            return f"{prefix}{quote}{repaired}{quote}"

        return pattern.sub(_repl, command)

    def _test_task(self) -> str:
        """
        Validate and optionally run benchmark with current working_task.json.

        First validates the task structure, then attempts to run benchmark
        if the environment is available.
        """
        _skip = self.skip_steps or []
        if "test" in _skip or "task-evolution" in _skip:
            self.last_test_passed = True
            _reason = "--remove task-evolution" if "task-evolution" in _skip else "--remove test"
            return json.dumps({"gate": "PASSED", "skipped": True, "reason": _reason})

        # Check task file exists
        if not self.task_file.exists():
            return "Error: working_task.json does not exist. Create it first with bash."

        # Load task
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in working_task.json: {e}"

        self._log(f"Testing task: {task_data.get('title', 'Untitled')}")

        # First, validate task structure
        validation_result = self._validate_task_structure(task_data)
        if "error" in validation_result:
            return json.dumps(validation_result, indent=2)

        # Try to run benchmark if environment is available
        try:
            results = self._run_benchmark(task_data)
            # Check if benchmark returned an error (import error, etc.)
            if results.get("error"):
                # Merge validation result with benchmark error
                validation_result["benchmark_error"] = results["error"]
                validation_result["summary"] = f"Task structure valid. Benchmark skipped: {results['error']}"
                return json.dumps(validation_result, indent=2)

            # Save calibration results to task JSON for dataset tracking (needs action_history)
            self._save_calibration_result(task_data, results)

            for run_mode in ("standard", "baseline"):
                run_result = results.get(run_mode, {})
                action_history = run_result.get("action_history", [])
                if action_history:
                    self._log(f"\n=== {run_mode.upper()} Action History ===")
                    for entry in action_history:
                        self._log(
                            f"  Turn {entry.get('turn', '?')}: "
                            f"{entry.get('agent', '?')} -> {entry.get('action', '?')}"
                        )
                run_result.pop("action_history", None)

            # Benchmark ran successfully - merge results with validation
            validation_result.update(results)
            comparison = results.get("comparison", {})
            self._last_test_comparison = comparison
            self.last_test_passed = comparison.get("gate_passed", False)
            validation_result["gate"] = "PASSED" if self.last_test_passed else "REJECTED"
            validation_result["gate_reason"] = " ".join(comparison.get("reasons", []))

            return json.dumps(validation_result, indent=2)
        except Exception as e:
            # If benchmark fails due to environment issues, return validation result
            # with a note that benchmark couldn't run
            validation_result["benchmark_error"] = str(e)
            validation_result["summary"] = f"Task structure valid. Benchmark skipped: {e}"
            return json.dumps(validation_result, indent=2)

    def _build_trajectory(self, action_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform flat action_history into grouped trajectory format.

        Returns list of turn entries, each containing:
        - turn: turn number
        - agents: dict mapping agent_id -> {action, observation}
        - subtasks_completed: list of subtask IDs completed this turn
        """
        from collections import defaultdict

        # Group by turn
        turns: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "agents": {},
            "subtasks_completed": []
        })

        for record in action_history:
            turn = record.get("turn", 0)

            # Check if this is a subtask completion record
            if record.get("type") == "subtask_completion":
                turns[turn]["subtasks_completed"].extend(record.get("subtasks_completed", []))
            else:
                # Regular action record
                agent_id = record.get("agent", "unknown")
                turns[turn]["agents"][agent_id] = {
                    "action": record.get("action", ""),
                    "observation": record.get("result", ""),
                }

        # Convert to sorted list
        trajectory = []
        for turn_num in sorted(turns.keys()):
            entry = turns[turn_num]
            trajectory.append({
                "turn": turn_num,
                "agents": entry["agents"],
                "subtasks_completed": entry["subtasks_completed"],
            })

        return trajectory

    def _save_calibration_result(self, task_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Save calibration results to task JSON for dataset pass rate tracking.

        Uses the unified array-based calibration format with per-agent model info.
        """
        from datetime import datetime
        from emtom.evolve.benchmark_wrapper import _migrate_legacy_calibration

        # Build per-agent model mapping from the last benchmark run
        agent_models = getattr(self, "_last_agent_models", None)
        if not agent_models:
            # Fallback: derive from test_model or generator model
            model_name = self.test_model or "unknown"
            if not self.test_model:
                if hasattr(self.llm, 'llm_conf'):
                    params = self.llm.llm_conf.get('generation_params', {})
                    model_name = params.get('model', 'unknown')
                elif hasattr(self.llm, 'generation_params'):
                    model_name = getattr(self.llm.generation_params, 'model', 'unknown')
            num_agents = task_data.get("num_agents", 2)
            agent_models = {f"agent_{i}": model_name for i in range(num_agents)}

        category = task_data.get("category", "")

        # Migrate legacy dict format and write as array
        raw_cal = task_data.get("calibration", [])
        calibration = _migrate_legacy_calibration(raw_cal)
        tested_at = datetime.now().isoformat()

        for run_mode in ("standard", "baseline"):
            run_result = results.get(run_mode)
            if not isinstance(run_result, dict):
                continue

            action_history = run_result.get("action_history", [])
            trajectory = self._build_trajectory(action_history)
            calibration_entry = {
                "tested_at": tested_at,
                "run_mode": run_mode,
                "agent_models": agent_models,
                "steps": run_result.get("steps", 0),
                "results": _build_results_block(category, run_result.get("evaluation", {})),
                "trajectory": trajectory,
            }

            replaced = False
            for i, existing in enumerate(calibration):
                existing_run_mode = str(existing.get("run_mode", "standard") or "standard")
                if existing.get("agent_models") == agent_models and existing_run_mode == run_mode:
                    calibration[i] = calibration_entry
                    replaced = True
                    break
            if not replaced:
                calibration.append(calibration_entry)

        task_data["calibration"] = calibration

        # Write back to working_task.json
        try:
            with open(self.task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            self._log(f"[Calibration] Saved standard+baseline results, agents={agent_models}")
        except Exception as e:
            self._log(f"[Calibration] Warning: Failed to save calibration result: {e}")

    def _validate_task_structure(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task JSON structure without running benchmark."""
        from emtom.cli.validate_task import validate
        result = validate(task_data, self.scene_data)
        if result["success"]:
            return result["data"]
        return {"valid": False, "error": result["error"], "summary": "Task validation failed"}

    def _static_validate_trajectory(self, task_data: Dict[str, Any], golden: List[Dict]) -> List[str]:
        """Fast static validation of golden trajectory (no simulator required)."""
        from emtom.cli.validate_task import static_validate_trajectory
        return static_validate_trajectory(task_data, golden, self.scene_data)

    def _run_benchmark(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run standard and baseline benchmarks in parallel subprocesses."""
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        current_task_num = len(self.submitted_tasks) + 1
        self._test_run_count += 1
        run_dir = self.trajectories_dir / f"task_{current_task_num}" / f"run_{self._test_run_count}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task_data, f)
                temp_task_file = f.name
        except Exception as e:
            return {"error": f"Failed to write temp task file: {e}"}

        self._last_agent_models = self._determine_agent_models(task_data)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    run_mode: executor.submit(
                        self._run_benchmark_mode,
                        task_data,
                        temp_task_file,
                        run_mode,
                        run_dir / run_mode,
                    )
                    for run_mode in ("standard", "baseline")
                }
                mode_results = {
                    run_mode: future.result()
                    for run_mode, future in futures.items()
                }
        finally:
            self._cleanup_temp_files(temp_task_file)

        mode_errors = {
            run_mode: result["error"]
            for run_mode, result in mode_results.items()
            if result.get("error")
        }
        if mode_errors:
            return {
                "error": "; ".join(f"{mode}: {err}" for mode, err in sorted(mode_errors.items())),
                "mode_errors": mode_errors,
                "trajectory_dir": str(run_dir),
            }

        category = task_data.get("category", "")
        comparison = _build_mode_comparison(
            category,
            mode_results["standard"],
            mode_results["baseline"],
            current_rate=self.calibration_stats.get("rate"),
            target_rate=self.calibration_stats.get("target_rate", 0.20),
            current_passed=self.calibration_stats.get("passed"),
            current_failed=self.calibration_stats.get("failed"),
        )
        merged = {
            "standard": mode_results["standard"],
            "baseline": mode_results["baseline"],
            "comparison": comparison,
            "trajectory_dir": str(run_dir),
        }
        self._write_benchmark_comparison(run_dir, merged)
        return merged

    def _determine_agent_models(self, task_data: Dict[str, Any]) -> Dict[str, str]:
        """Resolve the model assignment used by the benchmark runs."""
        num_agents = task_data.get("num_agents", 2)
        if task_data.get("category") == "competitive":
            base_model = self.test_model or "gpt-5.2"
            opponent = "sonnet" if base_model != "sonnet" else "gpt-5.2"
            team_assignment = task_data.get("team_assignment", {})
            agent_models: Dict[str, str] = {}
            team_models = {"team_0": base_model, "team_1": opponent}
            for team_id, agents in team_assignment.items():
                model = team_models.get(team_id, base_model)
                for agent_id in agents:
                    agent_models[agent_id] = model
            if agent_models:
                return agent_models
            return {f"agent_{i}": base_model for i in range(num_agents)}

        model_name = self.test_model or "gpt-5.2"
        return {f"agent_{i}": model_name for i in range(num_agents)}

    def _run_benchmark_mode(
        self,
        task_data: Dict[str, Any],
        temp_task_file: str,
        run_mode: str,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Run one benchmark mode in a subprocess (fresh GL context)."""
        run_dir.mkdir(parents=True, exist_ok=True)

        num_agents = task_data.get("num_agents", 2)
        cmd = [
            sys.executable, "-m", "emtom.cli.test_task",
            temp_task_file,
            "--working-dir", str(self.working_dir),
            "--trajectory-dir", str(run_dir),
            "--config-name", f"examples/emtom_{num_agents}_robots",
            "--run-mode", run_mode,
        ]
        if self.test_model:
            cmd.extend(["--test-model", self.test_model])

        if task_data.get("category") == "competitive":
            base_model = self.test_model or "gpt-5.2"
            opponent = "sonnet" if base_model != "sonnet" else "gpt-5.2"
            team_model_map_str = f"team_0={base_model},team_1={opponent}"
            cmd.extend(["--team-model-map", team_model_map_str])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200,
            )
            result = self._parse_benchmark_subprocess(proc, run_dir)
            result["run_mode"] = run_mode
            return result

        except subprocess.TimeoutExpired:
            return {"steps": 0, "done": False, "error": "Test timed out", "trajectory_dir": str(run_dir), "run_mode": run_mode}
        except Exception as e:
            return {"steps": 0, "done": False, "error": f"Subprocess error: {e}", "trajectory_dir": str(run_dir), "run_mode": run_mode}

    def _parse_benchmark_subprocess(self, proc: subprocess.CompletedProcess[str], run_dir: Path) -> Dict[str, Any]:
        """Parse the JSON payload emitted by emtom.cli.test_task."""
        try:
            stdout = proc.stdout
            json_start = stdout.find("{")
            if json_start >= 0:
                stdout = stdout[json_start:]
            result_data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return {"steps": 0, "done": False, "error": f"Failed to parse output: {proc.stderr[:500]}", "trajectory_dir": str(run_dir)}

        if not isinstance(result_data, dict):
            return {"steps": 0, "done": False, "error": f"Unexpected output type ({type(result_data).__name__})", "trajectory_dir": str(run_dir)}

        if result_data.get("success"):
            result = result_data.get("data", {})
        else:
            result = {"steps": 0, "done": False, "error": result_data.get("error", "Unknown error")}

        result["trajectory_dir"] = str(run_dir)
        result_txt_path = run_dir / "result.txt"
        if result_txt_path.exists():
            try:
                with open(result_txt_path) as f:
                    result["result_txt"] = f.read()
            except Exception:
                pass
        result.pop("planner_traces", None)
        return result

    def _write_benchmark_comparison(self, run_dir: Path, results: Dict[str, Any]) -> None:
        """Persist a compact dual-run comparison artifact for judge/debug use."""
        comparison_file = run_dir / "comparison.json"
        payload = {
            "standard": {
                "run_mode": "standard",
                "steps": results["standard"].get("steps", 0),
                "turns": results["standard"].get("turns", 0),
                "evaluation": results["standard"].get("evaluation", {}),
                "trajectory_dir": results["standard"].get("trajectory_dir"),
            },
            "baseline": {
                "run_mode": "baseline",
                "steps": results["baseline"].get("steps", 0),
                "turns": results["baseline"].get("turns", 0),
                "evaluation": results["baseline"].get("evaluation", {}),
                "trajectory_dir": results["baseline"].get("trajectory_dir"),
            },
            "comparison": results.get("comparison", {}),
            "agent_models": getattr(self, "_last_agent_models", {}),
        }
        try:
            with open(comparison_file, "w") as f:
                json.dump(payload, f, indent=2)
            self._log(f"[Calibration] Comparison saved to: {comparison_file}")
        except Exception as e:
            self._log(f"[Calibration] Warning: Failed to save comparison file: {e}")
    def _regenerate_golden_trajectory(
        self,
        task_data: Dict[str, Any],
        source: str,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Regenerate golden trajectory deterministically via emtom.pddl.planner."""
        from emtom.pddl.planner import regenerate_golden_trajectory

        return regenerate_golden_trajectory(
            task_data,
            scene_data=self.scene_data,
            source=source,
            task_file=str(self.task_file) if persist else None,
        )

    def _verify_golden_trajectory(self) -> str:
        return json.dumps({
            "valid": False,
            "error": (
                "verify_golden_trajectory[] is no longer available. "
                "Run judge[]; it now regenerates the plan, simulator-verifies it when needed, "
                "and then evaluates task quality."
            ),
        }, indent=2)

    def _judge(self) -> str:
        """Evaluate task quality using multi-model council."""
        from emtom.cli.judge_task import run

        # Find trajectory dir for the CURRENT task (not the latest overall)
        current_task_num = len(self.submitted_tasks) + 1
        task_traj_dir = self.trajectories_dir / f"task_{current_task_num}"
        traj_dir = None
        if task_traj_dir.exists():
            run_dirs = sorted(task_traj_dir.glob("run_*"), key=lambda p: p.name)
            if run_dirs:
                traj_dir = str(run_dirs[-1])

        result = run(
            str(self.task_file),
            working_dir=str(self.working_dir),
            trajectory_dir=traj_dir,
            threshold=self.judge.overall_threshold,
            difficulty=self.difficulty if self.difficulty else None,
            required_tom_level=self._current_k_level,
            verified_trajectory_hash=(
                self.last_verified_trajectory_hash if self.last_verify_passed else None
            ),
            skip_steps=self.skip_steps or None,
        )

        if not result["success"]:
            return json.dumps({"valid": False, "error": result["error"]}, indent=2)

        data = result["data"]
        golden = data.get("golden_trajectory", {})
        if golden.get("sim_verified"):
            self.last_verify_passed = True
            self.last_verified_spec_hash = golden.get("spec_hash")
            self.last_verified_trajectory_hash = golden.get("trajectory_hash")
        self.last_judge_passed = data["passed"]

        # Reconstruct CouncilVerdict for state tracking
        if data["passed"]:
            self.consecutive_tom_failures = 0
            # Strip required_fixes on pass — the LLM judge sometimes returns
            # suggestions even for passing tasks, which misleads the agent into
            # editing instead of progressing to test_task/submit.
            data.pop("required_fixes", None)
            data["next_step"] = (
                "Task passed judge. Do NOT change the task design. "
                "Run test_task[] -> submit_task[] immediately."
            )
        else:
            self.consecutive_tom_failures += 1
            data["action_required"] = "Modify the task using required_fixes and run judge[] again."
            data["failure_count"] = self.consecutive_tom_failures
            if self.consecutive_tom_failures >= 5:
                data["recommendation"] = (
                    f"You've failed judge {self.consecutive_tom_failures} times. "
                    "Consider new_scene[] for a fresh start."
                )

        self._log(f"[Judge] Result: {'PASS' if data['passed'] else 'FAIL'} "
                   f"(score: {data['overall_score']:.2f}) [failures: {self.consecutive_tom_failures}]")

        return json.dumps(data, indent=2)

    def _submit_task(self) -> str:
        """Copy working task to output directory (requires judge/test gates)."""
        _skip = set(self.skip_steps or [])
        if not self.last_judge_passed and "llm-council" not in _skip:
            return json.dumps({
                "error": (
                    "Must run judge[] first and pass before submitting. "
                    "judge includes task quality evaluation."
                ),
                "hint": "Run judge[] to verify the task quality.",
                "last_score": self.last_judgment.overall_score if self.last_judgment else None,
                "required_fixes": self.last_judgment.required_fixes if self.last_judgment else []
            })
        # In some environments simulator verification is unavailable (e.g., missing Hydra/GL deps).
        # When simulation is skipped, allow submission as long as judge + test passed.
        if not self.last_verify_passed and "simulation" not in _skip:
            return json.dumps({
                "error": (
                    "Must run judge[] after the latest task changes so the regenerated golden "
                    "trajectory is simulator-verified before submitting. "
                    "If simulator verification is unavailable, add \"simulation\" to skip_steps in taskgen_state.json."
                ),
                "hint": "Run judge[] again.",
            })
        if not self.last_test_passed and "test" not in _skip and "task-evolution" not in _skip:
            return json.dumps({
                "error": "Must run test_task[] before submitting (required for calibration data).",
                "hint": "Run test_task[] to benchmark LLM agent performance."
            })

        from emtom.cli.submit_task import run

        # Enforce the k-level assigned for this task.
        allowed_tom_levels = [self._current_k_level] if self._current_k_level else None

        result = run(
            str(self.task_file),
            output_dir=str(self.output_dir),
            working_dir=str(self.working_dir),
            submitted_dir=str(self.submitted_tasks_dir),
            subtasks_min=self.subtasks_min,
            subtasks_max=self.subtasks_max,
            agents_min=self.agents_min,
            agents_max=self.agents_max,
            allowed_tom_levels=allowed_tom_levels,
        )
        if not result["success"]:
            return json.dumps({"error": result["error"]}, indent=2)

        data = result["data"]
        self.submitted_tasks.append(data["output_path"])
        self._advance_calibration_stats(self._last_test_comparison)

        # Reset verification state for next task
        self.last_verify_passed = False
        self.last_judge_passed = False
        self.last_test_passed = False
        self.last_judgment = None
        self._last_test_comparison = None
        self.consecutive_tom_failures = 0

        # Extract memory and track diversity
        task_title = data.get("title", "untitled")
        memory = self._extract_task_memory(task_title)
        self.task_memories.append(memory)

        with open(self.task_file) as f:
            task_data = json.load(f)
        pattern = self.diversity_tracker.add_pattern(data["filename"], task_data)
        self._log(f"Diversity pattern added: {pattern}")

        try:
            from emtom.scripts.categorize_tasks import categorize_tasks
            categories_file = Path("data/emtom/task_categories.json")
            categorize_tasks(self.llm, self.output_dir, categories_file)
            self._refresh_diversity_in_prompt()
        except Exception as e:
            self._log(f"Warning: Failed to re-categorize tasks: {e}")

        self._reset_context_for_next_task()

        return (
            f"Task '{task_title}' saved!\n"
            f"  - {data['output_path']} (permanent)\n"
            f"  - {data.get('submitted_path', 'N/A')} (session)\n"
            f"Total submitted: {len(self.submitted_tasks)}\n\n"
            "[Context reset for next task.]"
        )

    def _advance_calibration_stats(self, comparison: Optional[Dict[str, Any]]) -> None:
        if not comparison or "standard_passed" not in comparison:
            return

        passed = int(self.calibration_stats.get("passed", 0))
        failed = int(self.calibration_stats.get("failed", 0))
        if comparison.get("standard_passed"):
            passed += 1
        else:
            failed += 1

        self.calibration_stats["passed"] = passed
        self.calibration_stats["failed"] = failed
        self.calibration_stats["total"] = passed + failed
        self.calibration_stats["rate"] = (
            passed / (passed + failed) if (passed + failed) > 0 else None
        )

    def _extract_task_memory(self, task_title: str) -> str:
        """
        Ask LLM what to remember from the completed task.

        Returns a concise memory string to carry forward.
        """
        self._log(f"Extracting memory from task: {task_title}")

        # Get recent conversation context
        recent_messages = self.messages[-10:] if len(self.messages) > 10 else self.messages[2:]
        conversation_snippet = []
        for msg in recent_messages:
            content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
            conversation_snippet.append(f"{msg['role']}: {content}")

        memory_prompt = f"""You just completed task: "{task_title}"

Recent conversation:
{chr(10).join(conversation_snippet)}

What are the 2-3 most important learnings to remember for future tasks?
Focus on: patterns that worked, mistakes to avoid, useful techniques.
Be very concise (2-3 bullet points, max 100 words total).

LEARNINGS:"""

        try:
            memory = self.llm.generate(memory_prompt)
            # Keep it concise
            if len(memory) > 500:
                memory = memory[:500] + "..."
            self._log(f"Task memory: {memory}")
            return f"Task {len(self.submitted_tasks)}: '{task_title}'\n{memory}"
        except Exception as e:
            self._log(f"Memory extraction failed: {e}")
            return f"Task {len(self.submitted_tasks)}: '{task_title}' - completed successfully"

    def _reset_context_for_next_task(self) -> None:
        """
        Reset conversation context for the next task.

        Preserves:
        - System prompt
        - Task memories (learnings from completed tasks)
        - Current scene info

        Resets:
        - Run counter (for fresh task_N/run_1 paths)
        """
        self._log("Resetting context for next task...")

        # Pick a new k-level for the next task
        import random
        if self._allowed_k_levels:
            self._current_k_level = random.choice(self._allowed_k_levels)
        else:
            self._current_k_level = random.choice([1, 2, 3])
        self._log(f"Next task k-level target: {self._current_k_level}")

        # Reset run counter for new task (trajectories use task_N/run_M structure)
        self._test_run_count = 0

        # Keep system prompt
        system_prompt = self.messages[0]["content"] if self.messages else ""

        # Build memories summary
        if self.task_memories:
            memories_text = "\n\n".join(self.task_memories[-5:])  # Keep last 5 task memories
            memories_msg = f"""## Learnings from Previous Tasks ({len(self.submitted_tasks)} completed)

{memories_text}

Use these learnings to improve your next task. Avoid repeating mistakes."""
        else:
            memories_msg = ""

        # Build fresh scene info
        scene_info = self._format_scene_data()

        # Reset messages with fresh context
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

        if memories_msg:
            self.messages.append({"role": "user", "content": memories_msg})
            self.messages.append({"role": "assistant", "content": "I'll apply these learnings to create better tasks. What's next?"})

        # Rebuild extra sections with the new k-level for this task
        k_section = ""
        if self._current_k_level is not None:
            k_section = (
                f"\n## Required K-Level: {self._current_k_level}\n"
                f"This task MUST be Theory-of-Mind level {self._current_k_level}.\n"
                "K=0 tasks are invalid and will be rejected.\n"
                f"Design epistemic goals so that `judge[]`'s strict PDDL verification computes tom_level = {self._current_k_level}.\n"
                f"submit_task[] will REJECT the task if the computed tom_level is not {self._current_k_level}.\n"
            )
        static = getattr(self, "_static_extra_sections", "")
        extra = static + k_section
        self._extra_sections = extra

        # Add current scene context with task number
        next_task_num = len(self.submitted_tasks) + 1
        scene_msg = f"""## Task {next_task_num} - Current Scene
{extra}{scene_info}

working_task.json is ready. Trajectories will be saved to agent_trajectories/task_{next_task_num}/run_N/.
Use new_scene[] if you want a different scene, or start creating your next task."""
        self.messages.append({"role": "user", "content": scene_msg})

        new_size = self._estimate_context_size()
        self._log(f"Context reset. New size: {new_size} chars. Memories preserved: {len(self.task_memories)}")

    def _new_scene(self, args: str = "") -> str:
        """Load a scene for task generation (subprocess for GL context)."""
        import tempfile
        from emtom.task_gen.scene_loader import SceneData
        from habitat_llm.utils import get_random_seed

        args = args.strip()
        if not args:
            return "Error: new_scene requires num_agents. Usage: new_scene[N] or new_scene[N, keep]"

        parts = [p.strip() for p in args.split(",")]
        try:
            num_agents = int(parts[0])
        except ValueError:
            return f"Error: num_agents must be an integer, got '{parts[0]}'"

        if num_agents < 2 or num_agents > 10:
            return f"Error: num_agents must be 2-10, got {num_agents}"

        keep_mode = len(parts) > 1 and parts[1].lower() == "keep"
        scene_id = None
        if keep_mode:
            if self.scene_data is None:
                return "Error: new_scene[N, keep] requires a scene. Use new_scene[N] first."
            scene_id = self.scene_data.scene_id

        self._log(f"Loading scene (num_agents={num_agents}, keep={keep_mode})...")

        max_scene_retries = 5
        last_error = None

        for attempt in range(1, max_scene_retries + 1):
            try:
                new_seed = get_random_seed()
                cmd = [
                    sys.executable, "-m", "emtom.cli.new_scene",
                    str(num_agents),
                    "--working-dir", str(self.working_dir),
                    "--seed", str(new_seed),
                ]
                if scene_id:
                    cmd.extend(["--scene-id", scene_id])

                self._log(f"Running (attempt {attempt}/{max_scene_retries}): {' '.join(cmd)}")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                try:
                    # Extract JSON from stdout (Hydra may prepend non-JSON lines)
                    stdout = proc.stdout
                    json_start = stdout.find("{")
                    if json_start >= 0:
                        stdout = stdout[json_start:]
                    result = json.loads(stdout)
                except (json.JSONDecodeError, ValueError):
                    last_error = proc.stderr or "No output from subprocess"
                    self._log(f"Scene loading failed (attempt {attempt}): {last_error}")
                    if attempt < max_scene_retries:
                        continue
                    return f"Error loading new scene after {max_scene_retries} attempts. Last error: {last_error}"

                if not isinstance(result, dict) or not result.get("success"):
                    last_error = result.get("error", "Unknown error") if isinstance(result, dict) else f"Unexpected output type: {type(result).__name__}"
                    self._log(f"Scene loading failed (attempt {attempt}): {last_error}")
                    if attempt < max_scene_retries:
                        continue
                    return f"Error loading new scene after {max_scene_retries} attempts. Last error: {last_error}"
            except Exception as e:
                last_error = str(e)
                self._log(f"Scene loading error (attempt {attempt}): {e}")
                if attempt < max_scene_retries:
                    continue
                return f"Error loading new scene after {max_scene_retries} attempts. Last error: {last_error}"

            # Convert dict back to SceneData
            data = result.get("data", {})
            if not isinstance(data, dict) or "scene_data" not in data:
                return f"Error: scene loading returned malformed data (missing 'data.scene_data')"
            scene_dict = data["scene_data"]
            candidate = SceneData(
                episode_id=scene_dict["episode_id"],
                scene_id=scene_dict["scene_id"],
                rooms=scene_dict["rooms"],
                furniture=scene_dict["furniture"],
                objects=scene_dict["objects"],
                articulated_furniture=scene_dict["articulated_furniture"],
                furniture_in_rooms=scene_dict["furniture_in_rooms"],
                objects_on_furniture=scene_dict["objects_on_furniture"],
                agent_spawns=scene_dict.get("agent_spawns", {}),
            )

            # Reject sparse scenes that routinely fail novelty/necessity checks.
            min_objects = 5
            if self.category == "competitive":
                min_objects = 4
            if len(candidate.objects) < min_objects:
                last_error = (
                    f"Scene {candidate.scene_id} (ep {candidate.episode_id}) "
                    f"has only {len(candidate.objects)} locatable objects "
                    f"(need >= {min_objects}). Retrying..."
                )
                self._log(last_error)
                if attempt < max_scene_retries:
                    continue
                return f"Error loading new scene after {max_scene_retries} attempts. Last error: {last_error}"

            # Require object distribution across at least 2 rooms when possible.
            furniture_to_room: Dict[str, str] = {}
            for room_id, room_furniture in (candidate.furniture_in_rooms or {}).items():
                if not isinstance(room_id, str) or not isinstance(room_furniture, list):
                    continue
                for furn_id in room_furniture:
                    if isinstance(furn_id, str):
                        furniture_to_room[furn_id] = room_id
            if furniture_to_room:
                rooms_with_objects = set()
                for furn_id, objects_on_furn in (candidate.objects_on_furniture or {}).items():
                    if not isinstance(furn_id, str) or not isinstance(objects_on_furn, list):
                        continue
                    if not objects_on_furn:
                        continue
                    room_id = furniture_to_room.get(furn_id)
                    if room_id:
                        rooms_with_objects.add(room_id)
                if len(rooms_with_objects) < 2:
                    last_error = (
                        f"Scene {candidate.scene_id} (ep {candidate.episode_id}) "
                        f"has objects concentrated in {len(rooms_with_objects)} room(s) "
                        "(need >= 2 for robust multi-agent coordination). Retrying..."
                    )
                    self._log(last_error)
                    if attempt < max_scene_retries:
                        continue
                    return (
                        f"Error loading new scene after {max_scene_retries} attempts. "
                        f"Last error: {last_error}"
                    )

            self.scene_data = candidate

            self._log(f"Loaded scene {self.scene_data.scene_id} (episode {self.scene_data.episode_id})")

            # Reset verification state
            self.last_verify_passed = False
            self.last_judge_passed = False
            self.last_test_passed = False
            self.last_judgment = None
            self.consecutive_tom_failures = 0

            if keep_mode:
                task_file = self.working_dir / "working_task.json"
                if task_file.exists():
                    try:
                        with open(task_file) as f:
                            task_data = json.load(f)
                        task_data["num_agents"] = num_agents
                        if self.scene_data.agent_spawns:
                            task_data["agent_spawns"] = self.scene_data.agent_spawns
                        # Clear derived fields that reference the old scene
                        for field in (
                            "golden_trajectory",
                            "functional_goal_pddl",
                            "literal_tom_probes",
                            "calibration",
                            "benchmark_results",
                        ):
                            task_data.pop(field, None)
                        with open(task_file, "w") as f:
                            json.dump(task_data, f, indent=2)
                    except (json.JSONDecodeError, KeyError):
                        self._create_working_task_from_template(num_agents=num_agents)
                else:
                    self._create_working_task_from_template(num_agents=num_agents)
                return f"""Scene reloaded with {num_agents} agents. Task preserved.
Scene ID: {self.scene_data.scene_id}
Rooms: {', '.join(self.scene_data.rooms)}
working_task.json preserved. Verification flags reset."""
            else:
                self._create_working_task_from_template(num_agents=num_agents)
                self._reset_context_for_next_task()
                return f"""New scene loaded! Context refreshed.
Scene ID: {self.scene_data.scene_id}
Episode ID: {self.scene_data.episode_id}
Agents: {num_agents}
Rooms: {', '.join(self.scene_data.rooms)}
Furniture: {len(self.scene_data.furniture)} items
Objects: {len(self.scene_data.objects)} items
working_task.json reset."""

    def _fail(self, reason: str) -> str:
        """
        Mark task generation as failed.

        This should only be used for truly unrecoverable errors.
        """
        # Prevent premature give-up: require at least 40% of iteration budget
        min_iterations = max(60, int(self.iterations_per_task * 0.4))
        if self.iteration_count < min_iterations:
            return (
                f"Cannot abort yet — only {self.iteration_count}/{min_iterations} "
                f"minimum iterations used. Keep trying: load a new_scene[], "
                f"simplify the task design, or try a different approach. "
                f"Do NOT call fail[] again until you have used at least {min_iterations} iterations. "
                f"Your reason was: {reason}"
            )
        self.failed = True
        self.fail_reason = reason
        self._log(f"FAIL: {reason}")
        return f"Task generation aborted: {reason}"

    def _format_scene_data(self) -> str:
        """Format scene data as room→furniture→object hierarchy for the LLM prompt."""
        if not self.scene_data:
            return "No scene data available."

        lines = []

        # Guidance about IDs vs natural language
        lines.append("**ID Usage Rules:**")
        lines.append("- `problem_pddl` and `mechanic_bindings`: Use EXACT object IDs from this list")
        lines.append("- `task` description: Keep it high-level and non-leaking; natural language is preferred and exact IDs are optional")
        lines.append("- `agent_secrets`: Use EXACT scene IDs for goal-critical objects, furniture, and rooms whenever the agent needs precise grounding\n")

        # Build reverse mappings
        obj_locations = {}
        for furn, objs in self.scene_data.objects_on_furniture.items():
            for obj in objs:
                obj_locations[obj] = furn

        furn_to_room = {}
        for room, furns in self.scene_data.furniture_in_rooms.items():
            for furn in furns:
                furn_to_room[furn] = room

        articulated_set = set(self.scene_data.articulated_furniture)

        # Room → Furniture → Object hierarchy
        lines.append("### Scene Layout (Room → Furniture → Objects)")
        lines.append("Furniture marked with [A] can be opened/closed (articulated). Use `is_open`/`is_closed` ONLY on [A] furniture.\n")

        for room in self.scene_data.rooms:
            lines.append(f"**{room}**")
            room_furniture = self.scene_data.furniture_in_rooms.get(room, [])
            if not room_furniture:
                lines.append("  (no furniture)")
            for furn in room_furniture:
                tag = " [A]" if furn in articulated_set else ""
                # Find objects on this furniture
                objs_on = self.scene_data.objects_on_furniture.get(furn, [])
                if objs_on:
                    obj_list = ", ".join(objs_on)
                    lines.append(f"  - {furn}{tag} ← {obj_list}")
                else:
                    lines.append(f"  - {furn}{tag}")
            lines.append("")

        # Unassigned furniture (not in any room mapping)
        all_mapped = set()
        for furns in self.scene_data.furniture_in_rooms.values():
            all_mapped.update(furns)
        unmapped = [f for f in self.scene_data.furniture if f not in all_mapped]
        if unmapped:
            lines.append("**Unmapped Furniture (room unknown — avoid in room_restriction goals)**")
            for furn in unmapped:
                tag = " [A]" if furn in articulated_set else ""
                lines.append(f"  - {furn}{tag}")
            lines.append("")

        # Objects without known locations (warn)
        orphan_objs = [obj for obj in self.scene_data.objects if obj not in obj_locations]
        if orphan_objs:
            lines.append(f"**WARNING: {len(orphan_objs)} objects have unknown locations (DO NOT USE):** {', '.join(orphan_objs)}")
            lines.append("")

        # Summary counts for quick reference
        located_count = sum(1 for obj in self.scene_data.objects if obj in obj_locations)
        lines.append(f"**Summary:** {len(self.scene_data.rooms)} rooms, {len(self.scene_data.furniture)} furniture ({len(articulated_set)} articulated), {located_count} usable objects")

        return "\n".join(lines)

    def _build_diversity_section(self) -> str:
        """Build the diversity section for the system prompt."""
        pattern_count = self.diversity_tracker.get_pattern_count()
        patterns = self.diversity_tracker.get_patterns_for_prompt()

        # Load existing task categories to avoid repetition
        existing_categories = self._load_existing_categories()

        if pattern_count == 0 and not existing_categories:
            return "No previous tasks yet. Be creative with your task structure and win conditions!"

        sections = []

        # Add existing category information
        if existing_categories:
            sections.append(f"""## EXISTING TASK CATEGORIES (DO NOT REPEAT THESE)

The following task types ALREADY EXIST in the dataset. You MUST create something FUNDAMENTALLY DIFFERENT.

{existing_categories}

**YOU ARE FORBIDDEN FROM CREATING TASKS THAT FIT THE ABOVE CATEGORIES.**
Think of entirely new gameplay patterns, win conditions, and mechanics combinations.""")

        # Add recent patterns if any
        if pattern_count > 0:
            sections.append(f"""## Recent Task Patterns ({pattern_count} total)
{patterns}""")

        # Add diversity guidance
        sections.append("""**CRITICAL - NOVELTY REQUIRED**:
- Your task MUST NOT fit any of the existing categories above
- Invent NEW win conditions (not race/retrieval/unlock/stash/placement)
- Invent NEW mechanics interactions (not just key chains or remote triggers)
- Invent NEW collaboration patterns (not just information sharing or room restrictions)
- Think: deception, time pressure, resource management, defense, sabotage, trading, voting, signaling, bluffing
- The task should make reviewers say "we've never seen a task like this before" """)

        return "\n\n".join(sections)

    def _load_existing_categories(self) -> str:
        """Load and format existing task categories from task_categories.json."""
        categories_file = Path("data/emtom/task_categories.json")
        if not categories_file.exists():
            return ""

        try:
            with open(categories_file) as f:
                categories = json.load(f)
        except (json.JSONDecodeError, IOError):
            return ""

        lines = []
        for main_category, subcategories in categories.items():
            lines.append(f"### {main_category.upper()}")
            for subcat_name, subcat_info in subcategories.items():
                description = subcat_info.get("description", "")
                lines.append(f"- **{subcat_name}**: {description}")
            lines.append("")  # Empty line between categories

        return "\n".join(lines)

    def _wrap_diversity_section(self, content: str) -> str:
        """Wrap diversity content with markers so it can be found and replaced later."""
        return f"{_DIVERSITY_START_MARKER}\n{content}\n{_DIVERSITY_END_MARKER}"

    def _refresh_diversity_in_prompt(self) -> None:
        """Rebuild the diversity section and replace it in the system prompt."""
        new_section = self._wrap_diversity_section(self._build_diversity_section())
        system_content = self.messages[0]["content"]

        start = system_content.find(_DIVERSITY_START_MARKER)
        end = system_content.find(_DIVERSITY_END_MARKER)

        if start == -1 or end == -1:
            self._log("Warning: Could not find diversity markers in system prompt, skipping refresh")
            return

        end += len(_DIVERSITY_END_MARKER)
        self.messages[0]["content"] = system_content[:start] + new_section + system_content[end:]
        self._log("Refreshed diversity section in system prompt")

    @staticmethod
    def _cleanup_temp_files(*paths: str) -> None:
        """Remove temp files, ignoring errors."""
        for path in paths:
            try:
                os.unlink(path)
            except Exception:
                pass

    def _log(self, message: str, truncate_terminal: int = 0) -> None:
        """Print log message and write to log file.

        Args:
            message: Message to log
            truncate_terminal: If > 0, truncate terminal output to this many chars
                               (log file always gets full content)
        """
        # Always write full message to log file
        if hasattr(self, '_log_handle') and self._log_handle:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_handle.write(f"[{timestamp}] {message}\n")
            self._log_handle.flush()

        # Print to terminal if verbose (optionally truncated)
        if self.verbose:
            if truncate_terminal > 0 and len(message) > truncate_terminal:
                print(f"{message[:truncate_terminal]}...", flush=True)
            else:
                print(message, flush=True)

    def close(self) -> None:
        """Close log file handle."""
        if hasattr(self, '_log_handle') and self._log_handle:
            self._log_handle.close()
            self._log_handle = None
