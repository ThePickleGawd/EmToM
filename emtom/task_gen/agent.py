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
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .judge import Judge, Judgment, CouncilVerdict, Colors
from .diversity import DiversityTracker
from .spec_validator import (
    validate_blocking_spec,
    validate_room_restriction_trajectory,
)
from emtom.actions import ActionRegistry

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


# Markers for the diversity section in the system prompt so it can be found and replaced
_DIVERSITY_START_MARKER = "<!-- DIVERSITY_SECTION_START -->"
_DIVERSITY_END_MARKER = "<!-- DIVERSITY_SECTION_END -->"


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
            verification_feedback: Optional dict with suggestions from a failed ToM verification to incorporate
            calibration_stats: Dataset calibration stats (pass rate, target rate) for difficulty guidance
            category: Task category to generate: "cooperative", "competitive", or "mixed" (None = random)
            seed_task: Optional path to existing task JSON to use as seed instead of blank template
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
        self.difficulty = difficulty  # Difficulty level for evolve pipeline
        self.test_model = test_model  # Override model for test_task calibration

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
            default_actions = ["Navigate", "Open", "Close", "Pick", "Place", "UseItem", "Communicate", "Wait"]
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
        self.task_memories: List[str] = []  # Learnings from completed tasks
        self.consecutive_tom_failures = 0  # Track failures to suggest new_scene
        self.diversity_tracker = DiversityTracker(llm=self.llm)  # Track task patterns for diversity
        # When difficulty is set (evolution mode), the query describes the
        # generation *process* (e.g. "study benchmark results"), not a task
        # design requirement. Don't pass it to the judge as user_query —
        # the difficulty parameter already constrains quality expectations.
        judge_query = query if not difficulty else None
        judge_kwargs = dict(
            verbose=verbose,
            user_query=judge_query,
            diversity_tracker=self.diversity_tracker,
            difficulty=difficulty,
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

        # Get available items from registry
        from emtom.state.item_registry import ItemRegistry
        available_items = ItemRegistry.get_items_for_task_generation()

        # Get available mechanics
        from emtom.mechanics import get_mechanics_for_task_generation
        available_mechanics = get_mechanics_for_task_generation()

        # Select category (random if not specified)
        import random
        task_category = self.category or random.choice(["cooperative", "competitive", "mixed"])

        # Get available predicates from domain
        from emtom.pddl.domain import get_predicates_for_prompt
        available_predicates = get_predicates_for_prompt()

        # Initialize conversation with action descriptions and paths injected
        replacements = {
            "{action_descriptions}": ActionRegistry.get_all_action_descriptions(),
            "{task_file}": str(self.task_file),
            "{working_dir}": str(self.working_dir),
            "{available_items}": available_items,
            "{available_mechanics}": available_mechanics,
            "{available_predicates}": available_predicates,
            "{category}": task_category.upper(),
            "{diversity_section}": self._wrap_diversity_section(self._build_diversity_section()),
        }
        system_prompt = SYSTEM_PROMPT
        for key, value in replacements.items():
            system_prompt = system_prompt.replace(key, value)
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
            suggestions = self.verification_feedback.get("suggestions", [])
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

            verification_section += "\n**Suggestions to Incorporate**:\n"
            for i, suggestion in enumerate(suggestions, 1):
                verification_section += f"{i}. {suggestion}\n"

            verification_section += "\nCreate a NEW task that specifically addresses these issues.\n"

        # Build calibration/difficulty guidance section
        calibration_section = ""
        if self.difficulty:
            # When difficulty is explicitly set (evolve pipeline), use it directly
            # and skip calibration stats to avoid conflicting guidance
            difficulty_guidance = {
                "easy": (
                    "## Difficulty: EASY\n"
                    "Generate SIMPLE tasks that weaker models can solve:\n"
                    "- Use 0-1 mechanics (prefer limited_bandwidth or room_restriction)\n"
                    "- Avoid inverse_state and chained conditional_unlock — models cannot discover these\n"
                    "- 2-3 agents with clear roles\n"
                    "- 2-3 subtasks maximum\n"
                    "- Secrets MUST explain any active mechanic in plain language\n"
                    "- All effects should be observable (no remote effects in unseen rooms)\n"
                    "- tom_level 1 only\n"
                    "- limited_bandwidth with generous limits (4-5 messages) works well at this level\n"
                ),
                "medium": (
                    "## Difficulty: MEDIUM\n"
                    "Generate moderately complex tasks:\n"
                    "- 1-2 mechanics with hints in secrets\n"
                    "- STRONGLY prefer including limited_bandwidth (2-3 messages per agent)\n"
                    "- 2-4 agents, meaningful coordination required\n"
                    "- 3-4 subtasks with some dependencies\n"
                    "- tom_level 1-2\n"
                ),
                "hard": (
                    "## Difficulty: HARD\n"
                    "Generate challenging tasks for top-tier models:\n"
                    "- 2+ mechanics, complex interactions\n"
                    "- MUST include limited_bandwidth with tight limits (1-2 messages per agent)\n"
                    "- 3-4+ agents with deep interdependencies\n"
                    "- 4+ subtasks with chained dependencies\n"
                    "- tom_level 2-3, multi-step reasoning required\n"
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

        # Build ToM ratio calibration guidance (independent of difficulty mode).
        tom_calibration_section = ""
        if self.calibration_stats:
            tom_target = self.calibration_stats.get("tom_target", {})
            tom_tolerance = self.calibration_stats.get("tom_tolerance", 0.08)
            tom_counts = self.calibration_stats.get("tom_counts", {})
            tom_total = self.calibration_stats.get("tom_total", 0)
            tom_unknown = self.calibration_stats.get("tom_unknown", 0)

            if isinstance(tom_target, dict) and any(level in tom_target for level in (1, 2, 3)):
                t1 = float(tom_target.get(1, 0.0))
                t2 = float(tom_target.get(2, 0.0))
                t3 = float(tom_target.get(3, 0.0))
                tol = float(tom_tolerance)

                tom_calibration_section = (
                    "\n## ToM Ratio Calibration\n"
                    f"Target ToM mix: level 1 = {t1:.0%}, level 2 = {t2:.0%}, level 3 = {t3:.0%}\n"
                    f"Tolerance: +/-{tol:.0%}\n\n"
                )

                if tom_total > 0:
                    current_ratios = {
                        1: float(tom_counts.get(1, 0)) / tom_total,
                        2: float(tom_counts.get(2, 0)) / tom_total,
                        3: float(tom_counts.get(3, 0)) / tom_total,
                    }
                    tom_calibration_section += (
                        "Current ToM mix (computed from dataset):\n"
                        f"- Level 1: {int(tom_counts.get(1, 0))}/{tom_total} ({current_ratios[1]:.1%})\n"
                        f"- Level 2: {int(tom_counts.get(2, 0))}/{tom_total} ({current_ratios[2]:.1%})\n"
                        f"- Level 3: {int(tom_counts.get(3, 0))}/{tom_total} ({current_ratios[3]:.1%})\n"
                    )
                    if tom_unknown:
                        tom_calibration_section += f"- Unknown/failed ToM inference: {tom_unknown}\n"

                    deficits = []
                    surpluses = []
                    for level, target in ((1, t1), (2, t2), (3, t3)):
                        current = current_ratios[level]
                        delta = current - target
                        if delta < -tol:
                            deficits.append((level, -delta))
                        elif delta > tol:
                            surpluses.append((level, delta))

                    if deficits:
                        deficits.sort(key=lambda x: x[1], reverse=True)
                        primary_level = deficits[0][0]
                        delta_pp = deficits[0][1] * 100
                        tom_calibration_section += (
                            f"\n**Guidance**: ToM level {primary_level} is most under target "
                            f"({delta_pp:.1f} percentage points). Prioritize generating level "
                            f"{primary_level} tasks until distribution rebalances.\n"
                            "Use `verify_pddl[]` while authoring to confirm computed tom_level.\n"
                        )
                    elif surpluses:
                        surpluses.sort(key=lambda x: x[1], reverse=True)
                        primary_level = surpluses[0][0]
                        delta_pp = surpluses[0][1] * 100
                        tom_calibration_section += (
                            f"\n**Guidance**: ToM level {primary_level} is over target "
                            f"({delta_pp:.1f} percentage points). De-emphasize level {primary_level} "
                            "and focus on underrepresented levels.\n"
                        )
                    else:
                        tom_calibration_section += (
                            "\n**Guidance**: Current ToM mix is within tolerance. "
                            "Maintain diversity while keeping the ratio balanced.\n"
                        )
                else:
                    tom_calibration_section += (
                        "No ToM-calibrated tasks yet. Start building toward the target mix.\n"
                        "- Generate a balanced seed set spanning ToM levels 1, 2, and 3.\n"
                        "- Use nested epistemic goals for higher levels and confirm with `verify_pddl[]`.\n"
                    )

        # Build seed task section if using a seed
        seed_section = ""
        if self.seed_task:
            seed_section = f"""
## Seed Task
A previous task has been loaded into working_task.json as your starting point.
Use it as a foundation and modify it based on the query/requirements above.
After calling `new_scene[N]`, view it with: `bash[cat {self.task_file}]`
The seed task's structure (subtasks, secrets, mechanics) is pre-populated - adapt it to the new scene and any requested changes.
"""

        # Build extra sections string and persist for context resets
        extra_sections = (
            query_section
            + seed_section
            + verification_section
            + calibration_section
            + tom_calibration_section
        )
        self._extra_sections = extra_sections

        # Initial user message - use template from prompts.py
        user_msg = USER_PROMPT_TEMPLATE.format(
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

        If self.seed_task is set, loads the seed task instead of the blank template.
        Scene fields (scene_id, episode_id, agent_spawns) are always updated from the current scene.

        Args:
            num_agents: Number of agents to use. If None, uses agents_max.

        Returns:
            The number of agents used.
        """
        if num_agents is None:
            num_agents = self.agents_max

        # Load seed task or blank template
        if self.seed_task:
            seed_path = Path(self.seed_task)
            with open(seed_path) as f:
                task = json.load(f)
            self._log(f"Loaded seed task from {seed_path}")
            # Use seed task's num_agents if not overridden
            if num_agents == self.agents_max and "num_agents" in task:
                num_agents = task["num_agents"]
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

        task["num_agents"] = num_agents

        # Only generate placeholder agent fields when not using a seed task
        if not self.seed_task:
            # Include Find* tools so agents can discover objects at runtime instead of hardcoded IDs
            default_actions = ["Navigate", "Open", "Close", "Pick", "Place", "UseItem", "FindObjectTool", "FindReceptacleTool", "FindRoomTool", "Communicate", "Wait"]
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

        self._log(f"Created {self.task_file} with {num_agents} agents{' (from seed)' if self.seed_task else ''}")
        return num_agents

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
            return self._bash(args)
        elif tool == "test_task":
            return self._test_task()
        elif tool == "verify_golden_trajectory":
            return self._verify_golden_trajectory()
        elif tool == "judge":
            return self._judge()
        elif tool == "verify_pddl":
            return self._verify_pddl()
        elif tool == "submit_task":
            return self._submit_task()
        elif tool == "new_scene":
            return self._new_scene(args)
        elif tool == "fail":
            return self._fail(args)
        else:
            return f"Unknown tool: {tool}. Available: bash, test_task, verify_pddl, verify_golden_trajectory, judge, submit_task, new_scene, fail"

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

            # Log detailed action history
            action_history = results.get("action_history", [])
            if action_history:
                self._log("\n=== Agent Action History ===")
                for entry in action_history:
                    self._log(f"  Turn {entry.get('turn', '?')}: {entry.get('agent', '?')} -> {entry.get('action', '?')}")

            # Save planner traces to separate file for reference
            planner_traces = results.get("planner_traces", {})
            if planner_traces:
                trace_file = self.log_dir / f"planner_traces_{datetime.now().strftime('%H%M%S')}.txt"
                with open(trace_file, 'w') as f:
                    for agent_id, trace in planner_traces.items():
                        f.write(f"\n{'='*60}\n")
                        f.write(f"=== {agent_id} Trace ===\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(trace)
                        f.write("\n")
                self._log(f"Planner traces saved to: {trace_file}")
                # Don't include full traces in the JSON response (too large)
                results["planner_traces"] = f"See {trace_file}"

            # Save calibration results to task JSON for dataset tracking (needs action_history)
            self._save_calibration_result(task_data, results)

            # Remove action_history from results before merging (too large for agent context)
            results.pop("action_history", None)

            # Benchmark ran successfully - merge results with validation
            validation_result.update(results)

            # Mark test as passed (for submit_task gate)
            self.last_test_passed = True

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

        # Build trajectory from action history
        action_history = results.get("action_history", [])
        trajectory = self._build_trajectory(action_history)

        # Build structured results block per category
        evaluation = results.get("evaluation", {})
        category = task_data.get("category", "")

        if category == "competitive":
            teams: Dict[str, Any] = {}
            for team_id, prog in evaluation.get("team_progress", {}).items():
                teams[team_id] = {"progress": prog}
            for team_id, status in evaluation.get("team_status", {}).items():
                teams.setdefault(team_id, {})["passed"] = status
            results_block = {"winner": evaluation.get("winner"), "teams": teams}

        elif category == "mixed":
            agents = {
                aid: {"subgoal_passed": passed}
                for aid, passed in evaluation.get("agent_subgoal_status", {}).items()
            }
            results_block = {
                "main_goal": {
                    "passed": evaluation.get("main_goal_success", False),
                    "progress": evaluation.get("main_goal_progress",
                                               evaluation.get("percent_complete", 0.0)),
                },
                "agents": agents,
            }

        else:
            # Cooperative / default
            results_block = {
                "passed": results.get("done", False),
                "progress": evaluation.get("percent_complete", 0.0),
            }

        calibration_entry = {
            "tested_at": datetime.now().isoformat(),
            "agent_models": agent_models,
            "steps": results.get("steps", 0),
            "results": results_block,
            "trajectory": trajectory,
        }

        # Migrate legacy dict format and write as array
        raw_cal = task_data.get("calibration", [])
        calibration = _migrate_legacy_calibration(raw_cal)

        # Deduplicate: replace existing entry with same agent_models, else append
        replaced = False
        for i, existing in enumerate(calibration):
            if existing.get("agent_models") == agent_models:
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
            self._log(f"[Calibration] Saved result: {calibration_entry['results']}, agents={agent_models}")
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
        """Run benchmark test in a subprocess (fresh GL context)."""
        import tempfile

        current_task_num = len(self.submitted_tasks) + 1
        self._test_run_count += 1
        run_dir = self.trajectories_dir / f"task_{current_task_num}" / f"run_{self._test_run_count}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task_data, f)
                temp_task_file = f.name
        except Exception as e:
            return {"steps": 0, "done": False, "error": f"Failed to write temp task file: {e}"}

        num_agents = task_data.get("num_agents", 2)
        cmd = [
            sys.executable, "-m", "emtom.cli.test_task",
            temp_task_file,
            "--working-dir", str(self.working_dir),
            "--trajectory-dir", str(run_dir),
            "--config-name", f"examples/emtom_{num_agents}_robots",
        ]
        if self.test_model:
            cmd.extend(["--test-model", self.test_model])

        # For competitive tasks, build cross-model matchup from test_model
        if task_data.get("category") == "competitive":
            base_model = self.test_model or "gpt-5.2"
            opponent = "sonnet" if base_model != "sonnet" else "gpt-5.2"
            team_model_map_str = f"team_0={base_model},team_1={opponent}"
            cmd.extend(["--team-model-map", team_model_map_str])

            # Store per-agent model mapping for _save_calibration_result
            team_assignment = task_data.get("team_assignment", {})
            agent_models = {}
            team_models = {"team_0": base_model, "team_1": opponent}
            for team_id, agents in team_assignment.items():
                model = team_models.get(team_id, base_model)
                for agent_id in agents:
                    agent_models[agent_id] = model
            if not agent_models:
                # Fallback if no team_assignment
                for i in range(num_agents):
                    agent_models[f"agent_{i}"] = base_model
            self._last_agent_models = agent_models
        else:
            # Non-competitive: all agents use the same model
            model_name = self.test_model or "gpt-5.2"
            self._last_agent_models = {
                f"agent_{i}": model_name for i in range(num_agents)
            }

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200,
            )
            self._cleanup_temp_files(temp_task_file)

            try:
                stdout = proc.stdout
                json_start = stdout.find("{")
                if json_start >= 0:
                    stdout = stdout[json_start:]
                result_data = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                return {"steps": 0, "done": False, "error": f"Failed to parse output: {proc.stderr[:500]}"}

            if result_data.get("success"):
                result = result_data["data"]
            else:
                result = {"steps": 0, "done": False, "error": result_data.get("error", "Unknown error")}

            result["trajectory_dir"] = str(run_dir)

            # Read result.txt if available
            result_txt_path = run_dir / "result.txt"
            if result_txt_path.exists():
                try:
                    with open(result_txt_path) as f:
                        result["result_txt"] = f.read()
                except Exception:
                    pass

            result.pop("planner_traces", None)
            return result

        except subprocess.TimeoutExpired:
            self._cleanup_temp_files(temp_task_file)
            return {"steps": 0, "done": False, "error": "Test timed out", "trajectory_dir": str(run_dir)}
        except Exception as e:
            self._cleanup_temp_files(temp_task_file)
            return {"steps": 0, "done": False, "error": f"Subprocess error: {e}", "trajectory_dir": str(run_dir)}
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

    def _verify_pddl(self) -> str:
        """Verify PDDL goal solvability and compute ToM depth."""
        from emtom.cli.verify_pddl import run

        result = run(str(self.task_file), working_dir=str(self.working_dir))
        if result["success"]:
            return json.dumps(result["data"], indent=2)
        return json.dumps({"valid": False, "error": result["error"]}, indent=2)

    def _verify_golden_trajectory(self) -> str:
        """Regenerate trajectory from PDDL, then execute in subprocess (fresh GL context)."""
        if not self.task_file.exists():
            return json.dumps({"valid": False, "error": "working_task.json does not exist."})

        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return json.dumps({"valid": False, "error": f"Invalid JSON: {e}"})

        # Always regenerate golden trajectory from the authoritative task spec.
        try:
            regen = self._regenerate_golden_trajectory(task_data, source="verify")
            self._log(
                f"Regenerated trajectory: {regen['num_steps']} steps "
                f"(spec_hash={regen['spec_hash'][:8]})"
            )
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": f"Failed to regenerate trajectory from task spec: {e}",
            })

        # Validate structure (after regeneration so golden_trajectory exists)
        validation = self._validate_task_structure(task_data)
        if "error" in validation:
            return json.dumps(validation, indent=2)

        golden = task_data.get("golden_trajectory", [])
        if not golden:
            return json.dumps({"valid": False, "error": "Deterministic planner produced empty trajectory."})

        self._log(f"Verifying golden trajectory: {len(golden)} steps")

        # Static pre-validation
        static_errors = self._static_validate_trajectory(task_data, golden)
        if static_errors:
            return json.dumps({
                "valid": False,
                "error": f"Static validation failed: {static_errors[0]}",
                "all_errors": static_errors,
            })

        num_agents = task_data.get("num_agents", 2)
        cmd = [
            sys.executable, "-m", "emtom.cli.verify_trajectory",
            str(self.task_file),
            "--working-dir", str(self.working_dir),
            "--config-name", f"examples/emtom_{num_agents}_robots",
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200,
            )
            try:
                stdout = proc.stdout
                json_start = stdout.find("{")
                if json_start >= 0:
                    stdout = stdout[json_start:]
                result = json.loads(stdout)
            except (json.JSONDecodeError, ValueError):
                return json.dumps({
                    "valid": False,
                    "error": f"Failed to parse verify output: {proc.stderr[:500]}",
                })

            if result.get("success") and result.get("data", {}).get("valid"):
                self.last_verify_passed = True
            if result.get("data", {}).get("navmesh_issue"):
                result["data"]["recommendation"] = "This scene has navigation issues. Use new_scene[]."

            # Return data portion as JSON for the agent
            if result.get("success"):
                return json.dumps(result["data"], indent=2)

            # On failure, include diagnostic data so the agent can debug.
            fail_resp: Dict[str, Any] = {
                "valid": False,
                "error": result.get("error", "Unknown error"),
            }
            data = result.get("data", {})
            if data.get("evaluation"):
                eval_info = data["evaluation"]
                fail_resp["failure_explanations"] = eval_info.get(
                    "failure_explanations", []
                )
                fail_resp["predicates_achieved"] = eval_info.get(
                    "predicates_achieved", []
                )
                fail_resp["predicates_failed"] = eval_info.get(
                    "predicates_failed", []
                )
            # Show last few executed steps for context
            steps = data.get("executed_steps", [])
            if steps:
                fail_resp["last_steps"] = steps[-3:]
                fail_resp["failed_step"] = data.get("failed_step")
            return json.dumps(fail_resp, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps({
                "valid": False,
                "error": "Verification timed out (20 min). Possible navmesh issue.",
                "hint": "Consider new_scene[].",
            })
        except Exception as e:
            return json.dumps({"valid": False, "error": f"Subprocess error: {e}"})

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
        )

        if not result["success"]:
            return json.dumps({"valid": False, "error": result["error"]}, indent=2)

        data = result["data"]
        self.last_judge_passed = data["passed"]

        # Reconstruct CouncilVerdict for state tracking
        if data["passed"]:
            self.consecutive_tom_failures = 0
            data.pop("suggestions", None)
            data["next_step"] = (
                "Task passed judge. Do NOT change the task design. "
                "Run verify_golden_trajectory[] -> test_task[] -> submit_task[]."
            )
        else:
            self.consecutive_tom_failures += 1
            data["action_required"] = "Modify the task based on suggestions and run judge[] again."
            data["failure_count"] = self.consecutive_tom_failures
            if self.consecutive_tom_failures >= 3:
                data["recommendation"] = (
                    f"You've failed judge {self.consecutive_tom_failures} times. "
                    "Consider new_scene[] for a fresh start."
                )

        self._log(f"[Judge] Result: {'PASS' if data['passed'] else 'FAIL'} "
                   f"(score: {data['overall_score']:.2f}) [failures: {self.consecutive_tom_failures}]")

        return json.dumps(data, indent=2)

    def _submit_task(self) -> str:
        """Copy working task to output directory (requires verify/judge/test gates)."""
        if not self.last_verify_passed:
            return json.dumps({
                "error": "Must run verify_golden_trajectory[] first and pass before submitting.",
                "hint": "Run verify_golden_trajectory[] to prove the golden trajectory works."
            })
        if not self.last_judge_passed:
            return json.dumps({
                "error": "Must run judge[] first and pass before submitting.",
                "hint": "Run judge[] to verify the task quality and ToM requirements.",
                "last_score": self.last_judgment.overall_score if self.last_judgment else None,
                "suggestions": self.last_judgment.suggestions if self.last_judgment else []
            })
        if not self.last_test_passed:
            return json.dumps({
                "error": "Must run test_task[] before submitting (required for calibration data).",
                "hint": "Run test_task[] to benchmark LLM agent performance."
            })

        from emtom.cli.submit_task import run
        result = run(
            str(self.task_file),
            output_dir=str(self.output_dir),
            working_dir=str(self.working_dir),
            submitted_dir=str(self.submitted_tasks_dir),
            subtasks_min=self.subtasks_min,
            subtasks_max=self.subtasks_max,
            agents_min=self.agents_min,
            agents_max=self.agents_max,
        )
        if not result["success"]:
            return json.dumps({"error": result["error"]}, indent=2)

        data = result["data"]
        self.submitted_tasks.append(data["output_path"])

        # Reset verification state for next task
        self.last_verify_passed = False
        self.last_judge_passed = False
        self.last_test_passed = False
        self.last_judgment = None
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

        # Re-inject extra sections (difficulty guidance, query, calibration) that
        # were in the original USER_PROMPT_TEMPLATE but would be lost on context reset
        extra = getattr(self, "_extra_sections", "")

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

                if not result.get("success"):
                    last_error = result.get("error", "Unknown error")
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
            scene_dict = result["data"]["scene_data"]
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
        # Prevent premature give-up: require at least 25% of iteration budget
        min_iterations = max(30, self.iterations_per_task // 4)
        if self.iteration_count < min_iterations:
            return (
                f"Cannot abort yet — only {self.iteration_count}/{min_iterations} "
                f"minimum iterations used. Keep trying: load a new_scene[], "
                f"simplify the task design, or try a different approach. "
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
        lines.append("- `problem_pddl`, `mechanic_bindings`, `locked_containers`: Use EXACT object IDs from this list")
        lines.append("- `task` description: Use NATURAL LANGUAGE (e.g., 'the microwave', 'a toy airplane') - agents use FindObjectTool")
        lines.append("- `agent_secrets`: Use NATURAL LANGUAGE (e.g., 'a drawer in the bedroom') - no object IDs\n")

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
