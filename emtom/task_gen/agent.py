"""
Agentic task generator for EMTOM benchmark.

A ReAct-style agent with 3 tools:
- bash: Shell commands for file exploration and task editing
- test_task: Run benchmark with current working_task.json
- submit_task: Save current task to output directory
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .judge import Judge, Judgment, CouncilVerdict, Colors
from .diversity import DiversityTracker
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

        # Create reference_tasks directory with simple planning examples
        self.reference_tasks_dir = self.working_dir / "reference_tasks"
        self._sample_reference_tasks()

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
            default_actions = ["Navigate", "Open", "Pick", "Place", "UseItem", "Communicate", "Wait"]
            template["agent_secrets"] = {
                f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"]
                for i in range(self.agents_max)
            }
            template["agent_actions"] = {
                f"agent_{i}": default_actions.copy()
                for i in range(self.agents_max)
            }
            template["golden_trajectory"] = [
                {
                    "actions": [
                        {"agent": f"agent_{i}", "action": "ACTION_NAME[TARGET]" if i == 0 else "Wait"}
                        for i in range(self.agents_max)
                    ]
                }
            ]
            with open(self.template_file, 'w') as f:
                json.dump(template, f, indent=2)

        # Track state
        self.submitted_tasks: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.iteration_count = 0
        self.last_verify_passed = False  # Track if golden trajectory verified
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

    def _sample_reference_tasks(self) -> None:
        """Sample simple planning tasks for reference into reference_tasks directory."""
        from emtom.task_gen.sample_partnr import sample_planning_tasks_to_directory

        # Only sample if directory doesn't exist or is empty
        if self.reference_tasks_dir.exists() and any(self.reference_tasks_dir.iterdir()):
            return

        sample_planning_tasks_to_directory(
            output_dir=self.reference_tasks_dir,
            num_samples=10,
            seed=42,
            dataset='train',
            verbose=self.verbose,
        )

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

        # Initialize conversation with action descriptions and paths injected
        replacements = {
            "{action_descriptions}": ActionRegistry.get_all_action_descriptions(),
            "{task_file}": str(self.task_file),
            "{working_dir}": str(self.working_dir),
            "{available_items}": available_items,
            "{available_mechanics}": available_mechanics,
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
                    "- Use 0-1 mechanics (prefer none, or a single simple one like room_restriction)\n"
                    "- Avoid inverse_state and chained conditional_unlock — models cannot discover these\n"
                    "- 2-3 agents with clear roles\n"
                    "- 2-3 subtasks maximum\n"
                    "- Secrets MUST explain any active mechanic in plain language\n"
                    "- All effects should be observable (no remote effects in unseen rooms)\n"
                    "- tom_level 1 only\n"
                ),
                "medium": (
                    "## Difficulty: MEDIUM\n"
                    "Generate moderately complex tasks:\n"
                    "- 1-2 mechanics with hints in secrets\n"
                    "- 2-4 agents, meaningful coordination required\n"
                    "- 3-4 subtasks with some dependencies\n"
                    "- tom_level 1-2\n"
                ),
                "hard": (
                    "## Difficulty: HARD\n"
                    "Generate challenging tasks for top-tier models:\n"
                    "- 2+ mechanics, complex interactions\n"
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
        extra_sections = query_section + seed_section + verification_section + calibration_section
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
            except Exception as e:
                self._log(f"LLM error: {e}")
                self.messages.append({
                    "role": "user",
                    "content": f"Error getting LLM response: {e}. Please try again."
                })
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
            default_actions = ["Navigate", "Open", "Pick", "Place", "UseItem", "FindObjectTool", "FindReceptacleTool", "FindRoomTool", "Communicate", "Wait"]
            task["agent_secrets"] = {
                f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"]
                for i in range(num_agents)
            }
            task["agent_actions"] = {
                f"agent_{i}": default_actions.copy()
                for i in range(num_agents)
            }

            # Generate golden_trajectory template
            task["golden_trajectory"] = [
                {
                    "actions": [
                        {"agent": f"agent_{i}", "action": "ACTION_NAME[TARGET]" if i == 0 else "Wait"}
                        for i in range(num_agents)
                    ]
                }
            ]

        # Clear task_id so a new one gets assigned on submit
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
                self.last_test_passed = False
            return self._bash(args)
        elif tool == "test_task":
            return self._test_task()
        elif tool == "verify_golden_trajectory":
            return self._verify_golden_trajectory()
        elif tool == "judge":
            return self._judge()
        elif tool == "submit_task":
            return self._submit_task()
        elif tool == "new_scene":
            return self._new_scene(args)
        elif tool == "fail":
            return self._fail(args)
        else:
            return f"Unknown tool: {tool}. Available: bash, test_task, verify_golden_trajectory, judge, submit_task, new_scene, fail"

    def _bash(self, command: str) -> str:
        """
        Execute a shell command with safety limits.

        Allows file exploration and editing within allowed directories only.
        Allows command chaining (&&, ||, ;, |) but validates each sub-command.
        """
        # NOTE: Do NOT blindly replace \n → newline here. LLM outputs already
        # contain real newlines for line breaks. Literal \n sequences inside
        # code strings (e.g. Python "\n".join(...)) must stay as-is, otherwise
        # they become actual newlines and cause SyntaxError: EOL.

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
        """Save calibration results to task JSON for dataset pass rate tracking."""
        from datetime import datetime

        # Get model name from LLM config
        model_name = "unknown"
        if hasattr(self.llm, 'llm_conf'):
            params = self.llm.llm_conf.get('generation_params', {})
            model_name = params.get('model', 'unknown')
        elif hasattr(self.llm, 'generation_params'):
            model_name = getattr(self.llm.generation_params, 'model', 'unknown')

        # Build trajectory from action history
        action_history = results.get("action_history", [])
        trajectory = self._build_trajectory(action_history)

        # Extract calibration data from results
        calibration_entry = {
            "passed": results.get("done", False),
            "tested_at": datetime.now().isoformat(),
            "steps": results.get("steps", 0),
            "percent_complete": results.get("evaluation", {}).get("percent_complete", 0.0),
            "trajectory": trajectory,
        }

        # Update task data with calibration results
        if "calibration" not in task_data:
            task_data["calibration"] = {}
        task_data["calibration"][model_name] = calibration_entry

        # Write back to working_task.json
        try:
            with open(self.task_file, 'w') as f:
                json.dump(task_data, f, indent=2)
            self._log(f"[Calibration] Saved result for {model_name}: passed={calibration_entry['passed']}")
        except Exception as e:
            self._log(f"[Calibration] Warning: Failed to save calibration result: {e}")

    def _validate_task_structure(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task JSON structure without running benchmark."""
        # Core required fields (success_condition is optional if subtasks have valid DAG)
        required_fields = [
            "task_id", "title", "task", "episode_id",
            "mechanic_bindings", "agent_secrets",
            "agent_actions"
        ]

        missing = [f for f in required_fields if f not in task_data]
        if missing:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing}",
                "summary": "Task validation failed"
            }

        # Validate episode_id matches the loaded scene
        if self.scene_data:
            expected_episode = self.scene_data.episode_id
            task_episode = task_data.get("episode_id", "")
            if task_episode != expected_episode:
                return {
                    "valid": False,
                    "error": f"episode_id must be '{expected_episode}' (from loaded scene), got '{task_episode}'",
                    "summary": "Task validation failed - wrong episode"
                }

        # Collect defined item IDs from task
        defined_items = set()
        for item in task_data.get("items", []):
            item_id = item.get("item_id")
            if item_id:
                defined_items.add(item_id)

        # Validate object IDs in golden_trajectory exist in scene or are custom items
        if self.scene_data and task_data.get("golden_trajectory"):
            all_valid_ids = set(
                self.scene_data.rooms +
                self.scene_data.furniture +
                self.scene_data.objects
            )
            # Also include custom items from task definition
            all_valid_ids.update(defined_items)

            invalid_ids = []
            invalid_items = []
            for step in task_data["golden_trajectory"]:
                actions = step.get("actions", [])
                for action_entry in actions:
                    # Parse PARTNR-style action: "Navigate[table_22]" -> args = "table_22"
                    action_str = action_entry.get("action", "")
                    import re
                    # Allow empty bracket args for commands like Wait[]
                    match = re.match(r'(\w+)(?:\[(.*)\])?$', action_str)
                    if not match:
                        continue
                    action_name, args = match.group(1), match.group(2)

                    # Skip actions without args or communication/wait
                    if not args or action_name in ["Communicate", "Wait"]:
                        continue

                    # For Place/UseItem, args is comma-separated - check all parts
                    # Place format: Place[obj, relation, target, constraint, ref]
                    # Skip relation/constraint keywords (from habitat_llm/world_model/entities/furniture.py)
                    place_keywords = {"on", "within", "next_to"}
                    parts = [p.strip() for p in args.split(",")]
                    for target_id in parts:
                        if not target_id or target_id == "None" or target_id in place_keywords:
                            continue
                        # Check item_ prefixed IDs against defined items
                        if target_id.startswith("item_"):
                            if target_id not in defined_items:
                                invalid_items.append(target_id)
                        elif target_id not in all_valid_ids:
                            invalid_ids.append(target_id)

            if invalid_ids:
                return {
                    "valid": False,
                    "error": f"golden_trajectory contains invalid object IDs not in scene: {list(set(invalid_ids))}",
                    "summary": "Task validation failed - invalid object IDs"
                }
            if invalid_items:
                return {
                    "valid": False,
                    "error": f"golden_trajectory references undefined items: {list(set(invalid_items))}. Defined items: {list(defined_items)}",
                    "summary": "Task validation failed - item ID mismatch"
                }

        # Validate success condition OR subtasks DAG
        has_success_condition = "success_condition" in task_data and task_data["success_condition"]
        has_subtasks = "subtasks" in task_data and task_data["subtasks"]

        if not has_success_condition and not has_subtasks:
            return {
                "valid": False,
                "error": "Task must have either 'success_condition' or 'subtasks' with valid DAG",
                "summary": "Task validation failed"
            }

        # Validate success-condition predicates use supported schema.
        from emtom.evaluation import PARTNR_PREDICATES, EMTOM_PREDICATES
        from emtom.state.manager import GameStateManager
        supported_predicates = PARTNR_PREDICATES | EMTOM_PREDICATES | GameStateManager.GAME_STATE_PREDICATES

        def _check_supported_predicate(condition: Dict[str, Any], scope: str) -> Optional[Dict[str, Any]]:
            prop = condition.get("property")
            if not prop:
                return {
                    "valid": False,
                    "error": f"{scope} missing 'property' field in success_condition",
                    "summary": "Task validation failed - invalid success_condition schema",
                }
            if prop not in supported_predicates:
                return {
                    "valid": False,
                    "error": (
                        f"{scope} uses unsupported predicate '{prop}'. "
                        f"Supported predicates: {sorted(supported_predicates)}"
                    ),
                    "summary": "Task validation failed - unsupported predicate",
                }
            return None

        top_level_condition = task_data.get("success_condition")
        if isinstance(top_level_condition, dict):
            for idx, cond in enumerate(top_level_condition.get("required_states", [])):
                if isinstance(cond, dict):
                    error = _check_supported_predicate(cond, f"success_condition.required_states[{idx}]")
                    if error:
                        return error

        # If using subtasks, validate DAG structure
        if has_subtasks:
            from emtom.task_gen.dag import validate_dag
            from emtom.task_gen import Subtask

            subtasks = []
            for s in task_data["subtasks"]:
                if isinstance(s, dict):
                    subtasks.append(Subtask.from_dict(s))

            is_valid, errors = validate_dag(subtasks)
            if not is_valid:
                return {
                    "valid": False,
                    "error": f"Invalid subtask DAG: {'; '.join(errors)}",
                    "summary": "Task validation failed"
                }

            # Validate at least one required subtask exists
            required_subtasks = [s for s in subtasks if getattr(s, 'required', True) is True]
            if not required_subtasks:
                return {
                    "valid": False,
                    "error": "At least one subtask must have 'required: true' for task success",
                    "summary": "Task validation failed - no required subtasks"
                }

            # Validate item IDs in subtask success conditions
            invalid_items_in_subtasks = []
            for s in task_data["subtasks"]:
                if isinstance(s, dict):
                    condition = s.get("success_condition", {})
                    if isinstance(condition, dict):
                        # Check has_item conditions
                        if condition.get("property") == "has_item":
                            target_item = condition.get("target") or condition.get("value")
                            if target_item and target_item.startswith("item_"):
                                if target_item not in defined_items:
                                    invalid_items_in_subtasks.append(target_item)

            if invalid_items_in_subtasks:
                return {
                    "valid": False,
                    "error": f"subtasks reference undefined items: {list(set(invalid_items_in_subtasks))}. Defined items: {list(defined_items)}",
                    "summary": "Task validation failed - item ID mismatch in subtasks"
                }

            for i, s in enumerate(task_data["subtasks"]):
                if isinstance(s, dict):
                    condition = s.get("success_condition", {})
                    if isinstance(condition, dict):
                        error = _check_supported_predicate(condition, f"subtasks[{i}]")
                        if error:
                            return error

        # Validate locked_containers references
        locked_containers = task_data.get("locked_containers", {})
        if locked_containers and isinstance(locked_containers, dict):
            # Keys should be valid articulated furniture (must be openable)
            if self.scene_data:
                all_articulated = set(self.scene_data.articulated_furniture)
                invalid_containers = [c for c in locked_containers.keys() if c not in all_articulated]
                if invalid_containers:
                    return {
                        "valid": False,
                        "error": f"locked_containers must reference articulated furniture (openable containers). Invalid: {invalid_containers}",
                        "summary": "Task validation failed - invalid locked container"
                    }
            # Values should be defined items
            invalid_key_items = [v for v in locked_containers.values() if v not in defined_items]
            if invalid_key_items:
                return {
                    "valid": False,
                    "error": f"locked_containers references undefined key items: {invalid_key_items}. Defined items: {list(defined_items)}",
                    "summary": "Task validation failed - invalid key item"
                }

        # Validate items inside references are articulated furniture (Open[] only)
        for item in task_data.get("items", []):
            # Backward-compat: accept legacy hidden_in from older tasks.
            inside = item.get("inside") or item.get("hidden_in")
            if inside and self.scene_data:
                all_articulated = set(self.scene_data.articulated_furniture)
                if inside not in all_articulated:
                    return {
                        "valid": False,
                        "error": f"item '{item.get('item_id')}' has inside='{inside}' which is not articulated/openable furniture",
                        "summary": "Task validation failed - invalid inside container"
                    }

        # Validate agent IDs are consistent with num_agents
        num_agents = task_data.get("num_agents", 2)
        valid_agent_ids = {f"agent_{i}" for i in range(num_agents)}

        # Check agent_actions keys
        for agent_id in task_data.get("agent_actions", {}).keys():
            if agent_id not in valid_agent_ids:
                return {
                    "valid": False,
                    "error": f"agent_actions contains invalid agent ID '{agent_id}'. Valid: {sorted(valid_agent_ids)} (num_agents={num_agents})",
                    "summary": "Task validation failed - invalid agent ID"
                }

        # Check agent_secrets keys
        for agent_id in task_data.get("agent_secrets", {}).keys():
            if agent_id not in valid_agent_ids:
                return {
                    "valid": False,
                    "error": f"agent_secrets contains invalid agent ID '{agent_id}'. Valid: {sorted(valid_agent_ids)} (num_agents={num_agents})",
                    "summary": "Task validation failed - invalid agent ID"
                }

        # Check subtask success_condition entity references valid agents
        for subtask in task_data.get("subtasks", []):
            if not isinstance(subtask, dict):
                continue
            sc = subtask.get("success_condition", {})
            if not isinstance(sc, dict):
                continue
            entity = sc.get("entity", "")
            if not isinstance(entity, str):
                entity = str(entity)
            if entity.startswith("agent_") and entity not in valid_agent_ids:
                return {
                    "valid": False,
                    "error": f"subtask '{subtask.get('id')}' references invalid agent '{entity}'. Valid: {sorted(valid_agent_ids)}",
                    "summary": "Task validation failed - invalid agent in subtask"
                }

        # Check task description is not empty
        if not task_data.get("task") or len(task_data.get("task", "")) < 20:
            return {
                "valid": False,
                "error": "task field must be at least 20 characters",
                "summary": "Task validation failed"
            }

        # Validate object IDs in task description (if any) exist in scene
        # Note: Task descriptions should use natural language (e.g., "the microwave in the kitchen")
        # and agents use FindObjectTool to resolve to actual IDs at runtime.
        # We only validate that any IDs mentioned actually exist (to catch typos).
        task_desc = task_data.get("task", "")
        import re
        object_pattern = r'\b[a-z_]+_\d+\b'
        object_refs = re.findall(object_pattern, task_desc)

        # Check that any object IDs in task actually exist in scene
        if self.scene_data:
            valid_scene_ids = set(
                self.scene_data.rooms +
                self.scene_data.furniture +
                self.scene_data.objects
            )
            # Also allow item_ prefixed IDs (custom items defined in task)
            defined_items = {item.get("item_id") for item in task_data.get("items", []) if item.get("item_id")}
            valid_scene_ids.update(defined_items)

            invalid_task_refs = [ref for ref in object_refs if ref not in valid_scene_ids and not ref.startswith(("item_", "agent_", "team_"))]
            if invalid_task_refs:
                return {
                    "valid": False,
                    "error": f"task references objects that don't exist in scene: {invalid_task_refs}. Use only: {list(self.scene_data.objects)[:10]}...",
                    "summary": "Task validation failed - invented object IDs"
                }

            # Also check agent_secrets for invented object IDs
            for agent_id, secrets in task_data.get("agent_secrets", {}).items():
                for secret in secrets:
                    secret_refs = re.findall(object_pattern, secret)
                    invalid_secret_refs = [ref for ref in secret_refs if ref not in valid_scene_ids and not ref.startswith(("item_", "agent_", "team_"))]
                    if invalid_secret_refs:
                        return {
                            "valid": False,
                            "error": f"agent_secrets[{agent_id}] references objects that don't exist in scene: {invalid_secret_refs}",
                            "summary": "Task validation failed - invented object IDs in secrets"
                        }

            # Check subtask success_conditions for invented object IDs
            for subtask in task_data.get("subtasks", []):
                if not isinstance(subtask, dict):
                    continue
                sc = subtask.get("success_condition", {})
                if not isinstance(sc, dict):
                    continue
                for field in ["entity", "target"]:
                    val = sc.get(field, "")
                    if val and isinstance(val, str) and not val.startswith("agent_"):
                        val_refs = re.findall(object_pattern, val)
                        invalid_sc_refs = [ref for ref in val_refs if ref not in valid_scene_ids and not ref.startswith("item_")]
                        if invalid_sc_refs:
                            return {
                                "valid": False,
                                "error": f"subtask '{subtask.get('id')}' success_condition references objects that don't exist: {invalid_sc_refs}",
                                "summary": "Task validation failed - invented object IDs in subtasks"
                            }

        # Check mechanic_bindings structure
        # Mechanics that require trigger_object vs those that use other keys
        TRIGGER_OBJECT_MECHANICS = {"inverse_state", "remote_control", "conditional_unlock", "state_mirroring"}
        for i, binding in enumerate(task_data.get("mechanic_bindings", [])):
            if "mechanic_type" not in binding:
                return {
                    "valid": False,
                    "error": f"mechanic_bindings[{i}] missing mechanic_type",
                    "summary": "Task validation failed"
                }
            # Only require trigger_object for mechanics that use it
            mechanic_type = binding.get("mechanic_type", "")
            if mechanic_type in TRIGGER_OBJECT_MECHANICS and "trigger_object" not in binding:
                return {
                    "valid": False,
                    "error": f"mechanic_bindings[{i}] ({mechanic_type}) missing trigger_object",
                    "summary": "Task validation failed"
                }

        # Check agent_secrets has proper structure
        if not isinstance(task_data.get("agent_secrets"), dict):
            return {
                "valid": False,
                "error": "agent_secrets must be a dict",
                "summary": "Task validation failed"
            }

        # Try to parse as GeneratedTask
        try:
            from emtom.task_gen import GeneratedTask
            task = GeneratedTask.from_dict(task_data)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to parse as GeneratedTask: {e}",
                "summary": "Task validation failed"
            }

        return {
            "valid": True,
            "task_id": task_data.get("task_id"),
            "title": task_data.get("title"),
            "mechanics": [b.get("mechanic_type") for b in task_data.get("mechanic_bindings", [])],
            "tom_required": task_data.get("theory_of_mind_required", False),
            "summary": "Task structure is valid"
        }

    def _static_validate_trajectory(self, task_data: Dict[str, Any], golden: List[Dict]) -> List[str]:
        """
        Fast static validation of golden trajectory before running simulation.

        Catches common errors without expensive Habitat initialization:
        - Invalid object/furniture IDs
        - Invalid action names
        - Missing agents in trajectory steps
        - Malformed action syntax

        Returns list of error messages (empty if valid).
        """
        errors = []
        num_agents = task_data.get("num_agents", 2)
        valid_agents = {f"agent_{i}" for i in range(num_agents)}

        # Valid action names
        valid_actions = {
            "Navigate", "Open", "Close", "Pick", "Place",
            "UseItem", "Communicate", "Wait", "Clean", "Pour", "PowerOn", "PowerOff",
            "Fill", "FindObjectTool", "FindReceptacleTool", "FindRoomTool"
        }

        # Build set of valid scene IDs
        valid_ids = set()
        if self.scene_data:
            valid_ids.update(self.scene_data.rooms)
            valid_ids.update(self.scene_data.furniture)
            valid_ids.update(self.scene_data.objects)

        # Add defined items
        defined_items = {item.get("item_id") for item in task_data.get("items", []) if item.get("item_id")}
        valid_ids.update(defined_items)

        # Check each step
        for step_idx, step in enumerate(golden):
            actions = step.get("actions", [])
            if not actions:
                errors.append(f"Step {step_idx}: No actions array")
                continue

            agents_in_step = set()
            for action_entry in actions:
                agent = action_entry.get("agent", "")
                action_str = action_entry.get("action", "")

                # Check agent ID
                if agent not in valid_agents:
                    errors.append(f"Step {step_idx}: Invalid agent '{agent}' (valid: {sorted(valid_agents)})")
                agents_in_step.add(agent)

                # Parse action
                # Allow empty bracket args for commands like Wait[]
                match = re.match(r'(\w+)(?:\[(.*)\])?$', action_str)
                if not match:
                    errors.append(f"Step {step_idx}: Malformed action '{action_str}'")
                    continue

                action_name, args = match.group(1), match.group(2)

                # Check action name
                if action_name not in valid_actions:
                    errors.append(f"Step {step_idx}: Unknown action '{action_name}'")

                # Skip ID validation for actions without targets
                if action_name in ("Wait", "Communicate") or not args:
                    continue

                # Check object IDs in args (skip None, relation keywords)
                skip_words = {"on", "within", "next_to", "None", ""}
                parts = [p.strip() for p in args.split(",")]
                for part in parts:
                    if part in skip_words:
                        continue
                    # Check if it's a valid ID
                    if valid_ids and part not in valid_ids:
                        # Only error for non-item IDs (items might be dynamically created)
                        if not part.startswith("item_"):
                            errors.append(f"Step {step_idx}: Unknown object '{part}' in {action_str}")

            # Check all agents have an action (warn only, not error)
            missing_agents = valid_agents - agents_in_step
            if missing_agents:
                errors.append(f"Step {step_idx}: Missing actions for {sorted(missing_agents)} (add Wait if idle)")

        return errors[:10]  # Limit to first 10 errors

    def _run_benchmark(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the actual benchmark test in a subprocess.

        Uses subprocess to get a fresh GL context (avoids context corruption
        when running multiple tests in the same process).

        Saves agent trajectories to agent_trajectories/run_N/ directory.
        Returns result with trajectory_dir path and result.txt content included.
        """
        import tempfile

        # Current task number (1-indexed: task being worked on, not yet submitted)
        current_task_num = len(self.submitted_tasks) + 1

        # Increment run counter and create task-scoped trajectory directory
        self._test_run_count += 1
        run_dir = self.trajectories_dir / f"task_{current_task_num}" / f"run_{self._test_run_count}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write task to temp file for subprocess
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task_data, f)
                temp_task_file = f.name
            # Also create a temp file for results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_result_file = f.name
        except Exception as e:
            return {
                "steps": 0,
                "done": False,
                "error": f"Failed to write temp task file: {e}",
                "summary": f"Setup error: {e}"
            }

        # Run test in subprocess with trajectory output directory
        num_agents = task_data.get("num_agents", 2)
        config_name = f"examples/emtom_{num_agents}_robots"
        script_path = Path(__file__).parent / "test_task.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--task-file", temp_task_file,
            "--result-file", temp_result_file,
            "--trajectory-dir", str(run_dir),
            "--config-name", config_name,
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # Suppress C++ simulator spam (navmesh errors etc.)
                text=True,
                timeout=1200,  # 20 minute timeout for LLM agents
            )

            # Read result from file (cleaner than parsing stdout)
            try:
                with open(temp_result_file) as f:
                    result = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                result = {
                    "steps": 0,
                    "done": False,
                    "error": f"Failed to read result file: {e}",
                    "summary": "Result file error"
                }

            self._cleanup_temp_files(temp_task_file, temp_result_file)

            # Add trajectory directory path to result
            result["trajectory_dir"] = str(run_dir)

            # Read result.txt content and include in observation (saves agent a turn)
            result_txt_path = run_dir / "result.txt"
            if result_txt_path.exists():
                try:
                    with open(result_txt_path) as f:
                        result["result_txt"] = f.read()
                except Exception:
                    pass

            # Remove full planner traces from result (agent reads files instead via bash)
            # Keep action_history for calibration tracking - it will be removed after saving
            result.pop("planner_traces", None)

            return result

        except subprocess.TimeoutExpired:
            self._cleanup_temp_files(temp_task_file, temp_result_file)
            return {
                "steps": 0,
                "done": False,
                "error": "Test timed out after 10 minutes",
                "summary": "Timeout",
                "trajectory_dir": str(run_dir)
            }
        except Exception as e:
            self._cleanup_temp_files(temp_task_file, temp_result_file)
            return {
                "steps": 0,
                "done": False,
                "error": f"Subprocess error: {e}",
                "summary": f"Subprocess error: {e}",
                "trajectory_dir": str(run_dir)
            }

    def _verify_golden_trajectory(self) -> str:
        """
        Execute golden_trajectory step-by-step to prove task is completable.

        Runs verification in a subprocess to get a fresh GL context
        (avoids OpenGL context issues when reusing the main process).
        """
        import subprocess
        import tempfile

        # Check task file exists
        if not self.task_file.exists():
            return json.dumps({
                "valid": False,
                "error": "working_task.json does not exist. Create it first with bash."
            })

        # Load task to check golden_trajectory exists
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return json.dumps({
                "valid": False,
                "error": f"Invalid JSON in working_task.json: {e}"
            })

        # Validate task structure before expensive simulation
        validation = self._validate_task_structure(task_data)
        if "error" in validation:
            return json.dumps(validation, indent=2)

        golden = task_data.get("golden_trajectory", [])
        if not golden:
            return json.dumps({
                "valid": False,
                "error": "No golden_trajectory found in task. Add a golden_trajectory field with the expected action sequence."
            })

        self._log(f"Verifying golden trajectory: {len(golden)} steps")

        # Fast static pre-validation (catches errors before expensive simulation)
        static_errors = self._static_validate_trajectory(task_data, golden)
        if static_errors:
            return json.dumps({
                "valid": False,
                "error": f"Static validation failed: {static_errors[0]}",
                "all_errors": static_errors,
                "hint": "Fix these errors before running simulation."
            })

        # Create temp file for results
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_result_file = f.name
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": f"Failed to create temp result file: {e}"
            })

        # Run verification in subprocess (fresh GL context)
        num_agents = task_data.get("num_agents", 2)
        config_name = f"examples/emtom_{num_agents}_robots"
        script_path = Path(__file__).parent / "verify_trajectory.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--task-file", str(self.task_file),
            "--result-file", temp_result_file,
            "--working-dir", str(self.working_dir),
            "--config-name", config_name,
        ]

        try:
            timeout = 1200  # 20 minutes - golden trajectory verification
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # Suppress C++ simulator spam (navmesh errors etc.)
                text=True,
                timeout=timeout,
            )

            # Read result from file (cleaner than parsing stdout)
            try:
                with open(temp_result_file) as f:
                    result_data = json.load(f)
                if result_data.get("valid", False):
                    self.last_verify_passed = True
                # Add hint for navmesh issues
                if result_data.get("navmesh_issue", False):
                    result_data["recommendation"] = "This scene has navigation issues. Use new_scene[] to get a different scene."
            except (FileNotFoundError, json.JSONDecodeError) as e:
                result_data = {
                    "valid": False,
                    "error": f"Failed to read result file: {e}",
                    "summary": "Result file error"
                }

            self._cleanup_temp_files(temp_result_file)

            return json.dumps(result_data, indent=2)

        except subprocess.TimeoutExpired:
            self._cleanup_temp_files(temp_result_file)
            return json.dumps({
                "valid": False,
                "error": f"Verification timed out after {timeout} seconds. This often indicates navmesh/PathFinder issues with the scene.",
                "hint": "The scene may have navigation problems. Consider using new_scene[] to get a different scene.",
                "summary": "Timeout - possible navmesh issue"
            })
        except Exception as e:
            self._cleanup_temp_files(temp_result_file)
            return json.dumps({
                "valid": False,
                "error": f"Verification subprocess error: {e}",
                "summary": f"Subprocess error: {e}"
            })

    def _judge(self) -> str:
        """
        Evaluate task quality using multi-model council (Claude Opus + GPT-5).

        Scores task on 8 criteria:
        - ToM: information asymmetry, interdependence, mental state reasoning, coordination
        - Quality: narrative consistency, subtask relevance, mechanic utilization, trajectory efficiency

        Both models must agree for task to pass.
        Returns detailed feedback and suggestions for improvement.
        Saves JSON output to log_dir/judgments/ directory.
        """
        if not self.task_file.exists():
            return json.dumps({
                "valid": False,
                "error": "working_task.json does not exist. Create and test it first."
            })

        # Load task
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return json.dumps({
                "valid": False,
                "error": f"Invalid JSON: {e}"
            })

        # Validate task structure before expensive LLM calls
        validation = self._validate_task_structure(task_data)
        if "error" in validation:
            return json.dumps(validation, indent=2)

        self._log("[Judge] Evaluating task with council...")

        # Find latest trajectory dir for this task if available
        trajectory_dir = None
        task_dirs = sorted(self.trajectories_dir.glob("task_*"), key=lambda p: p.name)
        if task_dirs:
            latest_task = task_dirs[-1]
            run_dirs = sorted(latest_task.glob("run_*"), key=lambda p: p.name)
            if run_dirs:
                trajectory_dir = run_dirs[-1]
                self._log(f"[Judge] Including rollout data from: {trajectory_dir}")

        # Run evaluation with council
        try:
            verdict = self.judge.evaluate(
                task_data,
                scene_data=self.scene_data,
                trajectory_dir=trajectory_dir,
            )
            self.last_judgment = verdict
            self.last_judge_passed = verdict.passed

            # Format result for agent feedback
            result = {
                "valid": verdict.passed,
                "overall_score": verdict.overall_score,
                "threshold": self.judge.overall_threshold,
                "models": list(verdict.judgments.keys()),
                "model_results": {
                    model: {
                        "passed": j.is_valid,
                        "score": j.overall_score,
                    }
                    for model, j in verdict.judgments.items()
                },
                "suggestions": verdict.suggestions,
            }

            if verdict.disagreements:
                result["disagreements"] = verdict.disagreements

            if verdict.passed:
                # On PASS: remove suggestions to prevent agent from editing the task
                # and invalidating the judge pass. Direct to next step instead.
                result.pop("suggestions", None)
                result["summary"] = f"PASS - All models agree task is valid (score: {verdict.overall_score:.2f})"
                result["next_step"] = (
                    "Task passed judge. Do NOT change the task design. "
                    "If golden_trajectory is missing, add it now, then run "
                    "verify_golden_trajectory[] → test_task[] → submit_task[]."
                )
                self.consecutive_tom_failures = 0  # Reset on success
            else:
                self.consecutive_tom_failures += 1
                result["summary"] = f"FAIL - Task did not pass council (score: {verdict.overall_score:.2f})"
                result["action_required"] = "Modify the task based on suggestions and run judge[] again."
                result["failure_count"] = self.consecutive_tom_failures

                # After 3+ failures, suggest considering new_scene
                if self.consecutive_tom_failures >= 3:
                    result["recommendation"] = (
                        f"You've failed judge {self.consecutive_tom_failures} times on this task. "
                        "Consider whether the core design can actually work, or if a fresh start would be faster. "
                        "Otherwise, new_scene[] gives you a fresh scene to try a different approach."
                    )

            # Save JSON to judgments directory
            judgments_dir = self.log_dir / "judgments"
            judgments_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            attempt_num = self.consecutive_tom_failures if not verdict.passed else self.consecutive_tom_failures + 1
            judgment_file = judgments_dir / f"judgment_{timestamp}_attempt{attempt_num}.json"
            with open(judgment_file, "w") as f:
                json.dump(verdict.to_dict(), f, indent=2)

            # Print colored output to stderr
            status = "PASS" if verdict.passed else "FAIL"
            color = Colors.GREEN if verdict.passed else Colors.RED
            print(f"\n{Colors.BOLD}{Colors.CYAN}=== Task Evaluation (Council) - attempt {attempt_num} ==={Colors.RESET}", file=sys.stderr)
            print(f"{Colors.BOLD}{color}{status}{Colors.RESET} - Score: {verdict.overall_score:.2f} (threshold: {self.judge.overall_threshold})", file=sys.stderr)
            print(f"Models: {', '.join(verdict.judgments.keys())}", file=sys.stderr)
            print(f"Saved to: {Colors.CYAN}{judgment_file}{Colors.RESET}", file=sys.stderr)

            if verdict.disagreements:
                print(f"\n{Colors.YELLOW}Model disagreements:{Colors.RESET}", file=sys.stderr)
                for d in verdict.disagreements:
                    print(f"  - {d}", file=sys.stderr)

            if not verdict.passed and verdict.suggestions:
                print(f"\n{Colors.YELLOW}Suggestions for improvement:{Colors.RESET}", file=sys.stderr)
                for i, suggestion in enumerate(verdict.suggestions, 1):
                    print(f"  {i}. {suggestion}", file=sys.stderr)

            self._log(f"[Judge] Result: {status} (score: {verdict.overall_score:.2f}) [failures: {self.consecutive_tom_failures}]")

            return json.dumps(result, indent=2)

        except Exception as e:
            self._log(f"[Judge] Error: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                "valid": False,
                "error": f"Evaluation failed: {e}",
                "summary": "Evaluation error - try again"
            })

    def _submit_task(self) -> str:
        """
        Copy working task to output directory AND submitted_tasks/.

        Called when the agent determines task quality is good.
        Requires verify_golden_trajectory[], judge[], and test_task[] to pass first.
        """
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
                "hint": "Run test_task[] to benchmark LLM agent performance and record pass/fail for dataset calibration."
            })

        if not self.task_file.exists():
            return "Error: working_task.json does not exist. Create and test it first."

        # Load task to validate
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        # Run structure validation (catches invented object IDs, bad mechanics, etc.)
        validation_result = self._validate_task_structure(task_data)
        if "error" in validation_result:
            return json.dumps(validation_result, indent=2)

        # Validate subtask count
        subtasks = task_data.get("subtasks", [])
        num_subtasks = len(subtasks)
        if num_subtasks < self.subtasks_min:
            return json.dumps({
                "error": f"Task has {num_subtasks} subtasks, minimum is {self.subtasks_min}.",
                "hint": f"Add more subtasks to reach at least {self.subtasks_min}.",
                "current": num_subtasks,
                "required_min": self.subtasks_min,
            })
        if num_subtasks > self.subtasks_max:
            return json.dumps({
                "error": f"Task has {num_subtasks} subtasks, maximum is {self.subtasks_max}.",
                "hint": f"Reduce subtasks to at most {self.subtasks_max}, or consolidate steps.",
                "current": num_subtasks,
                "required_max": self.subtasks_max,
            })

        # Validate agent count
        num_agents = task_data.get("num_agents", 2)
        if num_agents < self.agents_min:
            return json.dumps({
                "error": f"Task has {num_agents} agents, minimum is {self.agents_min}.",
                "hint": f"Increase num_agents to at least {self.agents_min}.",
                "current": num_agents,
                "required_min": self.agents_min,
            })
        if num_agents > self.agents_max:
            return json.dumps({
                "error": f"Task has {num_agents} agents, maximum is {self.agents_max}.",
                "hint": f"Reduce num_agents to at most {self.agents_max}.",
                "current": num_agents,
                "required_max": self.agents_max,
            })

        # Generate filename: {datetime}_{title_slug}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        title = task_data.get("title", "untitled")
        # Slugify title: lowercase, replace spaces with underscores, keep only alphanumeric and underscores
        title_slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:50]
        output_filename = f"{timestamp}_{title_slug}.json"
        output_path = self.output_dir / output_filename

        # Copy to main output directory (permanent storage)
        shutil.copy(self.task_file, output_path)

        # Also copy to submitted_tasks/ for this session's tracking
        submitted_path = self.submitted_tasks_dir / output_filename
        shutil.copy(self.task_file, submitted_path)

        self.submitted_tasks.append(str(output_path))

        # Reset verification state for next task
        self.last_verify_passed = False
        self.last_judge_passed = False
        self.last_test_passed = False
        self.last_judgment = None
        self.consecutive_tom_failures = 0  # Reset failure counter for new task

        # Extract what to remember from this task and reset context
        task_title = task_data.get("title", "untitled")
        memory = self._extract_task_memory(task_title)
        self.task_memories.append(memory)

        # Track task pattern for diversity (persists across runs)
        pattern = self.diversity_tracker.add_pattern(output_filename, task_data)
        self._log(f"Diversity pattern added: {pattern}")

        # Re-categorize all tasks and refresh diversity in prompt
        try:
            from emtom.scripts.categorize_tasks import categorize_tasks
            categories_file = Path("data/emtom/task_categories.json")
            categorize_tasks(self.llm, self.output_dir, categories_file)
            self._refresh_diversity_in_prompt()
            self._log("Task categories updated and diversity prompt refreshed")
        except Exception as e:
            self._log(f"Warning: Failed to re-categorize tasks: {e}")

        self._reset_context_for_next_task()

        return f"Task '{task_title}' saved!\n  - {output_path} (permanent)\n  - {submitted_path} (session)\nTotal submitted: {len(self.submitted_tasks)}\n\n[Context reset for next task. Your learnings have been preserved.]\n\nFor next task: Use new_scene[] for a fresh scene, or modify working_task.json to build on a previous task from submitted_tasks/."

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
        """
        Load a scene for task generation.

        Usage:
            new_scene[N] - Load random scene with N agents, reset task
            new_scene[N, keep] - Same scene with N agents, preserve task edits

        Uses subprocess to get a fresh GL context (avoids sensor registry conflicts).
        """
        import tempfile
        from emtom.task_gen.scene_loader import SceneData
        from habitat_llm.utils import get_random_seed

        # Parse arguments
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

        # Check for 'keep' mode
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
                # Use subprocess to load scene (fresh GL context)
                config_name = f"examples/emtom_{num_agents}_robots"
                new_seed = get_random_seed()
                script_path = Path(__file__).parent / "load_scene.py"

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    result_file = f.name

                cmd = [
                    sys.executable,
                    str(script_path),
                    "--result-file", result_file,
                    "--working-dir", str(self.working_dir),
                    "--config-name", config_name,
                    "--seed", str(new_seed),
                ]
                if scene_id:
                    cmd.extend(["--scene-id", scene_id])

                self._log(f"Running (attempt {attempt}/{max_scene_retries}): {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes for scene loading
                )

                # Read result
                try:
                    with open(result_file) as f:
                        result_data = json.load(f)
                finally:
                    Path(result_file).unlink(missing_ok=True)

                if not result_data.get("success"):
                    last_error = result_data.get("error", "Unknown error")
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
            scene_dict = result_data["scene_data"]
            self.scene_data = SceneData(
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

            self._log(f"Loaded scene {self.scene_data.scene_id} (episode {self.scene_data.episode_id})")
            self._log(f"  Rooms: {len(self.scene_data.rooms)}, Furniture: {len(self.scene_data.furniture)}, Objects: {len(self.scene_data.objects)}")

            # Save new scene data to working directory
            scene_file = self.working_dir / "current_scene.json"
            with open(scene_file, "w") as f:
                json.dump(self.scene_data.to_dict(), f, indent=2)

            # Reset verification state (task structure changed)
            self.last_verify_passed = False
            self.last_judge_passed = False
            self.last_test_passed = False
            self.last_judgment = None
            self.consecutive_tom_failures = 0

            if keep_mode:
                # Keep mode: preserve working_task.json, update num_agents and spawns
                task_file = self.working_dir / "working_task.json"
                if task_file.exists():
                    with open(task_file) as f:
                        task_data = json.load(f)
                    task_data["num_agents"] = num_agents
                    # Update spawn positions (recalculated for new agent count)
                    if self.scene_data.agent_spawns:
                        task_data["agent_spawns"] = self.scene_data.agent_spawns
                    with open(task_file, "w") as f:
                        json.dump(task_data, f, indent=2)
                    self._log(f"Updated num_agents to {num_agents} and spawns in working_task.json")
                else:
                    # No task file exists, create from template
                    self._create_working_task_from_template(num_agents=num_agents)

                return f"""Scene reloaded with {num_agents} agents. Task preserved.

Scene ID: {self.scene_data.scene_id}
Agents: {num_agents}
Rooms: {', '.join(self.scene_data.rooms)}

working_task.json preserved (num_agents updated to {num_agents}).
Context preserved. Verification flags reset - run judge[] again."""
            else:
                # Fresh start: reset working_task.json and context
                self._create_working_task_from_template(num_agents=num_agents)
                self._reset_context_for_next_task()

                return f"""New scene loaded! Context refreshed.

Scene ID: {self.scene_data.scene_id}
Episode ID: {self.scene_data.episode_id}
Agents: {num_agents}
Rooms: {', '.join(self.scene_data.rooms)}
Furniture: {len(self.scene_data.furniture)} items
Objects: {len(self.scene_data.objects)} items

working_task.json reset. Use new_scene[N, keep] to change agent count without losing work."""

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
        """Format scene data for the LLM prompt."""
        if not self.scene_data:
            return "No scene data available."

        lines = []

        # Guidance about IDs vs natural language
        lines.append("**ID Usage Rules:**")
        lines.append("- `golden_trajectory`, `subtasks`, `locked_containers`: Use EXACT object IDs from this list")
        lines.append("- `task` description: Use NATURAL LANGUAGE (e.g., 'the microwave', 'a toy airplane') - agents use FindObjectTool")
        lines.append("- `agent_secrets`: Use NATURAL LANGUAGE (e.g., 'a drawer in the bedroom') - no object IDs\n")

        # Rooms - show all
        lines.append("### Rooms")
        for room in self.scene_data.rooms:
            lines.append(f"  - {room}")

        # Articulated furniture (can open/close - good for mechanics) - show all
        lines.append("\n### Articulated Furniture (can open/close)")
        for furn in self.scene_data.articulated_furniture:
            lines.append(f"  - {furn}")

        # Other furniture - show all
        other_furniture = [f for f in self.scene_data.furniture
                          if f not in self.scene_data.articulated_furniture]
        lines.append("\n### Other Furniture (tables, counters, etc.)")
        for furn in other_furniture:
            lines.append(f"  - {furn}")

        # Objects with their locations - show all
        lines.append("\n### Objects (can be picked up and placed)")
        # Build reverse mapping: object -> furniture it's on
        obj_locations = {}
        for furn, objs in self.scene_data.objects_on_furniture.items():
            for obj in objs:
                obj_locations[obj] = furn

        for obj in self.scene_data.objects:
            location = obj_locations.get(obj)
            if location:
                lines.append(f"  - {obj} (on {location})")
            # Skip objects without a known location — they resolve to
            # "unknown" at runtime and cause agent navigation failures.

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
