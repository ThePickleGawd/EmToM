"""
Benchmark runner for EMTOM evaluation.

Uses LLMPlanner for multi-agent task execution with proper video recording
and planner logging.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .base import EMTOMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from emtom.task_gen import GeneratedTask


class BenchmarkRunner(EMTOMBaseRunner):
    """
    Runner for benchmark evaluation with LLM planners.

    Each agent gets an EmtomPlanner (from emtom/planner.py) that uses
    ReAct-style prompting with exploration-style logging.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.planners: Dict[int, Any] = {}  # uid -> LLMPlanner
        self.task: Optional["GeneratedTask"] = None
        self._completed_subtasks: set = set()  # Track completed subtask IDs

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task: Optional["GeneratedTask"] = None,
        save_video: Optional[bool] = None,
    ) -> None:
        """
        Setup benchmark runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics/bindings (can be from task.to_mechanics_dict())
            output_dir: Output directory
            task: Optional GeneratedTask object for full task info
            save_video: Whether to save video. If None, uses config.evaluation.save_video
        """
        self.task = task

        # If task provided but no task_data, convert task to mechanics format
        if task and not task_data:
            task_data = self._task_to_mechanics_dict(task)

        # Get agent_actions from task if available
        agent_actions = task.agent_actions if task else None

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video)
        self._setup_planners()

    def _task_to_mechanics_dict(self, task: "GeneratedTask") -> Dict[str, Any]:
        """Convert GeneratedTask to task data for GameStateManager initialization."""
        result = {}
        if task.mechanic_bindings:
            result["mechanics"] = [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        if task.items:
            result["items"] = task.items  # Already list of dicts
        if task.locked_containers:
            result["locked_containers"] = task.locked_containers
        return result

    def _setup_planners(self) -> None:
        """Initialize LLM planner for each agent using our custom LLMPlanner."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Use our EmtomPlanner with custom logging
        from emtom.planner import EmtomPlanner

        if not hasattr(self.config, 'evaluation') or not hasattr(self.config.evaluation, 'agents'):
            print("[BenchmarkRunner] Warning: No agents in config for planner setup")
            return

        agent_confs = list(self.config.evaluation.agents.values())

        for uid, agent in self.agents.items():
            if uid >= len(agent_confs):
                continue

            agent_conf = agent_confs[uid]
            if not hasattr(agent_conf, 'planner'):
                print(f"[BenchmarkRunner] Warning: No planner config for agent_{uid}")
                continue

            # Override _target_ to use our EmtomPlanner
            planner_conf = OmegaConf.to_container(agent_conf.planner, resolve=True)
            if '_target_' in planner_conf and 'LLMPlanner' in planner_conf['_target_']:
                # Use our custom EmtomPlanner
                planner_conf['_target_'] = 'emtom.planner.EmtomPlanner'

            planner = instantiate(OmegaConf.create(planner_conf))
            planner = planner(env_interface=self.env_interface)
            planner.agents = [agent]
            self.planners[uid] = planner
            print(f"[BenchmarkRunner] Created planner for agent_{uid}")

    def run(
        self,
        instruction: Dict[str, str],
        max_steps: int = 20000,
        max_turns: int = 20,
    ) -> Dict[str, Any]:
        """
        Run benchmark task with LLM planners.

        Args:
            instruction: Per-agent instruction dict (agent_id -> instruction)
            max_steps: Maximum simulation steps (safety limit)
            max_turns: Maximum LLM turns per agent (default: 20)

        Returns:
            Results dict with steps, turns, success, history
        """
        n_agents = len(self.planners)
        task_title = self.task.title if self.task else "Unknown Task"

        print(f"\n{'='*60}", flush=True)
        print(f"BENCHMARK: {task_title}", flush=True)
        print(f"Agents: {n_agents} | Max turns: {max_turns}", flush=True)
        print(f"{'='*60}\n", flush=True)

        observations = self.get_observations()
        self.record_frame(observations)

        done = False
        all_planners_done = False
        turn_count = 0
        planners_done: set = set()  # Track which planners have finished

        while self._step_count < max_steps and not done and not self._episode_done:
            # Check turn limit
            if max_turns and turn_count >= max_turns:
                print(f"\n[Benchmark] Reached max LLM turns ({max_turns})", flush=True)
                break

            self._step_count += 1

            world_graph = self.get_world_graph()

            for uid, planner in self.planners.items():
                # Skip planners that are already done
                if uid in planners_done:
                    continue

                agent_id = f"agent_{uid}"
                agent_instruction = instruction.get(agent_id, instruction.get(str(uid), ""))

                try:
                    # Get action from planner (logging is handled by emtom_planner.py)
                    low_level_actions, planner_info, planner_done = planner.get_next_action(
                        agent_instruction, observations, world_graph
                    )

                    # Track high-level action in history
                    high_level_action = self._extract_high_level_action(planner_info, uid)
                    if high_level_action:
                        turn_count += 1
                        self._action_history.append({
                            "step": self._step_count,
                            "turn": turn_count,
                            "agent": agent_id,
                            "action": high_level_action,
                        })

                        # Print thought + action + observation for debugging
                        thought = planner_info.get("thought", "")
                        response = planner_info.get("response", "")
                        if thought:
                            print(f"Thought: {thought}", flush=True)
                        print(f"Agent_{uid}: {high_level_action}", flush=True)
                        if response:
                            print(f"  → {response}", flush=True)

                        # Check for subtask completion after each action
                        self._check_subtasks()

                    if planner_done:
                        planners_done.add(uid)
                        print(f"[Agent {uid} DONE]", flush=True)

                except AssertionError as e:
                    if "Episode over" in str(e) or "call reset before calling step" in str(e):
                        print(f"\n[Benchmark] Episode ended at step {self._step_count}", flush=True)
                        self._episode_done = True
                        break
                    raise
                except Exception as e:
                    print(f"[Agent {uid} ERROR] {e}", flush=True)
                    continue

            # Check if all planners are done
            if len(planners_done) == len(self.planners):
                all_planners_done = True
                done = True

            # Get new observations and record
            if not self._episode_done:
                observations = self.get_observations()
                self.record_frame(observations)

        # Evaluate task with PARTNR-style metrics
        evaluation = {}
        if self.game_manager:
            # Derive success condition from task's subtasks if available
            success_condition = None
            if self.task:
                effective = self.task.get_effective_success_condition()
                if effective:
                    success_condition = {
                        "description": effective.description,
                        "required_states": effective.required_states,
                    }
            evaluation = self.game_manager.evaluate_task(success_condition)

        # Save outputs
        self._save_outputs(instruction, evaluation)

        return {
            "steps": self._step_count,
            "turns": turn_count,
            "done": done or all_planners_done,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
            "evaluation": evaluation,
        }

    def _check_subtasks(self) -> List[str]:
        """
        Check all subtasks and return list of newly completed subtask IDs.

        Logs completion to console when a subtask is newly completed.
        """
        if not self.task or not self.task.subtasks:
            return []

        subtasks = self.task.subtasks
        newly_completed = []

        for subtask in subtasks:
            subtask_id = subtask.id if hasattr(subtask, 'id') else subtask.get("id", "")
            if not subtask_id or subtask_id in self._completed_subtasks:
                continue

            # Check if all dependencies are completed first
            if hasattr(subtask, 'depends_on'):
                depends_on = subtask.depends_on
            else:
                depends_on = subtask.get("depends_on", [])
            if not all(dep in self._completed_subtasks for dep in depends_on):
                # Dependencies not met, skip this subtask
                continue

            # Get success condition
            if hasattr(subtask, 'success_condition'):
                success_condition = subtask.success_condition
            else:
                success_condition = subtask.get("success_condition")

            if not success_condition:
                continue

            # Check this subtask's condition
            result = self.evaluate_task({"required_states": [success_condition]})
            if result and result.get("success"):
                self._completed_subtasks.add(subtask_id)
                newly_completed.append(subtask_id)

                # Get description
                if hasattr(subtask, 'description'):
                    desc = subtask.description
                else:
                    desc = subtask.get("description", subtask_id)

                # Log the completion
                total = len(subtasks)
                print(f"\n{'─'*50}", flush=True)
                print(f"✓ SUBTASK COMPLETE: {subtask_id}", flush=True)
                print(f"  {desc}", flush=True)
                print(f"  Progress: {len(self._completed_subtasks)}/{total} subtasks", flush=True)
                print(f"{'─'*50}", flush=True)

        return newly_completed

    def _extract_high_level_action(self, planner_info: Dict, uid: int) -> Optional[str]:
        """Extract high-level action string from planner info.

        LLMPlanner sets planner_info["high_level_actions"] = Dict[uid, (action_name, action_arg, ...)]
        """
        ha = planner_info.get("high_level_actions", {})
        if uid in ha:
            action_tuple = ha[uid]
            if action_tuple and action_tuple[0]:
                return f"{action_tuple[0]}[{action_tuple[1]}]"
        return None

    def _save_outputs(
        self,
        instruction: Dict[str, str],
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save video and planner log."""
        # Save video
        task_id = self.task.task_id if self.task else "unknown"
        video_suffix = f"benchmark_{task_id}_{self._step_count}"
        self.save_video(video_suffix)

        # Build planner log
        log_data = {
            "task_id": task_id,
            "task_title": self.task.title if self.task else "Unknown",
            "instruction": instruction,
            "mechanics_active": self.game_manager.get_state().active_mechanics if self.game_manager else [],
            "total_steps": self._step_count,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
        }

        # Include PARTNR-style evaluation metrics
        if evaluation:
            log_data["evaluation"] = evaluation
            log_data["percent_complete"] = evaluation.get("percent_complete", 0.0)
            log_data["success"] = evaluation.get("success", False)
            log_data["failure_explanations"] = evaluation.get("failure_explanations", [])

        # Include mechanic bindings if from task
        if self.task and self.task.mechanic_bindings:
            log_data["mechanic_bindings"] = [
                b.to_dict() for b in self.task.mechanic_bindings
            ]

        self.save_planner_log(log_data)

        # Save prompts from planners
        self._save_planner_prompts()

    def _save_planner_prompts(self) -> None:
        """Save prompts from each LLM planner."""
        prompts = {}
        traces = {}

        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"

            # Get prompt from planner
            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                prompts[agent_id] = planner.curr_prompt

                # Extract trace (interaction part after system prompt)
                prompt = planner.curr_prompt
                task_marker = "Task:"
                if task_marker in prompt:
                    trace_start = prompt.find(task_marker)
                    traces[agent_id] = prompt[trace_start:]
                else:
                    traces[agent_id] = prompt

        if prompts:
            self.save_prompts(prompts, traces)

    def get_planner_traces(self) -> Dict[str, str]:
        """Get the conversation traces from each planner for debugging."""
        traces = {}
        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"
            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                # Extract trace (interaction part after system prompt)
                prompt = planner.curr_prompt
                task_marker = "Task:"
                if task_marker in prompt:
                    trace_start = prompt.find(task_marker)
                    traces[agent_id] = prompt[trace_start:]
                else:
                    traces[agent_id] = prompt
        return traces


def task_to_instruction(task: "GeneratedTask") -> Dict[str, str]:
    """
    Convert a GeneratedTask to per-agent instructions.

    Args:
        task: GeneratedTask object

    Returns:
        Dict mapping agent_id -> instruction string
    """
    instructions = {}

    for agent_id in task.agent_roles.keys():
        parts = []

        # Add atmospheric story first (sets the scene)
        if task.story:
            parts.append(task.story)
            parts.append("")  # blank line

        parts.append(f"Goal: {task.public_goal}")

        if task.public_context:
            parts.append(task.public_context)

        # Per-agent actions available
        actions = task.agent_actions.get(agent_id, [])
        if actions:
            parts.append(f"\nYour available actions: {', '.join(actions)}")
            # Add hints about special actions
            if "Search" in actions:
                parts.append("(Search finds hidden items and adds them to your inventory)")
            if "UseItem" in actions:
                parts.append("(UseItem lets you use items from inventory, e.g., UseItem[item_key_1, cabinet] to unlock)")

        # Per-agent secrets (only for ToM tasks)
        secrets = task.agent_secrets.get(agent_id, [])
        if secrets:
            parts.append("\nSecret Knowledge:")
            for s in secrets:
                parts.append(f"- {s}")

        instructions[agent_id] = "\n".join(parts)

    return instructions
