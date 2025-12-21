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

    Each agent gets an LLMPlanner that uses ReAct-style prompting
    to solve the given task.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.planners: Dict[int, Any] = {}  # uid -> LLMPlanner
        self.task: Optional["GeneratedTask"] = None

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task: Optional["GeneratedTask"] = None,
    ) -> None:
        """
        Setup benchmark runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics/bindings (can be from task.to_mechanics_dict())
            output_dir: Output directory
            task: Optional GeneratedTask object for full task info
        """
        self.task = task

        # If task provided but no task_data, convert task to mechanics format
        if task and not task_data:
            task_data = self._task_to_mechanics_dict(task)

        # Get agent_actions from task if available
        agent_actions = task.agent_actions if task else None

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions)
        self._setup_planners()

    def _task_to_mechanics_dict(self, task: "GeneratedTask") -> Dict[str, Any]:
        """Convert GeneratedTask to mechanics initialization format."""
        if task.mechanic_bindings:
            return {
                "mechanics": [
                    {"mechanic_type": b.mechanic_type, **b.to_dict()}
                    for b in task.mechanic_bindings
                ]
            }
        return {"mechanics": []}

    def _setup_planners(self) -> None:
        """Initialize LLM planner for each agent."""
        from hydra.utils import instantiate

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

            # Use Hydra's instantiate pattern (same as DecentralizedEvaluationRunner)
            planner = instantiate(agent_conf.planner)
            planner = planner(env_interface=self.env_interface)
            planner.agents = [agent]
            self.planners[uid] = planner
            print(f"[BenchmarkRunner] Created planner for agent_{uid}")

    def run(
        self,
        instruction: Dict[str, str],
        max_steps: int = 200,
    ) -> Dict[str, Any]:
        """
        Run benchmark task with LLM planners.

        Args:
            instruction: Per-agent instruction dict (agent_id -> instruction)
            max_steps: Maximum simulation steps

        Returns:
            Results dict with steps, success, history
        """
        print(f"\n[BenchmarkRunner] Starting benchmark execution...")
        print(f"[BenchmarkRunner] Max steps: {max_steps}")

        observations = self.get_observations()
        self.record_frame(observations)

        done = False
        all_planners_done = False

        while self._step_count < max_steps and not done and not self._episode_done:
            self._step_count += 1

            if self._step_count % 10 == 0:
                print(f"[BenchmarkRunner] Step {self._step_count}/{max_steps}")

            world_graph = self.get_world_graph()
            planner_done_count = 0

            for uid, planner in self.planners.items():
                agent_id = f"agent_{uid}"
                agent_instruction = instruction.get(agent_id, instruction.get(str(uid), ""))

                try:
                    # Get action from planner
                    low_level_actions, planner_info, planner_done = planner.get_next_action(
                        agent_instruction, observations, world_graph
                    )

                    if planner_done:
                        planner_done_count += 1
                        print(f"[BenchmarkRunner] {agent_id} planner indicates done")

                    # Record high-level action in history
                    high_level_action = self._extract_high_level_action(planner_info, uid)
                    if high_level_action:
                        self._action_history.append({
                            "step": self._step_count,
                            "agent": agent_id,
                            "action": high_level_action,
                        })

                except AssertionError as e:
                    if "Episode over" in str(e) or "call reset before calling step" in str(e):
                        print(f"[BenchmarkRunner] Episode ended at step {self._step_count}")
                        self._episode_done = True
                        break
                    raise
                except Exception as e:
                    print(f"[BenchmarkRunner] Error during planner execution for {agent_id}: {e}")
                    continue

            # Check if all planners are done
            if planner_done_count == len(self.planners):
                all_planners_done = True
                done = True

            # Get new observations and record
            if not self._episode_done:
                observations = self.get_observations()
                self.record_frame(observations)

        # Save outputs
        self._save_outputs(instruction)

        return {
            "steps": self._step_count,
            "done": done or all_planners_done,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
        }

    def _extract_high_level_action(self, planner_info: Dict, uid: int) -> Optional[str]:
        """Extract high-level action string from planner info."""
        try:
            if "high_level_action" in planner_info:
                ha = planner_info["high_level_action"]
                if isinstance(ha, dict) and uid in ha:
                    action_tuple = ha[uid]
                    if action_tuple and len(action_tuple) >= 2:
                        return f"{action_tuple[0]}[{action_tuple[1]}]"

            for key in ["action", "actions", f"agent_{uid}"]:
                if key in planner_info:
                    val = planner_info[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, tuple) and len(val) >= 2:
                        return f"{val[0]}[{val[1]}]"
        except Exception:
            pass
        return None

    def _save_outputs(self, instruction: Dict[str, str]) -> None:
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

        # Per-agent secrets (only for ToM tasks)
        secrets = task.agent_secrets.get(agent_id, [])
        if secrets:
            parts.append("\nSecret Knowledge:")
            for s in secrets:
                parts.append(f"- {s}")

        if task.theory_of_mind_required:
            parts.append("\nUse Communicate[message] to coordinate with your teammate.")

        instructions[agent_id] = "\n".join(parts)

    return instructions
