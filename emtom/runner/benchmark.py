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

    Each agent gets an EmtomPlanner (from emtom/benchmark/) that uses
    ReAct-style prompting with exploration-style logging.
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
        """Initialize LLM planner for each agent using our custom LLMPlanner."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Use our EmtomPlanner with custom logging
        from emtom.benchmark import EmtomPlanner

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
                planner_conf['_target_'] = 'emtom.benchmark.emtom_planner.EmtomPlanner'

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

        while self._step_count < max_steps and not done and not self._episode_done:
            # Check turn limit
            if max_turns and turn_count >= max_turns:
                print(f"\n[Benchmark] Reached max LLM turns ({max_turns})", flush=True)
                break

            self._step_count += 1

            world_graph = self.get_world_graph()
            planner_done_count = 0

            for uid, planner in self.planners.items():
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

                    if planner_done:
                        planner_done_count += 1
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
            if planner_done_count == len(self.planners):
                all_planners_done = True
                done = True

            # Get new observations and record
            if not self._episode_done:
                observations = self.get_observations()
                self.record_frame(observations)

        # Evaluate task with PARTNR-style metrics
        evaluation = {}
        if self.game_manager:
            evaluation = self.game_manager.evaluate_task()

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

        # Per-agent secrets (only for ToM tasks)
        secrets = task.agent_secrets.get(agent_id, [])
        if secrets:
            parts.append("\nSecret Knowledge:")
            for s in secrets:
                parts.append(f"- {s}")

        instructions[agent_id] = "\n".join(parts)

    return instructions
