"""
EmtomMultiAgentEnv — Multi-agent RL environment wrapping EmToM benchmark.

Provides a clean reset/step interface over the existing BenchmarkRunner,
LLMPlanner, and GameStateManager stack. Observations are the exact ReAct
prompts that the LLM sees during benchmarking.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from emtom.rl.reward import RewardShaper

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from habitat_llm.agent.env.environment_interface import EnvironmentInterface
    from emtom.task_gen.task_generator import GeneratedTask


class MultiAgentEnv:
    """
    Base class for multi-agent environments (PettingZoo-style API).

    We define the interface ourselves to avoid external dependencies.
    Compatible with PettingZoo parallel API conventions.
    """

    def __init__(self):
        self.possible_agents: List[str] = []
        self.agents: List[str] = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        raise NotImplementedError

    def step(
        self, actions: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        raise NotImplementedError

    def observe(self, agent: str) -> str:
        raise NotImplementedError


class EmtomMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent RL wrapper around the EmToM benchmark stack.

    Wraps BenchmarkRunner + LLMPlanner so that:
    - reset(task) initializes Habitat, creates runner/planners, returns prompts
    - step(actions) parses actions, executes via runner, returns new prompts
    - observe(agent) returns planner.curr_prompt (byte-for-byte identical to benchmark)
    """

    def __init__(
        self,
        config: DictConfig,
        env_interface: EnvironmentInterface,
        task_pool: List[GeneratedTask],
        max_turns: int = 20,
    ):
        super().__init__()
        self.config = config
        self.env_interface = env_interface
        self.task_pool = task_pool
        self.max_turns = max_turns

        # Per-episode state (set during reset)
        self.runner = None
        self.planners: Dict[int, Any] = {}  # uid -> EmtomPlanner
        self.reward_shaper: Optional[RewardShaper] = None
        self.current_task: Optional[GeneratedTask] = None
        self.instruction: Dict[str, str] = {}
        self._turn_count = 0
        self._done = False
        self._task_index = 0

        # Episode tracking
        self._cumulative_rewards: Dict[str, float] = {}
        self._episode_stats: Optional[Dict] = None

    @property
    def last_episode_stats(self) -> Optional[Dict]:
        """Stats from the previous episode (set after reset)."""
        return self._episode_stats

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Reset environment for a new episode.

        Args:
            seed: Random seed (unused for now).
            options: Optional dict with:
                - "task_index": int — index into task_pool
                - "task": GeneratedTask — use this task directly

        Returns:
            (observations, infos) where observations maps agent_id to initial prompt.
        """
        from emtom.runner.benchmark import BenchmarkRunner, task_to_instruction

        # Select task
        options = options or {}
        if "task" in options:
            task = options["task"]
        elif "task_index" in options:
            self._task_index = options["task_index"]
            task = self.task_pool[self._task_index]
        else:
            task = self.task_pool[self._task_index % len(self.task_pool)]
            self._task_index += 1

        self.current_task = task

        # Reset Habitat to the task's episode
        self.env_interface.reset_environment(episode_id=task.episode_id)

        # Create fresh BenchmarkRunner
        self.runner = BenchmarkRunner(self.config)
        self.runner.setup(
            env_interface=self.env_interface,
            task=task,
        )

        # Build per-agent instructions
        self.instruction = task_to_instruction(task)

        # Set up agent list
        agent_ids = sorted(task.agent_actions.keys())
        self.possible_agents = list(agent_ids)
        self.agents = list(agent_ids)

        # Store planner references (created by runner.setup -> _setup_planners)
        self.planners = dict(self.runner.planners)

        # Prepare initial prompts for each agent
        observations = self.env_interface.get_observations()
        world_graph = self.runner.get_world_graph()

        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"
            agent_instruction = self.instruction.get(agent_id, "")
            planner.curr_prompt, planner.params = planner.prepare_prompt(
                agent_instruction, world_graph[uid]
            )
            planner.trace = f"Task: {agent_instruction}\nThought: "

        # Log previous episode stats before resetting
        if self._cumulative_rewards:
            self._episode_stats = {
                "task_id": getattr(self.current_task, "task_id", None) if self.current_task != task else None,
                "step_count": self._turn_count,
                "cumulative_rewards": dict(self._cumulative_rewards),
            }

        # Initialize reward shaper
        self.reward_shaper = RewardShaper(task)
        self._turn_count = 0
        self._done = False
        self._cumulative_rewards = {aid: 0.0 for aid in self.agents}

        obs = {aid: self.observe(aid) for aid in self.agents}
        infos = {
            aid: {"task_id": task.task_id, "turn": 0}
            for aid in self.agents
        }
        return obs, infos

    def observe(self, agent: str) -> str:
        """
        Get the current observation for an agent.

        Returns the exact ReAct prompt string that the planner maintains.
        """
        uid = self._agent_to_uid(agent)
        if uid in self.planners:
            return self.planners[uid].curr_prompt
        return ""

    def step(
        self,
        actions: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one turn of actions for all agents.

        Args:
            actions: Dict mapping agent_id to raw LLM completion text.
                     Expected format: "Thought: ...\nAgent_X_Action: Tool[target]\nAssigned!"

        Returns:
            (observations, rewards, terminations, truncations, infos)
        """
        if self._done:
            empty = {aid: {} for aid in self.agents}
            return (
                {aid: self.observe(aid) for aid in self.agents},
                {aid: 0.0 for aid in self.agents},
                {aid: True for aid in self.agents},
                {aid: False for aid in self.agents},
                empty,
            )

        self._turn_count += 1

        # Parse and execute each agent's action
        action_results = {}
        for agent_id, raw_completion in actions.items():
            uid = self._agent_to_uid(agent_id)
            if uid not in self.planners:
                continue

            planner = self.planners[uid]

            # Append the LLM completion to the prompt (mimics what LLMPlanner does)
            stopword = planner.stopword if hasattr(planner, "stopword") else "Assigned!"
            eot_tag = planner.planner_config.llm.eot_tag if hasattr(planner.planner_config.llm, "eot_tag") else ""
            planner.curr_prompt += f"{raw_completion}\n{stopword}{eot_tag}"
            planner.trace += f"{raw_completion}\n{stopword}{eot_tag}"

            # Parse the action from raw text
            parsed = self._parse_action_text(raw_completion, uid)
            if parsed:
                action_name, target = parsed
                action_results[uid] = (action_name, target)
            else:
                action_results[uid] = None

        # Execute actions via runner
        exec_results = {}
        communicate_results = {}

        for uid, parsed_action in action_results.items():
            if parsed_action is None:
                exec_results[uid] = {
                    "success": False,
                    "observation": "Could not parse action from your response. Use format: Agent_X_Action: ActionName[target]",
                }
                continue

            action_name, target = parsed_action

            if action_name == "Communicate":
                # Handle communication via env_interface
                self._handle_communicate(uid, target)
                communicate_results[uid] = {
                    "success": True,
                    "observation": "Message delivered.",
                }
                continue

            if action_name in ("Done", "Final"):
                exec_results[uid] = {
                    "success": True,
                    "observation": "Episode finished.",
                }
                continue

            if action_name == "Wait":
                exec_results[uid] = {
                    "success": True,
                    "observation": "Waited.",
                }
                continue

            # Execute via runner
            try:
                result = self.runner.execute_action(uid, action_name, target or "")
                exec_results[uid] = result
            except Exception as e:
                exec_results[uid] = {
                    "success": False,
                    "observation": f"Execution error: {e}",
                }

        # Merge results and add observations to prompts
        all_results = {}
        all_results.update(exec_results)
        all_results.update(communicate_results)

        for uid, planner in self.planners.items():
            observation = all_results.get(uid, {}).get("observation", "Action not executed.")
            responses = {uid: observation}
            planner._add_responses_to_prompt(responses)

        # Check subtask completion
        self.runner._check_subtasks()

        # Check task completion
        eval_result = self.runner._check_task_completion()

        # Determine termination
        task_success = eval_result.get("success", False) if eval_result else False
        truncated = self._turn_count >= self.max_turns

        terminal = task_success or truncated
        self._done = terminal

        # Compute rewards
        rewards = self.reward_shaper.compute(eval_result, terminal=terminal)

        # Accumulate per-agent rewards
        for aid, r in rewards.items():
            self._cumulative_rewards[aid] = self._cumulative_rewards.get(aid, 0.0) + r

        # Build return dicts
        observations = {aid: self.observe(aid) for aid in self.agents}
        terminations = {aid: task_success for aid in self.agents}
        truncations = {aid: truncated and not task_success for aid in self.agents}
        infos = {
            aid: {
                "turn": self._turn_count,
                "step_count": self._turn_count,
                "cumulative_reward": self._cumulative_rewards.get(aid, 0.0),
                "eval_result": eval_result,
                "action_result": all_results.get(self._agent_to_uid(aid), {}),
            }
            for aid in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _agent_to_uid(self, agent_id: str) -> int:
        """Convert 'agent_0' to 0."""
        if isinstance(agent_id, int):
            return agent_id
        match = re.match(r"agent_(\d+)", agent_id)
        if match:
            return int(match.group(1))
        return int(agent_id)

    def _parse_action_text(self, text: str, uid: int) -> Optional[Tuple[str, Optional[str]]]:
        """
        Parse action from raw LLM output text.

        Looks for 'Agent_X_Action: ActionName[target]' pattern.
        """
        for line in text.strip().split("\n"):
            line = line.strip()
            if f"Agent_{uid}_Action:" in line or "Action:" in line:
                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue
                action_info = parts[1].strip()

                if "Wait" in action_info:
                    return ("Wait", None)

                if "[" in action_info and "]" in action_info:
                    action_name = action_info.split("[", 1)[0].strip()
                    action_input = action_info.split("[", 1)[1].rstrip("]").strip()
                    return (action_name, action_input if action_input else None)
                else:
                    # No brackets — might be "Done" or malformed
                    action_name = action_info.strip()
                    if action_name:
                        return (action_name, None)

        return None

    def _handle_communicate(self, sender_uid: int, message: Optional[str]) -> None:
        """Handle Communicate action by posting message to env_interface."""
        if not message:
            return

        valid_uids = list(self.env_interface.agent_uids) if self.env_interface.agent_uids else list(self.env_interface.world_graph.keys())
        msg_text, target_uids = self.env_interface.parse_communicate_args(message, valid_uids)

        self.env_interface.post_agent_message(
            sender_uid, msg_text, target_uids=target_uids
        )
