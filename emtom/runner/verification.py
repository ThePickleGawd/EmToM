"""
Verification runner for EMTOM golden trajectory testing.

This runner is designed for step-by-step trajectory verification without
the overhead of LLM planners. It only provides execute_action() and
evaluate_task() capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import EMTOMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from emtom.task_gen import GeneratedTask


class VerificationRunner(EMTOMBaseRunner):
    """
    Simple runner for trajectory verification.

    Unlike BenchmarkRunner, this doesn't create LLM planners.
    Just provides execute_action() and evaluate_task() for step-by-step verification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.task: Optional["GeneratedTask"] = None
        self._llm_client = None

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task: Optional["GeneratedTask"] = None,
        save_video: Optional[bool] = None,
    ) -> None:
        """
        Setup verification runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics/bindings
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

        # Call parent setup (no planners created)
        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video)

        # Setup LLM for perception tools (FindObjectTool, etc.)
        self._setup_llm_for_tools()

    def _setup_llm_for_tools(self) -> None:
        """Setup LLM client for perception tools (FindObjectTool, etc.)."""
        try:
            from habitat_llm.llm import instantiate_llm

            # Use gpt-5 for perception tools
            self._llm_client = instantiate_llm("openai_chat", model="gpt-5")

            # Pass LLM to agent tools
            for uid, agent in self.agents.items():
                if hasattr(agent, 'pass_llm_to_tools'):
                    agent.pass_llm_to_tools(self._llm_client)
        except Exception as e:
            # Log but don't fail - some trajectories may not need perception tools
            print(f"[VerificationRunner] Warning: Could not setup LLM for tools: {e}")

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

    def run(self, **kwargs) -> Dict[str, Any]:
        """Not used for verification - actions are executed individually."""
        return {"error": "Use execute_action() directly for verification"}

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion using PARTNR-style predicates.

        If no success_condition is provided, derives from task's terminal subtasks.
        """
        # Try to derive success_condition from task's terminal subtasks
        if success_condition is None and self.task:
            effective_condition = self.task.get_effective_success_condition()
            if effective_condition:
                success_condition = {
                    "description": effective_condition.description,
                    "required_states": effective_condition.required_states,
                }

        # Fall back to parent implementation
        return super().evaluate_task(success_condition)
