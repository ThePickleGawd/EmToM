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
