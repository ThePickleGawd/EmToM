"""
RewardShaper — per-step reward computation from EmToM evaluation results.

Handles three task categories:
- Cooperative: shared delta(percent_complete) + terminal bonus
- Competitive: +1 winner / -1 loser at episode end
- Mixed: shared main goal progress + individual subgoal bonus
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.task_gen.task_generator import GeneratedTask


class RewardShaper:
    """Compute per-agent rewards from evaluation results."""

    STEP_COST = -0.01
    TERMINAL_BONUS = 1.0
    COMPETITIVE_WIN = 1.0
    COMPETITIVE_LOSE = -1.0
    SUBGOAL_BONUS = 0.3

    def __init__(self, task: GeneratedTask):
        self.task = task
        self.category = getattr(task, "category", "cooperative")
        self._prev_percent: float = 0.0
        self._prev_subgoal_status: Dict[str, bool] = {}
        self._agent_ids = sorted(task.agent_actions.keys())

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._prev_percent = 0.0
        self._prev_subgoal_status = {}

    def compute(
        self,
        eval_result: Optional[Dict[str, Any]],
        terminal: bool = False,
    ) -> Dict[str, float]:
        """
        Compute per-agent rewards from evaluation result.

        Args:
            eval_result: Result from BenchmarkRunner._check_task_completion()
                         or None if no evaluation available yet.
            terminal: Whether this is the final step of the episode.

        Returns:
            Dict mapping agent_id (e.g. "agent_0") to reward float.
        """
        if self.category == "cooperative":
            return self._cooperative_reward(eval_result, terminal)
        elif self.category == "competitive":
            return self._competitive_reward(eval_result, terminal)
        elif self.category == "mixed":
            return self._mixed_reward(eval_result, terminal)
        else:
            return self._cooperative_reward(eval_result, terminal)

    def _cooperative_reward(
        self,
        eval_result: Optional[Dict[str, Any]],
        terminal: bool,
    ) -> Dict[str, float]:
        """Shared reward: +delta(percent_complete) + terminal bonus + step cost."""
        reward = self.STEP_COST

        if eval_result:
            percent = eval_result.get(
                "percent_required_complete",
                eval_result.get("percent_complete", 0.0),
            )
            delta = percent - self._prev_percent
            self._prev_percent = percent
            reward += delta

            if terminal and eval_result.get("success", False):
                reward += self.TERMINAL_BONUS

        # All agents share the same reward
        return {aid: reward for aid in self._agent_ids}

    def _competitive_reward(
        self,
        eval_result: Optional[Dict[str, Any]],
        terminal: bool,
    ) -> Dict[str, float]:
        """Win/lose at episode end. Step cost during episode."""
        if not terminal or not eval_result:
            return {aid: self.STEP_COST for aid in self._agent_ids}

        winner = eval_result.get("winner")
        teams = getattr(self.task, "teams", None) or {}

        rewards = {}
        for aid in self._agent_ids:
            if not winner:
                # Draw
                rewards[aid] = 0.0
            else:
                # Find which team this agent is on
                agent_team = None
                for team_id, members in teams.items():
                    if aid in members:
                        agent_team = team_id
                        break
                if agent_team == winner:
                    rewards[aid] = self.COMPETITIVE_WIN
                elif agent_team is not None:
                    rewards[aid] = self.COMPETITIVE_LOSE
                else:
                    rewards[aid] = 0.0

        return rewards

    def _mixed_reward(
        self,
        eval_result: Optional[Dict[str, Any]],
        terminal: bool,
    ) -> Dict[str, float]:
        """Shared main goal progress + individual subgoal bonus."""
        reward_base = self.STEP_COST

        if eval_result:
            percent = eval_result.get(
                "percent_required_complete",
                eval_result.get("main_goal_progress", 0.0),
            )
            delta = percent - self._prev_percent
            self._prev_percent = percent
            reward_base += delta

            if terminal and eval_result.get(
                "success", eval_result.get("main_goal_success", False)
            ):
                reward_base += self.TERMINAL_BONUS

        rewards = {aid: reward_base for aid in self._agent_ids}

        # Add individual subgoal bonuses
        if eval_result:
            subgoal_status = eval_result.get("agent_subgoal_status", {})
            for aid, completed in subgoal_status.items():
                if aid in rewards and completed and not self._prev_subgoal_status.get(aid, False):
                    rewards[aid] += self.SUBGOAL_BONUS
            self._prev_subgoal_status = dict(subgoal_status)

        return rewards
