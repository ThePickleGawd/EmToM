"""Unit tests for RewardShaper — no Habitat needed."""

import pytest
from unittest.mock import MagicMock
from emtom.rl.reward import RewardShaper


def _make_task(category="cooperative", num_agents=2, teams=None):
    """Create a mock GeneratedTask."""
    task = MagicMock()
    task.category = category
    task.agent_actions = {f"agent_{i}": ["Navigate", "Pick"] for i in range(num_agents)}
    task.teams = teams
    return task


class TestCooperativeReward:
    def test_step_cost_only(self):
        task = _make_task()
        shaper = RewardShaper(task)
        rewards = shaper.compute(None, terminal=False)
        assert rewards == {"agent_0": -0.01, "agent_1": -0.01}

    def test_progress_delta(self):
        task = _make_task()
        shaper = RewardShaper(task)

        # 50% complete
        eval_result = {"percent_required_complete": 0.5, "success": False}
        rewards = shaper.compute(eval_result, terminal=False)
        expected = -0.01 + 0.5  # step cost + delta
        assert abs(rewards["agent_0"] - expected) < 1e-6
        assert abs(rewards["agent_1"] - expected) < 1e-6

        # Still 50% — no delta
        rewards = shaper.compute(eval_result, terminal=False)
        assert abs(rewards["agent_0"] - (-0.01)) < 1e-6

    def test_terminal_bonus(self):
        task = _make_task()
        shaper = RewardShaper(task)

        eval_result = {"percent_required_complete": 1.0, "success": True}
        rewards = shaper.compute(eval_result, terminal=True)
        expected = -0.01 + 1.0 + 1.0  # step cost + delta + terminal bonus
        assert abs(rewards["agent_0"] - expected) < 1e-6

    def test_shared_reward(self):
        task = _make_task(num_agents=3)
        shaper = RewardShaper(task)

        eval_result = {"percent_required_complete": 0.3, "success": False}
        rewards = shaper.compute(eval_result, terminal=False)
        assert rewards["agent_0"] == rewards["agent_1"] == rewards["agent_2"]

    def test_reset(self):
        task = _make_task()
        shaper = RewardShaper(task)

        shaper.compute({"percent_required_complete": 0.5, "success": False})
        shaper.reset()

        # After reset, delta should be from 0 again
        rewards = shaper.compute({"percent_required_complete": 0.5, "success": False})
        expected = -0.01 + 0.5
        assert abs(rewards["agent_0"] - expected) < 1e-6


class TestCompetitiveReward:
    def test_step_cost_during_episode(self):
        task = _make_task(category="competitive")
        shaper = RewardShaper(task)
        rewards = shaper.compute(None, terminal=False)
        assert rewards == {"agent_0": -0.01, "agent_1": -0.01}

    def test_winner_loser(self):
        teams = {"team_0": ["agent_0"], "team_1": ["agent_1"]}
        task = _make_task(category="competitive", teams=teams)
        shaper = RewardShaper(task)

        eval_result = {"winner": "team_0"}
        rewards = shaper.compute(eval_result, terminal=True)
        assert rewards["agent_0"] == 1.0
        assert rewards["agent_1"] == -1.0

    def test_draw(self):
        teams = {"team_0": ["agent_0"], "team_1": ["agent_1"]}
        task = _make_task(category="competitive", teams=teams)
        shaper = RewardShaper(task)

        eval_result = {"winner": None}
        rewards = shaper.compute(eval_result, terminal=True)
        assert rewards["agent_0"] == 0.0
        assert rewards["agent_1"] == 0.0


class TestMixedReward:
    def test_shared_plus_subgoal(self):
        task = _make_task(category="mixed")
        shaper = RewardShaper(task)

        eval_result = {
            "main_goal_progress": 0.5,
            "main_goal_success": False,
            "agent_subgoal_status": {"agent_0": True, "agent_1": False},
        }
        rewards = shaper.compute(eval_result, terminal=False)

        # agent_0 gets base + subgoal bonus
        base = -0.01 + 0.5
        assert abs(rewards["agent_0"] - (base + 0.3)) < 1e-6
        assert abs(rewards["agent_1"] - base) < 1e-6

    def test_subgoal_only_once(self):
        task = _make_task(category="mixed")
        shaper = RewardShaper(task)

        eval1 = {"main_goal_progress": 0.3, "agent_subgoal_status": {"agent_0": True}}
        shaper.compute(eval1, terminal=False)

        # Second time — subgoal already counted, no bonus
        eval2 = {"main_goal_progress": 0.3, "agent_subgoal_status": {"agent_0": True}}
        rewards = shaper.compute(eval2, terminal=False)
        assert abs(rewards["agent_0"] - (-0.01)) < 1e-6  # Only step cost, no delta or bonus
