"""
Integration test stubs for EmtomMultiAgentEnv.

These require Habitat + GPU. Marked with pytest.mark.integration so they
can be skipped in CI with: pytest -m "not integration"
"""

import pytest
from unittest.mock import MagicMock, patch
from emtom.rl.env import MultiAgentEnv, EmtomMultiAgentEnv


class TestMultiAgentEnvBase:
    def test_interface(self):
        env = MultiAgentEnv()
        assert env.possible_agents == []
        assert env.agents == []
        with pytest.raises(NotImplementedError):
            env.reset()
        with pytest.raises(NotImplementedError):
            env.step({})
        with pytest.raises(NotImplementedError):
            env.observe("agent_0")


class TestActionParsing:
    """Test action parsing without Habitat."""

    def _make_env(self):
        """Create an EmtomMultiAgentEnv with mocked dependencies."""
        env = EmtomMultiAgentEnv.__new__(EmtomMultiAgentEnv)
        env.possible_agents = ["agent_0", "agent_1"]
        env.agents = ["agent_0", "agent_1"]
        return env

    def test_parse_navigate(self):
        env = self._make_env()
        text = "Thought: I need to go to the table.\nAgent_0_Action: Navigate[table_1]\nAssigned!"
        result = env._parse_action_text(text, 0)
        assert result == ("Navigate", "table_1")

    def test_parse_pick(self):
        env = self._make_env()
        text = "Thought: Pick up the key.\nAgent_1_Action: Pick[key_0]\nAssigned!"
        result = env._parse_action_text(text, 1)
        assert result == ("Pick", "key_0")

    def test_parse_communicate(self):
        env = self._make_env()
        text = 'Thought: Tell my teammate.\nAgent_0_Action: Communicate["I found the key", agent_1]\nAssigned!'
        result = env._parse_action_text(text, 0)
        assert result[0] == "Communicate"

    def test_parse_wait(self):
        env = self._make_env()
        text = "Thought: Wait.\nAgent_0_Action: Wait[]\nAssigned!"
        result = env._parse_action_text(text, 0)
        assert result == ("Wait", None)

    def test_parse_no_action(self):
        env = self._make_env()
        text = "Thought: Hmm, let me think about this."
        result = env._parse_action_text(text, 0)
        assert result is None

    def test_agent_to_uid(self):
        env = self._make_env()
        assert env._agent_to_uid("agent_0") == 0
        assert env._agent_to_uid("agent_1") == 1
        assert env._agent_to_uid(2) == 2


class TestCumulativeRewards:
    """Test cumulative reward tracking (no Habitat needed)."""

    def _make_env(self):
        env = EmtomMultiAgentEnv.__new__(EmtomMultiAgentEnv)
        env.possible_agents = ["agent_0", "agent_1"]
        env.agents = ["agent_0", "agent_1"]
        env._cumulative_rewards = {}
        env._episode_stats = None
        env._turn_count = 0
        env._done = False
        env.current_task = None
        return env

    def test_initial_cumulative_rewards_empty(self):
        env = self._make_env()
        assert env._cumulative_rewards == {}

    def test_last_episode_stats_initially_none(self):
        env = self._make_env()
        assert env.last_episode_stats is None

    def test_episode_stats_property(self):
        env = self._make_env()
        env._episode_stats = {"task_id": "test", "step_count": 5}
        assert env.last_episode_stats["task_id"] == "test"
        assert env.last_episode_stats["step_count"] == 5


@pytest.mark.integration
class TestEmtomMultiAgentEnvIntegration:
    """Integration tests — require Habitat + GPU."""

    def test_reset_and_observe(self):
        """Verify reset produces valid observations for all agents."""
        # This would need a real Habitat env
        pytest.skip("Requires Habitat + GPU")

    def test_step_cycle(self):
        """Verify a full reset -> step -> observe cycle."""
        pytest.skip("Requires Habitat + GPU")
