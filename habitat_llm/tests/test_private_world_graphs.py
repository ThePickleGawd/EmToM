#!/usr/bin/env python3

from types import SimpleNamespace

from habitat_llm.agent.env.environment_interface import EnvironmentInterface
from habitat_llm.planner.planner import Planner


def test_agent_world_graph_observation_sources_are_private() -> None:
    env = EnvironmentInterface.__new__(EnvironmentInterface)

    assert env.get_agent_world_graph_observation_sources(0) == ["0"]
    assert env.get_agent_world_graph_observation_sources(7) == ["7"]


def test_planner_skips_other_agent_world_graph_updates_in_partial_obs() -> None:
    planner = Planner.__new__(Planner)
    planner.env_interface = SimpleNamespace(partial_obs=True)
    assert planner._should_update_other_agent_world_graphs() is False

    planner.env_interface = SimpleNamespace(partial_obs=False)
    assert planner._should_update_other_agent_world_graphs() is True
