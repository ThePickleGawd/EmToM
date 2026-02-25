"""Tests for Communicate step derivation from FD epistemic plans."""

import pytest

from emtom.pddl.dsl import (
    And,
    Knows,
    Literal,
    parse_goal_string,
)
from emtom.pddl.describe import _literal_to_nl, goal_to_natural_language
from emtom.pddl.planner import (
    InformAction,
    parse_fd_inform_actions,
    _derive_communicate_steps,
    generate_deterministic_trajectory,
    wrap_parallel_step,
)
from emtom.pddl.fd_solver import HAS_UP


# ---------------------------------------------------------------------------
# parse_fd_inform_actions
# ---------------------------------------------------------------------------

class TestParseFdInformActions:
    def test_simple_inform(self):
        plan = [
            "open(cabinet_27)",
            "observe_knows_agent_1_abc12345",
            "inform_knows_agent_0_abc12345_from_agent_1",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 1
        assert result[0].receiver == "agent_0"
        assert result[0].fact_hash == "abc12345"
        assert result[0].sender == "agent_1"

    def test_with_token(self):
        plan = [
            "inform_knows_agent_0_abc12345_from_agent_1_tok1",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 1
        assert result[0].sender == "agent_1"
        assert result[0].receiver == "agent_0"

    def test_relay_chain_ordering(self):
        """Relay chain: a0 observes → tells a1 → a1 tells a2."""
        plan = [
            "observe_knows_agent_0_abc12345",
            "inform_knows_agent_1_abc12345_from_agent_0",
            "inform_knows_agent_2_abc12345_from_agent_1",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 2
        # First inform: a0 → a1
        assert result[0].sender == "agent_0"
        assert result[0].receiver == "agent_1"
        # Second inform: a1 → a2
        assert result[1].sender == "agent_1"
        assert result[1].receiver == "agent_2"

    def test_no_informs(self):
        plan = [
            "open(cabinet_27)",
            "observe_knows_agent_0_abc12345",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 0

    def test_empty_plan(self):
        result = parse_fd_inform_actions([])
        assert len(result) == 0

    def test_multiple_facts(self):
        plan = [
            "inform_knows_agent_0_abc12345_from_agent_1",
            "inform_knows_agent_0_def67890_from_agent_1",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 2
        assert result[0].fact_hash == "abc12345"
        assert result[1].fact_hash == "def67890"

    def test_parenthesized_format(self):
        """unified-planning may format as action_name()."""
        plan = [
            "inform_knows_agent_0_abc12345_from_agent_1()",
        ]
        result = parse_fd_inform_actions(plan)
        assert len(result) == 1
        assert result[0].sender == "agent_1"


# ---------------------------------------------------------------------------
# Formula to message (NL conversion)
# ---------------------------------------------------------------------------

class TestFormulaToMessage:
    def test_simple_literal_nl(self):
        lit = Literal("is_open", ("cabinet_27",))
        msg = _literal_to_nl(lit)
        assert "cabinet 27" in msg
        assert "open" in msg

    def test_is_on_top_nl(self):
        lit = Literal("is_on_top", ("bottle_4", "table_13"))
        msg = _literal_to_nl(lit)
        assert "bottle 4" in msg
        assert "table 13" in msg
        assert "on top of" in msg

    def test_negated_literal(self):
        lit = Literal("is_open", ("cabinet_27",), negated=True)
        msg = _literal_to_nl(lit)
        assert "NOT" in msg

    def test_nested_k_message_format(self):
        """Nested K messages should reference the inner agent."""
        leaf = Literal("is_open", ("cabinet_27",))
        inner_msg = _literal_to_nl(leaf)
        nested_msg = f"agent_1 confirmed: {inner_msg}"
        assert "agent_1 confirmed:" in nested_msg
        assert "cabinet 27" in nested_msg


# ---------------------------------------------------------------------------
# _derive_communicate_steps (unit tests with mocked data)
# ---------------------------------------------------------------------------

def _make_k_task_data(
    pddl_goal: str = "(K agent_0 (is_open cabinet_27))",
    num_agents: int = 2,
    mechanics: list = None,
):
    """Build minimal task_data dict for K-goal tests."""
    if mechanics is None:
        mechanics = [
            {
                "mechanic_type": "room_restriction",
                "restricted_rooms": ["kitchen_1"],
                "for_agents": ["agent_0"],
            },
        ]

    problem_pddl = (
        "(define (problem test)\n"
        "  (:domain emtom)\n"
        "  (:objects\n"
        "    agent_0 agent_1 - agent\n"
        "    cabinet_27 - furniture\n"
        "    kitchen_1 bedroom_1 - room\n"
        "  )\n"
        "  (:init\n"
        "    (is_closed cabinet_27)\n"
        "    (can_communicate agent_0 agent_1)\n"
        "    (can_communicate agent_1 agent_0)\n"
        "  )\n"
        f"  (:goal {pddl_goal})\n"
        ")"
    )

    return {
        "task_id": "test_k",
        "title": "Test K Goal",
        "category": "cooperative",
        "scene_id": "test",
        "episode_id": "1",
        "num_agents": num_agents,
        "active_mechanics": ["room_restriction"],
        "mechanic_bindings": mechanics,
        "task": "Test task",
        "agent_secrets": {},
        "agent_actions": {},
        "pddl_domain": "emtom",
        "problem_pddl": problem_pddl,
        "items": [],
        "locked_containers": {},
        "initial_states": {},
        "message_targets": None,
    }


def _make_scene_data():
    return {
        "rooms": ["kitchen_1", "bedroom_1"],
        "furniture_in_rooms": {
            "kitchen_1": ["cabinet_27"],
            "bedroom_1": ["bed_5"],
        },
        "objects_on_furniture": {},
        "furniture": ["cabinet_27", "bed_5"],
        "objects": [],
    }


class TestDeriveCommStepsUnit:
    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_non_trivial_k_produces_steps(self):
        """Non-trivial K goal should produce communicate steps."""
        task_data = _make_k_task_data()
        scene_data = _make_scene_data()
        result = _derive_communicate_steps(
            task_data, scene_data,
            "(K agent_0 (is_open cabinet_27))", 2,
        )
        assert len(result["steps"]) >= 1
        # Check that at least one step has a Communicate action
        found_communicate = False
        for step in result["steps"]:
            for action in step["actions"]:
                if "Communicate" in action["action"]:
                    found_communicate = True
                    assert action["agent"] == "agent_1"  # sender
        assert found_communicate

    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_trivial_k_no_steps(self):
        """Trivial K (agent can observe) → no communicate steps."""
        task_data = _make_k_task_data(
            mechanics=[
                {
                    "mechanic_type": "room_restriction",
                    "restricted_rooms": ["kitchen_1"],
                    "for_agents": ["agent_1"],  # agent_1 restricted, not agent_0
                },
            ],
        )
        scene_data = _make_scene_data()
        result = _derive_communicate_steps(
            task_data, scene_data,
            "(K agent_0 (is_open cabinet_27))", 2,
        )
        assert len(result["steps"]) == 0
        assert any("belief_depth=0" in n for n in result["notes"])

    def test_no_scene_data_skips(self):
        """No scene data → skip communication derivation."""
        task_data = _make_k_task_data()
        result = _derive_communicate_steps(
            task_data, {},
            "(K agent_0 (is_open cabinet_27))", 2,
        )
        assert len(result["steps"]) == 0

    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_communicate_message_is_nl(self):
        """Communicate action should contain natural language."""
        task_data = _make_k_task_data()
        scene_data = _make_scene_data()
        result = _derive_communicate_steps(
            task_data, scene_data,
            "(K agent_0 (is_open cabinet_27))", 2,
        )
        for step in result["steps"]:
            for action in step["actions"]:
                if "Communicate" in action["action"]:
                    # Message should be NL, not PDDL
                    assert "cabinet 27" in action["action"]
                    assert "is_open" not in action["action"]

    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_correct_agent_roles(self):
        """Sender communicates, others wait."""
        task_data = _make_k_task_data()
        scene_data = _make_scene_data()
        result = _derive_communicate_steps(
            task_data, scene_data,
            "(K agent_0 (is_open cabinet_27))", 2,
        )
        for step in result["steps"]:
            actions_by_agent = {a["agent"]: a["action"] for a in step["actions"]}
            # Exactly one agent communicates
            comm_agents = [
                a for a, act in actions_by_agent.items()
                if "Communicate" in act
            ]
            assert len(comm_agents) == 1
            # Others wait
            for agent, act in actions_by_agent.items():
                if agent not in comm_agents:
                    assert act == "Wait[]"


# ---------------------------------------------------------------------------
# E2E: generate_deterministic_trajectory with K goals
# ---------------------------------------------------------------------------

class TestGenerateTrajectoryWithCommunicate:
    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_k_goal_appends_communicate(self):
        """Full E2E: K goal task → trajectory ends with Communicate steps."""
        task_data = _make_k_task_data()
        scene_data = _make_scene_data()
        result = generate_deterministic_trajectory(task_data, scene_data)
        # Check communication_derived flag
        assert result["communication_derived"] is True
        # Last step(s) should contain Communicate
        found_communicate = False
        for step in result["trajectory"]:
            for action in step["actions"]:
                if "Communicate" in action["action"]:
                    found_communicate = True
        assert found_communicate

    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_physical_steps_come_first(self):
        """Physical steps precede Communicate steps."""
        task_data = _make_k_task_data(
            pddl_goal="(and (is_open cabinet_27) (K agent_0 (is_open cabinet_27)))",
        )
        scene_data = _make_scene_data()
        result = generate_deterministic_trajectory(task_data, scene_data)

        # Find the first communicate step index
        first_comm_idx = None
        last_physical_idx = None
        for i, step in enumerate(result["trajectory"]):
            for action in step["actions"]:
                if "Communicate" in action["action"]:
                    if first_comm_idx is None:
                        first_comm_idx = i
                elif action["action"] != "Wait[]":
                    last_physical_idx = i

        if first_comm_idx is not None and last_physical_idx is not None:
            assert first_comm_idx > last_physical_idx

    def test_no_k_goal_no_communicate(self):
        """Non-K goal → no communicate steps, communication_derived=False."""
        task_data = {
            "task_id": "test",
            "title": "Test",
            "category": "cooperative",
            "num_agents": 2,
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:objects agent_0 agent_1 - agent cabinet_27 - furniture)\n"
                "  (:init (is_closed cabinet_27))\n"
                "  (:goal (is_open cabinet_27))\n"
                ")"
            ),
            "mechanic_bindings": [],
            "items": [],
            "locked_containers": {},
            "initial_states": {},
        }
        result = generate_deterministic_trajectory(task_data, _make_scene_data())
        assert result["communication_derived"] is False
        for step in result["trajectory"]:
            for action in step["actions"]:
                assert "Communicate" not in action["action"]

    @pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
    def test_planner_notes_reflect_communicate(self):
        """Planner notes should mention communicate steps when derived."""
        task_data = _make_k_task_data()
        scene_data = _make_scene_data()
        result = generate_deterministic_trajectory(task_data, scene_data)
        if result["communication_derived"]:
            assert any("Communicate" in n for n in result["planner_notes"])


# ---------------------------------------------------------------------------
# Relay chain E2E (3 agents)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
class TestRelayChainCommunicate:
    def test_relay_produces_two_communicate_steps(self):
        """K(a2, phi): a0 observes, 0→1→2 relay → two communicate steps."""
        problem_pddl = (
            "(define (problem relay_test)\n"
            "  (:domain emtom)\n"
            "  (:objects\n"
            "    agent_0 agent_1 agent_2 - agent\n"
            "    cabinet_27 - furniture\n"
            "    kitchen_1 bedroom_1 - room\n"
            "  )\n"
            "  (:init\n"
            "    (is_closed cabinet_27)\n"
            "    (can_communicate agent_0 agent_1)\n"
            "    (can_communicate agent_1 agent_2)\n"
            "  )\n"
            "  (:goal (K agent_2 (is_open cabinet_27)))\n"
            ")"
        )

        task_data = {
            "task_id": "relay_test",
            "title": "Relay Test",
            "category": "cooperative",
            "scene_id": "test",
            "episode_id": "1",
            "num_agents": 3,
            "active_mechanics": ["room_restriction"],
            "mechanic_bindings": [
                {
                    "mechanic_type": "room_restriction",
                    "restricted_rooms": ["kitchen_1"],
                    "for_agents": ["agent_1", "agent_2"],
                },
            ],
            "task": "Relay test",
            "agent_secrets": {},
            "agent_actions": {},
            "pddl_domain": "emtom",
            "problem_pddl": problem_pddl,
            "items": [],
            "locked_containers": {},
            "initial_states": {},
            "message_targets": None,
        }

        scene_data = {
            "rooms": ["kitchen_1", "bedroom_1"],
            "furniture_in_rooms": {
                "kitchen_1": ["cabinet_27"],
                "bedroom_1": ["bed_5"],
            },
            "objects_on_furniture": {},
            "furniture": ["cabinet_27", "bed_5"],
            "objects": [],
        }

        result = generate_deterministic_trajectory(task_data, scene_data)
        assert result["communication_derived"] is True

        # Count communicate steps
        comm_steps = []
        for step in result["trajectory"]:
            for action in step["actions"]:
                if "Communicate" in action["action"]:
                    comm_steps.append(action)

        # Relay chain should produce exactly 2 communicates
        assert len(comm_steps) == 2
