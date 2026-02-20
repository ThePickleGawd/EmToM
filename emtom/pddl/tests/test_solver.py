"""Tests for PDDL solver and ToM verifier."""

import pytest
from unittest.mock import MagicMock

from emtom.pddl.dsl import Literal, And, Problem
from emtom.pddl.domain import EMTOM_DOMAIN
from emtom.pddl.solver import PDKBSolver, _max_epistemic_depth, SolverResult
from emtom.pddl.dsl import Knows, Believes
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.tom_verifier import compute_tom_depth, explain_tom_depth


class TestSolver:
    def _make_problem(self, goal, objects=None, init=None):
        return Problem(
            name="test",
            domain_name="emtom",
            objects=objects or {"cabinet_27": "furniture", "agent_0": "agent"},
            init=init or [],
            goal=goal,
        )

    def test_solvable_simple(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable

    def test_unknown_predicate(self):
        goal = Literal("is_flying", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert not result.solvable
        assert "not in domain" in result.error

    def test_unknown_object(self):
        goal = Literal("is_open", ("nonexistent_99",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert not result.solvable
        assert "unknown object" in result.error

    def test_no_goal(self):
        problem = self._make_problem(None)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable
        assert result.belief_depth == 0

    def test_conjunction_solvable(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("cabinet_27", "cabinet_27")),
        ))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable

    def test_solve_time_tracked(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solve_time >= 0

    def test_belief_depth_with_asymmetry(self):
        goal = Literal("is_open", ("cabinet_27",))
        problem = self._make_problem(goal)
        obs = ObservabilityModel(
            restricted_rooms={"agent_0": {"kitchen_1"}},
        )
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem, obs)
        assert result.solvable
        assert result.belief_depth >= 1

    def test_static_literal_requires_exact_init_match(self):
        goal = Literal("is_inside", ("item_key_1", "cabinet_27"))
        problem = self._make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "item_key_1": "item",
                "cabinet_27": "furniture",
                "cabinet_30": "furniture",
            },
            init=[Literal("is_inside", ("item_key_1", "cabinet_30"))],
        )
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert not result.solvable
        assert "No action can achieve literal" in result.error

    def test_static_literal_satisfied_from_exact_init(self):
        goal = Literal("is_inside", ("item_key_1", "cabinet_27"))
        problem = self._make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "item_key_1": "item",
                "cabinet_27": "furniture",
            },
            init=[Literal("is_inside", ("item_key_1", "cabinet_27"))],
        )
        result = PDKBSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable


class TestEpistemicDepth:
    def test_literal_depth_0(self):
        assert _max_epistemic_depth(Literal("is_open", ("a",))) == 0

    def test_knows_depth_1(self):
        f = Knows("agent_0", Literal("is_open", ("a",)))
        assert _max_epistemic_depth(f) == 1

    def test_nested_knows_depth_2(self):
        inner = Knows("agent_1", Literal("is_open", ("a",)))
        outer = Knows("agent_0", inner)
        assert _max_epistemic_depth(outer) == 2

    def test_triple_nested_depth_3(self):
        l3 = Knows("agent_2", Literal("is_open", ("a",)))
        l2 = Knows("agent_1", l3)
        l1 = Knows("agent_0", l2)
        assert _max_epistemic_depth(l1) == 3

    def test_and_with_epistemic(self):
        f = And(operands=(
            Literal("is_open", ("a",)),
            Knows("agent_0", Literal("is_open", ("b",))),
        ))
        assert _max_epistemic_depth(f) == 1


class TestTomVerifier:
    def _make_task(self, pddl_goal="(is_open cabinet_27)", mechanics=None, num_agents=2):
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = num_agents
        task.items = []
        task.initial_states = {}
        task.mechanic_bindings = mechanics or []
        task.locked_containers = {}
        task.message_targets = None
        task.pddl_goal = pddl_goal
        return task

    def test_no_asymmetry_depth_0(self):
        task = self._make_task()
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == 0

    def test_room_restriction_depth_1(self):
        binding = MagicMock()
        binding.mechanic_type = "room_restriction"
        binding.restricted_rooms = ["kitchen_1"]
        binding.for_agents = ["agent_0"]
        binding.trigger_object = None
        binding.target_object = None
        binding.requires_item = None
        task = self._make_task(mechanics=[binding])
        scene = {"rooms": ["kitchen_1"], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth >= 1

    def test_explain_provides_reasoning(self):
        task = self._make_task()
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        info = explain_tom_depth(task, scene)
        assert "tom_level" in info
        assert "tom_reasoning" in info
        assert isinstance(info["tom_reasoning"], str)

    def test_unsolvable_returns_neg1(self):
        task = self._make_task(pddl_goal="(is_flying cabinet_27)")
        scene = {"rooms": [], "furniture": ["cabinet_27"], "objects": []}
        depth = compute_tom_depth(task, scene)
        assert depth == -1
