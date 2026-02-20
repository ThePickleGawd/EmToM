"""Tests for PDDL compiler and DSL."""

import pytest

from emtom.pddl.dsl import (
    Literal, And, Or, Not, Knows, Believes,
    parse_goal_string, goal_to_string,
    Type, Predicate, Param, Problem, Domain,
)
from emtom.pddl.goal_checker import PDDLGoalChecker
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.compiler import compile_task
from emtom.pddl.describe import goal_to_natural_language, describe_task


# ---------------------------------------------------------------------------
# DSL tests
# ---------------------------------------------------------------------------

class TestLiteral:
    def test_simple_literal(self):
        lit = Literal("is_open", ("cabinet_27",))
        assert lit.to_pddl() == "(is_open cabinet_27)"

    def test_relational_literal(self):
        lit = Literal("is_on_top", ("bottle_4", "table_13"))
        assert lit.to_pddl() == "(is_on_top bottle_4 table_13)"

    def test_negated_literal(self):
        lit = Literal("is_open", ("cabinet_27",), negated=True)
        assert lit.to_pddl() == "(not (is_open cabinet_27))"

    def test_to_proposition(self):
        lit = Literal("is_on_top", ("bottle_4", "table_13"))
        prop = lit.to_proposition()
        assert prop == {"entity": "bottle_4", "property": "is_on_top", "target": "table_13"}

    def test_negated_to_proposition(self):
        lit = Literal("is_open", ("cabinet_27",), negated=True)
        prop = lit.to_proposition()
        assert prop == {"entity": "cabinet_27", "property": "is_open", "value": False}

    def test_from_proposition(self):
        prop = {"entity": "bottle_4", "property": "is_on_top", "target": "table_13"}
        lit = Literal.from_proposition(prop)
        assert lit.predicate == "is_on_top"
        assert lit.args == ("bottle_4", "table_13")
        assert not lit.negated

    def test_from_proposition_negated(self):
        prop = {"entity": "cabinet_27", "property": "is_open", "value": False}
        lit = Literal.from_proposition(prop)
        assert lit.predicate == "is_open"
        assert lit.args == ("cabinet_27",)
        assert lit.negated

    def test_evaluate_true(self):
        lit = Literal("is_open", ("cabinet_27",))
        assert lit.evaluate(lambda p, a: True)

    def test_evaluate_negated(self):
        lit = Literal("is_open", ("cabinet_27",), negated=True)
        assert lit.evaluate(lambda p, a: False)  # not False = True


class TestFormulas:
    def test_and(self):
        f = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        assert "(and" in f.to_pddl()
        assert len(f.flatten()) == 2

    def test_or(self):
        f = Or(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_open", ("cabinet_28",)),
        ))
        assert "(or" in f.to_pddl()

    def test_not(self):
        f = Not(operand=Literal("is_open", ("cabinet_27",)))
        assert f.to_pddl() == "(not (is_open cabinet_27))"

    def test_nested(self):
        f = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Or(operands=(
                Literal("is_on_top", ("bottle_4", "table_13")),
                Literal("is_inside", ("bottle_4", "cabinet_27")),
            )),
        ))
        pddl = f.to_pddl()
        assert "(and" in pddl
        assert "(or" in pddl

    def test_and_evaluate(self):
        f = And(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        # All true
        assert f.evaluate(lambda p, a: True)
        # One false
        state = {"a": True, "b": False}
        assert not f.evaluate(lambda p, a: state.get(a[0], False))

    def test_or_evaluate(self):
        f = Or(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        state = {"a": False, "b": True}
        assert f.evaluate(lambda p, a: state.get(a[0], False))


class TestEpistemic:
    def test_knows(self):
        k = Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",)))
        assert k.to_pddl() == "(K agent_0 (is_open cabinet_27))"

    def test_believes(self):
        b = Believes(agent="agent_1", inner=Literal("is_inside", ("key_1", "drawer_5")))
        assert "(B agent_1" in b.to_pddl()

    def test_nested_epistemic(self):
        # K(agent_0, K(agent_1, is_open(cabinet_27))) — depth 2
        inner = Knows(agent="agent_1", inner=Literal("is_open", ("cabinet_27",)))
        outer = Knows(agent="agent_0", inner=inner)
        assert "(K agent_0 (K agent_1" in outer.to_pddl()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_simple(self):
        result = parse_goal_string("(is_open cabinet_27)")
        assert isinstance(result, Literal)
        assert result.predicate == "is_open"
        assert result.args == ("cabinet_27",)

    def test_and(self):
        result = parse_goal_string("(and (is_open cabinet_27) (is_on_top bottle_4 table_13))")
        assert isinstance(result, And)
        assert len(result.operands) == 2

    def test_nested_and_not(self):
        result = parse_goal_string("(and (is_open cabinet_27) (not (is_closed drawer_5)))")
        assert isinstance(result, And)
        assert isinstance(result.operands[1], Not)

    def test_roundtrip(self):
        original = "(and (is_open cabinet_27) (is_on_top bottle_4 table_13))"
        parsed = parse_goal_string(original)
        serialized = goal_to_string(parsed)
        reparsed = parse_goal_string(serialized)
        assert serialized == reparsed.to_pddl()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_goal_string("")


# ---------------------------------------------------------------------------
# Goal checker tests
# ---------------------------------------------------------------------------

class TestGoalChecker:
    def test_basic_check(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        checker = PDDLGoalChecker(goal)

        # Nothing complete yet
        result = checker.update(lambda p, a: False)
        assert result["percent_complete"] == 0.0
        assert not result["all_complete"]

        # First conjunct completes
        state = {("is_open", ("cabinet_27",)): True}
        result = checker.update(lambda p, a: state.get((p, a), False))
        assert result["percent_complete"] == 0.5
        assert len(result["newly_completed"]) == 1

        # Latching: first stays complete even if state changes
        result = checker.update(lambda p, a: False)
        assert result["percent_complete"] == 0.5  # Still latched

        # Second completes
        state2 = {("is_on_top", ("bottle_4", "table_13")): True}
        result = checker.update(lambda p, a: state2.get((p, a), False))
        assert result["percent_complete"] == 1.0
        assert result["all_complete"]

    def test_ordering(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        ordering = [
            {"before": "(is_open cabinet_27)", "after": "(is_on_top bottle_4 table_13)"}
        ]
        checker = PDDLGoalChecker(goal, ordering=ordering)

        # Only the second predicate is true — it should NOT complete
        # because its prerequisite (first) hasn't completed yet
        result = checker.update(
            lambda p, a: p == "is_on_top"  # only is_on_top is true
        )
        assert len(result["newly_completed"]) == 0

        # Now make the first true — both should complete in sequence
        result = checker.update(lambda p, a: True)
        assert result["all_complete"]

    def test_to_propositions(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        checker = PDDLGoalChecker(goal)
        props = checker.to_propositions()
        assert len(props) == 2
        assert props[0]["property"] == "is_open"
        assert props[1]["property"] == "is_on_top"

    def test_owners(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_inside", ("trophy_1", "cabinet_10")),
            Literal("is_inside", ("trophy_1", "cabinet_20")),
        ))
        owners = {
            "(is_inside trophy_1 cabinet_10)": "team_0",
            "(is_inside trophy_1 cabinet_20)": "team_1",
        }
        checker = PDDLGoalChecker(goal, owners=owners)

        assert len(checker.get_required_conjuncts()) == 1
        assert len(checker.get_team_conjuncts("team_0")) == 1
        assert len(checker.get_team_conjuncts("team_1")) == 1
        assert checker.get_all_teams() == ["team_0", "team_1"]

    def test_from_task_data(self):
        task_data = {
            "pddl_goal": "(and (is_open cabinet_27) (is_on_top bottle_4 table_13))",
            "pddl_ordering": [
                {"before": "(is_open cabinet_27)", "after": "(is_on_top bottle_4 table_13)"}
            ],
        }
        checker = PDDLGoalChecker.from_task_data(task_data)
        assert checker is not None
        assert len(checker.conjuncts) == 2

    def test_from_task_data_none(self):
        checker = PDDLGoalChecker.from_task_data({})
        assert checker is None

    def test_reset(self):
        goal = Literal("is_open", ("cabinet_27",))
        checker = PDDLGoalChecker(goal)
        checker.update(lambda p, a: True)
        assert len(checker.completed) == 1
        checker.reset()
        assert len(checker.completed) == 0


# ---------------------------------------------------------------------------
# Natural language description tests
# ---------------------------------------------------------------------------

class TestDescribe:
    def test_simple_literal(self):
        lit = Literal("is_open", ("cabinet_27",))
        nl = goal_to_natural_language(lit)
        assert "cabinet" in nl.lower()
        assert "open" in nl.lower()

    def test_relational(self):
        lit = Literal("is_on_top", ("bottle_4", "table_13"))
        nl = goal_to_natural_language(lit)
        assert "bottle" in nl.lower()
        assert "table" in nl.lower()
        assert "top" in nl.lower()

    def test_conjunction(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        nl = goal_to_natural_language(goal)
        assert "all" in nl.lower()


# ---------------------------------------------------------------------------
# Compiler tests (lightweight, no Habitat)
# ---------------------------------------------------------------------------

class TestCompiler:
    def _make_task(self, pddl_goal="(is_open cabinet_27)"):
        """Create a minimal mock task for compiler tests."""
        from unittest.mock import MagicMock
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.items = []
        task.initial_states = {}
        task.mechanic_bindings = []
        task.locked_containers = {}
        task.message_targets = None
        task.pddl_goal = pddl_goal
        return task

    def test_basic_compile(self):
        task = self._make_task()
        scene_data = {
            "rooms": ["kitchen_1", "bedroom_1"],
            "furniture": ["cabinet_27", "table_13"],
            "objects": ["bottle_4"],
        }
        problem = compile_task(task, scene_data)
        assert problem.name == "task_test_001"
        assert "agent_0" in problem.objects
        assert "cabinet_27" in problem.objects
        assert problem.goal is not None

    def test_mechanic_bindings(self):
        from unittest.mock import MagicMock
        task = self._make_task()
        binding = MagicMock()
        binding.mechanic_type = "inverse_state"
        binding.trigger_object = "cabinet_27"
        binding.target_object = None
        binding.restricted_rooms = None
        binding.for_agents = None
        binding.requires_item = None
        task.mechanic_bindings = [binding]

        problem = compile_task(task)
        init_preds = {l.predicate for l in problem.init}
        assert "is_inverse" in init_preds

    def test_no_goal(self):
        task = self._make_task(pddl_goal=None)
        problem = compile_task(task)
        assert problem.goal is None
