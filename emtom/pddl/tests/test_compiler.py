"""Tests for PDDL compiler and DSL."""

import pytest

from emtom.pddl.dsl import (
    Literal, And, Or, Not, Knows, Believes, EpistemicFormula,
    parse_goal_string, goal_to_string,
    Type, Predicate, Param, Problem, Domain,
)
from emtom.pddl.solver import _max_epistemic_depth
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
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:init)\n"
                "  (:goal (and (is_open cabinet_27) (is_on_top bottle_4 table_13)))\n"
                ")"
            ),
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
        if pddl_goal is not None:
            task.problem_pddl = (
                f"(define (problem test_001)\n"
                f"  (:domain emtom)\n"
                f"  (:objects)\n"
                f"  (:init)\n"
                f"  (:goal {pddl_goal})\n"
                f")"
            )
        else:
            task.problem_pddl = None
        return task

    def test_basic_compile(self):
        task = self._make_task()
        scene_data = {
            "rooms": ["kitchen_1", "bedroom_1"],
            "furniture": ["cabinet_27", "table_13"],
            "objects": ["bottle_4"],
        }
        problem = compile_task(task, scene_data)
        assert problem.name == "test_001"
        assert "agent_0" in problem.objects
        assert "cabinet_27" in problem.objects
        assert problem.goal is not None

    def test_mechanic_bindings(self):
        """Test that problem_pddl init literals are preserved in compiled problem."""
        from unittest.mock import MagicMock
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.items = []
        task.initial_states = {}
        task.mechanic_bindings = []
        task.locked_containers = {}
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain emtom)\n"
            "  (:objects cabinet_27 - furniture)\n"
            "  (:init (is_inverse cabinet_27))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )

        problem = compile_task(task)
        init_preds = {l.predicate for l in problem.init}
        assert "is_inverse" in init_preds

    def test_no_goal(self):
        task = self._make_task(pddl_goal=None)
        problem = compile_task(task)
        assert problem.goal is None


# ---------------------------------------------------------------------------
# Epistemic parser tests
# ---------------------------------------------------------------------------

class TestEpistemicParser:
    def test_parse_knows(self):
        result = parse_goal_string("(K agent_0 (is_open cabinet_27))")
        assert isinstance(result, Knows)
        assert result.agent == "agent_0"
        assert isinstance(result.inner, Literal)
        assert result.inner.predicate == "is_open"
        assert result.inner.args == ("cabinet_27",)

    def test_parse_believes(self):
        result = parse_goal_string("(B agent_1 (is_inside key_1 safe_3))")
        assert isinstance(result, Believes)
        assert result.agent == "agent_1"
        assert isinstance(result.inner, Literal)
        assert result.inner.predicate == "is_inside"

    def test_parse_nested_knows(self):
        result = parse_goal_string("(K agent_0 (K agent_1 (is_inside key_1 safe_3)))")
        assert isinstance(result, Knows)
        assert result.agent == "agent_0"
        assert isinstance(result.inner, Knows)
        assert result.inner.agent == "agent_1"
        assert isinstance(result.inner.inner, Literal)

    def test_parse_and_with_epistemic(self):
        result = parse_goal_string(
            "(and (K agent_0 (is_open cabinet_27)) (is_on_top bottle_4 table_13))"
        )
        assert isinstance(result, And)
        assert len(result.operands) == 2
        assert isinstance(result.operands[0], Knows)
        assert isinstance(result.operands[1], Literal)

    def test_parse_not_knows(self):
        result = parse_goal_string("(not (K agent_1 (is_inside gem_1 safe_3)))")
        assert isinstance(result, Not)
        assert isinstance(result.operand, Knows)
        assert result.operand.agent == "agent_1"

    def test_roundtrip_knows(self):
        original = "(K agent_0 (is_open cabinet_27))"
        parsed = parse_goal_string(original)
        serialized = goal_to_string(parsed)
        assert serialized == original
        reparsed = parse_goal_string(serialized)
        assert isinstance(reparsed, Knows)

    def test_roundtrip_nested_knows(self):
        original = "(K agent_0 (K agent_1 (is_inside key_1 safe_3)))"
        parsed = parse_goal_string(original)
        serialized = goal_to_string(parsed)
        assert serialized == original

    def test_roundtrip_and_with_epistemic(self):
        original = "(and (K agent_0 (is_inside key_1 cabinet_27)) (is_open safe_3))"
        parsed = parse_goal_string(original)
        serialized = goal_to_string(parsed)
        reparsed = parse_goal_string(serialized)
        assert serialized == reparsed.to_pddl()

    def test_malformed_k_no_args(self):
        with pytest.raises(ValueError, match="K\\(\\) requires"):
            parse_goal_string("(K)")

    def test_malformed_k_no_inner(self):
        with pytest.raises(ValueError, match="K\\(\\) requires an inner formula"):
            parse_goal_string("(K agent_0)")

    def test_case_insensitive(self):
        """K and k should both work."""
        result = parse_goal_string("(k agent_0 (is_open cabinet_27))")
        assert isinstance(result, Knows)


# ---------------------------------------------------------------------------
# Epistemic depth tests
# ---------------------------------------------------------------------------

class TestEpistemicDepth:
    def test_depth_0_literal(self):
        formula = Literal("is_open", ("cabinet_27",))
        assert _max_epistemic_depth(formula) == 0

    def test_depth_0_and(self):
        formula = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        assert _max_epistemic_depth(formula) == 0

    def test_depth_1_knows(self):
        formula = Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",)))
        assert _max_epistemic_depth(formula) == 1

    def test_depth_2_nested_knows(self):
        inner = Knows(agent="agent_1", inner=Literal("is_open", ("cabinet_27",)))
        outer = Knows(agent="agent_0", inner=inner)
        assert _max_epistemic_depth(outer) == 2

    def test_depth_mixed_and(self):
        """Max depth across And operands."""
        formula = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_open", ("c",))),
            Literal("is_on_top", ("b", "t")),
        ))
        assert _max_epistemic_depth(formula) == 1

    def test_depth_none(self):
        assert _max_epistemic_depth(None) == 0


# ---------------------------------------------------------------------------
# Epistemic flatten tests
# ---------------------------------------------------------------------------

class TestEpistemicFlatten:
    def test_knows_flatten_preserves_wrapper(self):
        k = Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",)))
        flat = k.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Knows)

    def test_believes_flatten_preserves_wrapper(self):
        b = Believes(agent="agent_1", inner=Literal("is_inside", ("key_1", "drawer_5")))
        flat = b.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Believes)

    def test_and_with_epistemic_flatten(self):
        """And with mixed Knows and Literal preserves both."""
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_open", ("c",))),
            Literal("is_on_top", ("b", "t")),
        ))
        flat = goal.flatten()
        assert len(flat) == 2
        assert isinstance(flat[0], Knows)
        assert isinstance(flat[1], Literal)

    def test_get_inner_literals(self):
        k = Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",)))
        literals = k.get_inner_literals()
        assert len(literals) == 1
        assert isinstance(literals[0], Literal)
        assert literals[0].predicate == "is_open"


# ---------------------------------------------------------------------------
# Epistemic goal checker tests
# ---------------------------------------------------------------------------

class TestEpistemicGoalChecker:
    def test_epistemic_conjuncts_tracked(self):
        """K() conjuncts should be tracked as whole units."""
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_inside", ("key_1", "cabinet_27"))),
            Literal("is_open", ("safe_3",)),
        ))
        checker = PDDLGoalChecker(goal)
        assert len(checker.conjuncts) == 2
        assert isinstance(checker.conjuncts[0], Knows)
        assert isinstance(checker.conjuncts[1], Literal)

    def test_epistemic_evaluate(self):
        """K() goal evaluates via inner literal's truth."""
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_inside", ("key_1", "cabinet_27"))),
            Literal("is_open", ("safe_3",)),
        ))
        checker = PDDLGoalChecker(goal)

        # K(agent_0, is_inside key_1 cabinet_27) completes when inner literal is true
        state = {("is_inside", ("key_1", "cabinet_27")): True}
        result = checker.update(lambda p, a: state.get((p, a), False))
        assert result["percent_complete"] == 0.5
        assert len(result["newly_completed"]) == 1
        assert "(K agent_0" in result["newly_completed"][0]

    def test_epistemic_ordering(self):
        """Ordering should work with K() conjunct strings."""
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_inside", ("key_1", "cabinet_27"))),
            Literal("is_open", ("safe_3",)),
        ))
        ordering = [
            {"before": "(K agent_0 (is_inside key_1 cabinet_27))", "after": "(is_open safe_3)"}
        ]
        checker = PDDLGoalChecker(goal, ordering=ordering)

        # safe_3 is open but K() prerequisite not met
        result = checker.update(lambda p, a: p == "is_open")
        assert len(result["newly_completed"]) == 0

        # Now both are true
        result = checker.update(lambda p, a: True)
        assert result["all_complete"]

    def test_epistemic_to_propositions(self):
        """K() conjuncts should extract inner literal for propositions."""
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_inside", ("key_1", "cabinet_27"))),
            Literal("is_open", ("safe_3",)),
        ))
        checker = PDDLGoalChecker(goal)
        props = checker.to_propositions()
        assert len(props) == 2
        assert props[0]["property"] == "is_inside"
        assert props[0]["entity"] == "key_1"
        assert props[1]["property"] == "is_open"

    def test_from_task_data_with_epistemic(self):
        task_data = {
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:objects)\n"
                "  (:init)\n"
                "  (:goal (and (K agent_0 (is_inside key_1 cabinet_27)) (is_open safe_3)))\n"
                ")"
            ),
        }
        checker = PDDLGoalChecker.from_task_data(task_data)
        assert checker is not None
        assert len(checker.conjuncts) == 2
        assert isinstance(checker.conjuncts[0], Knows)


# ---------------------------------------------------------------------------
# Epistemic describe tests
# ---------------------------------------------------------------------------

class TestEpistemicDescribe:
    def test_knows_nl(self):
        k = Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",)))
        nl = goal_to_natural_language(k)
        assert "agent 0" in nl.lower()
        assert "knows" in nl.lower()
        assert "open" in nl.lower()

    def test_nested_knows_nl(self):
        inner = Knows(agent="agent_1", inner=Literal("is_open", ("cabinet_27",)))
        outer = Knows(agent="agent_0", inner=inner)
        nl = goal_to_natural_language(outer)
        assert "agent 0" in nl.lower()
        assert "agent 1" in nl.lower()
        assert "knows" in nl.lower()

    def test_believes_nl(self):
        b = Believes(agent="agent_1", inner=Literal("is_inside", ("key_1", "drawer_5")))
        nl = goal_to_natural_language(b)
        assert "agent 1" in nl.lower()
        assert "believes" in nl.lower()

    def test_and_with_epistemic_nl(self):
        goal = And(operands=(
            Knows(agent="agent_0", inner=Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        nl = goal_to_natural_language(goal)
        assert "knows" in nl.lower()
        assert "top" in nl.lower()


# ---------------------------------------------------------------------------
# Predicate arity validation tests
# ---------------------------------------------------------------------------

from emtom.pddl.dsl import validate_goal_predicates
from emtom.pddl.domain import EMTOM_DOMAIN, get_predicates_for_prompt


class TestArityValidation:
    def test_valid_unary_predicate(self):
        goal = Literal("is_open", ("cabinet_27",))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert errors == []

    def test_valid_binary_predicate(self):
        goal = Literal("is_on_top", ("bottle_4", "table_13"))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert errors == []

    def test_wrong_arity_too_many(self):
        goal = Literal("is_open", ("cabinet_27", "extra_arg"))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert len(errors) == 1
        assert "expects 1" in errors[0]

    def test_wrong_arity_too_few(self):
        goal = Literal("is_on_top", ("bottle_4",))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert len(errors) == 1
        assert "expects 2" in errors[0]

    def test_unknown_predicate(self):
        goal = Literal("is_flying", ("cabinet_27",))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert len(errors) == 1
        assert "Unknown predicate" in errors[0]

    def test_epistemic_inner_checked(self):
        # K(agent_0, (is_open cabinet_27 extra))
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27", "extra")))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert len(errors) == 1
        assert "expects 1" in errors[0]

    def test_conjunction_all_checked(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("x",)),  # wrong arity
        ))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert len(errors) == 1

    def test_valid_conjunction(self):
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
        assert errors == []


class TestPredicatesForPrompt:
    def test_returns_string(self):
        result = get_predicates_for_prompt()
        assert isinstance(result, str)

    def test_contains_predicates(self):
        result = get_predicates_for_prompt()
        assert "is_open" in result
        assert "is_on_top" in result
        assert "is_inside" in result

    def test_contains_types(self):
        result = get_predicates_for_prompt()
        assert "furniture" in result
        assert "object" in result
        assert "agent" in result

    def test_contains_groups(self):
        result = get_predicates_for_prompt()
        assert "Spatial" in result
        assert "Unary State" in result
        assert "Agent" in result

    def test_mechanic_predicates_marked(self):
        result = get_predicates_for_prompt()
        assert "init-only" in result


# ---------------------------------------------------------------------------
# PDDL ordering cycle detection tests
# ---------------------------------------------------------------------------

try:
    from emtom.task_gen.spec_validator import _has_ordering_cycle
except ImportError:
    # Fall back to importing the function directly if task_gen has missing deps
    _has_ordering_cycle = None


@pytest.mark.skipif(_has_ordering_cycle is None, reason="task_gen dependencies unavailable")
class TestPDDLOrderingCycle:
    def test_no_cycle(self):
        ordering = [
            {"before": "(is_open a)", "after": "(is_open b)"},
            {"before": "(is_open b)", "after": "(is_open c)"},
        ]
        assert not _has_ordering_cycle(ordering)

    def test_simple_cycle(self):
        ordering = [
            {"before": "(is_open a)", "after": "(is_open b)"},
            {"before": "(is_open b)", "after": "(is_open a)"},
        ]
        assert _has_ordering_cycle(ordering)

    def test_indirect_cycle(self):
        ordering = [
            {"before": "(is_open a)", "after": "(is_open b)"},
            {"before": "(is_open b)", "after": "(is_open c)"},
            {"before": "(is_open c)", "after": "(is_open a)"},
        ]
        assert _has_ordering_cycle(ordering)

    def test_empty_ordering(self):
        assert not _has_ordering_cycle([])

    def test_single_constraint(self):
        ordering = [{"before": "(is_open a)", "after": "(is_open b)"}]
        assert not _has_ordering_cycle(ordering)


# ---------------------------------------------------------------------------
# Not.flatten() negation preservation tests
# ---------------------------------------------------------------------------

class TestNotFlatten:
    def test_not_literal_preserves_negation(self):
        """Not(Literal) should flatten to Literal(negated=True)."""
        f = Not(operand=Literal("is_open", ("cabinet_27",)))
        flat = f.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Literal)
        assert flat[0].negated is True
        assert flat[0].predicate == "is_open"
        assert flat[0].args == ("cabinet_27",)

    def test_not_literal_evaluates_correctly(self):
        """Flattened negated literal should evaluate as negation."""
        f = Not(operand=Literal("is_open", ("cabinet_27",)))
        flat = f.flatten()
        negated_lit = flat[0]
        # is_open is True in state, negated should be False
        assert not negated_lit.evaluate(lambda p, a: True)
        # is_open is False in state, negated should be True
        assert negated_lit.evaluate(lambda p, a: False)

    def test_not_literal_pddl_roundtrip(self):
        """Negated literal from Not.flatten() should serialize correctly."""
        f = Not(operand=Literal("is_open", ("cabinet_27",)))
        flat = f.flatten()
        assert flat[0].to_pddl() == "(not (is_open cabinet_27))"

    def test_not_complex_returns_self(self):
        """Not(And(...)) should return [self], not decompose."""
        inner = And(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        f = Not(operand=inner)
        flat = f.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Not)

    def test_double_negation(self):
        """Not(Literal(negated=True)) should flatten to Literal(negated=False)."""
        f = Not(operand=Literal("is_open", ("cabinet_27",), negated=True))
        flat = f.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Literal)
        assert flat[0].negated is False


# ---------------------------------------------------------------------------
# Or.flatten() tests
# ---------------------------------------------------------------------------

class TestOrFlatten:
    def test_or_flatten_returns_self(self):
        """Or.flatten() should return [self], not merge branches."""
        f = Or(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        flat = f.flatten()
        assert len(flat) == 1
        assert isinstance(flat[0], Or)

    def test_or_evaluate_still_works(self):
        """Or.evaluate() should still work correctly."""
        f = Or(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        state = {"a": False, "b": True}
        assert f.evaluate(lambda p, a: state.get(a[0], False))
        assert not f.evaluate(lambda p, a: False)


# ---------------------------------------------------------------------------
# collect_leaf_literals() tests
# ---------------------------------------------------------------------------

from emtom.pddl.dsl import collect_leaf_literals


class TestCollectLeafLiterals:
    def test_simple_literal(self):
        lit = Literal("is_open", ("a",))
        assert len(collect_leaf_literals(lit)) == 1

    def test_and(self):
        f = And(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        assert len(collect_leaf_literals(f)) == 2

    def test_or(self):
        f = Or(operands=(
            Literal("is_open", ("a",)),
            Literal("is_open", ("b",)),
        ))
        # collect_leaf_literals traverses into Or branches
        assert len(collect_leaf_literals(f)) == 2

    def test_not(self):
        f = Not(operand=Literal("is_open", ("a",)))
        result = collect_leaf_literals(f)
        assert len(result) == 1
        assert result[0].predicate == "is_open"

    def test_epistemic(self):
        f = Knows(agent="agent_0", inner=Literal("is_open", ("a",)))
        result = collect_leaf_literals(f)
        assert len(result) == 1

    def test_complex_competitive_goal(self):
        """A realistic competitive goal with Or branches."""
        goal = Or(operands=(
            And(operands=(
                Literal("is_inside", ("trophy_1", "cabinet_10")),
                Not(operand=Literal("is_inside", ("trophy_1", "cabinet_20"))),
            )),
            And(operands=(
                Literal("is_inside", ("trophy_1", "cabinet_20")),
                Not(operand=Literal("is_inside", ("trophy_1", "cabinet_10"))),
            )),
        ))
        result = collect_leaf_literals(goal)
        assert len(result) == 4  # 2 literals x 2 branches


# ---------------------------------------------------------------------------
# Competitive Or-branch goal checker tests
# ---------------------------------------------------------------------------

class TestCompetitiveGoalChecker:
    def _make_competitive_goal(self):
        """Create a typical competitive Or-branched goal."""
        return Or(operands=(
            And(operands=(
                Literal("is_inside", ("trophy_1", "cabinet_10")),
                Literal("is_closed", ("cabinet_10",)),
            )),
            And(operands=(
                Literal("is_inside", ("trophy_1", "cabinet_20")),
                Literal("is_closed", ("cabinet_20",)),
            )),
        ))

    def _make_competitive_owners(self):
        return {
            "(is_inside trophy_1 cabinet_10)": "team_0",
            "(is_closed cabinet_10)": "team_0",
            "(is_inside trophy_1 cabinet_20)": "team_1",
            "(is_closed cabinet_20)": "team_1",
        }

    def test_or_goal_detected(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        assert checker.is_or_goal
        assert checker.num_branches == 2

    def test_branch_conjuncts(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        assert len(checker.get_branch_conjuncts(0)) == 2
        assert len(checker.get_branch_conjuncts(1)) == 2
        assert len(checker.conjuncts) == 4

    def test_branch_for_team(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        assert checker.get_branch_for_team("team_0") == 0
        assert checker.get_branch_for_team("team_1") == 1
        assert checker.get_branch_for_team("team_99") is None

    def test_no_winner_initially(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        result = checker.update(lambda p, a: False)
        assert not result["all_complete"]
        assert result["percent_complete"] == 0.0
        assert result.get("winning_branch") is None

    def test_partial_progress(self):
        """One conjunct of team_0 satisfied."""
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        state = {("is_inside", ("trophy_1", "cabinet_10")): True}
        result = checker.update(lambda p, a: state.get((p, a), False))
        assert result["percent_complete"] == 0.5
        assert not result["all_complete"]

    def test_team_0_wins(self):
        """team_0 branch fully satisfied."""
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        state = {
            ("is_inside", ("trophy_1", "cabinet_10")): True,
            ("is_closed", ("cabinet_10",)): True,
        }
        result = checker.update(lambda p, a: state.get((p, a), False))
        assert result["all_complete"]
        assert result["winning_branch"] == 0

    def test_team_1_wins(self):
        """team_1 branch fully satisfied."""
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        state = {
            ("is_inside", ("trophy_1", "cabinet_20")): True,
            ("is_closed", ("cabinet_20",)): True,
        }
        result = checker.update(lambda p, a: state.get((p, a), False))
        assert result["all_complete"]
        assert result["winning_branch"] == 1

    def test_live_state_no_latching(self):
        """Competitive goals should NOT latch — progress can go backwards."""
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())

        # Step 1: trophy in cabinet_10
        state1 = {("is_inside", ("trophy_1", "cabinet_10")): True}
        result1 = checker.update(lambda p, a: state1.get((p, a), False))
        assert result1["percent_complete"] == 0.5

        # Step 2: trophy moved out — progress should go back to 0
        result2 = checker.update(lambda p, a: False)
        assert result2["percent_complete"] == 0.0
        assert len(checker.completed) == 0

    def test_branch_progress(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        state = {("is_inside", ("trophy_1", "cabinet_10")): True}
        checker.update(lambda p, a: state.get((p, a), False))
        assert checker.get_branch_progress(0) == 0.5
        assert checker.get_branch_progress(1) == 0.0
        assert not checker.is_branch_complete(0)

    def test_cooperative_goal_not_or(self):
        """Cooperative goal (And) should not be detected as Or."""
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        checker = PDDLGoalChecker(goal)
        assert not checker.is_or_goal
        assert checker.num_branches == 0

    def test_cooperative_still_latches(self):
        """Cooperative goals should still latch."""
        goal = And(operands=(
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        checker = PDDLGoalChecker(goal)

        # Step 1: first conjunct true
        state1 = {("is_open", ("cabinet_27",)): True}
        checker.update(lambda p, a: state1.get((p, a), False))
        assert checker.is_conjunct_completed(0)

        # Step 2: first goes false — should stay latched
        checker.update(lambda p, a: False)
        assert checker.is_conjunct_completed(0)  # Still latched!

    def test_get_owner(self):
        goal = self._make_competitive_goal()
        checker = PDDLGoalChecker(goal, owners=self._make_competitive_owners())
        assert checker.get_owner(0) == "team_0"
        assert checker.get_owner(1) == "team_0"
        assert checker.get_owner(2) == "team_1"
        assert checker.get_owner(3) == "team_1"


# ---------------------------------------------------------------------------
# Owner decomposition tests
# ---------------------------------------------------------------------------

class TestOwnerDecomposition:
    def test_compound_owner_decomposed(self):
        """(and A B) owned by team_0 should map each literal to team_0."""
        from emtom.pddl.problem_pddl import _parse_goal_owners

        pddl = (
            "(define (problem test)\n"
            "  (:domain emtom)\n"
            "  (:init)\n"
            "  (:goal (or (and (is_inside trophy_1 cabinet_10) (is_closed cabinet_10))"
            "            (and (is_inside trophy_1 cabinet_20) (is_closed cabinet_20))))\n"
            "  (:goal-owners\n"
            "    (team_0 (and (is_inside trophy_1 cabinet_10) (is_closed cabinet_10)))\n"
            "    (team_1 (and (is_inside trophy_1 cabinet_20) (is_closed cabinet_20))))\n"
            ")"
        )
        owners = _parse_goal_owners(pddl)
        assert owners.get("(is_inside trophy_1 cabinet_10)") == "team_0"
        assert owners.get("(is_closed cabinet_10)") == "team_0"
        assert owners.get("(is_inside trophy_1 cabinet_20)") == "team_1"
        assert owners.get("(is_closed cabinet_20)") == "team_1"

    def test_simple_owner_still_works(self):
        """Single literal ownership still maps correctly."""
        from emtom.pddl.problem_pddl import _parse_goal_owners

        pddl = (
            "(define (problem test)\n"
            "  (:domain emtom)\n"
            "  (:init)\n"
            "  (:goal (is_open cabinet_27))\n"
            "  (:goal-owners\n"
            "    (team_0 (is_open cabinet_27)))\n"
            ")"
        )
        owners = _parse_goal_owners(pddl)
        assert owners.get("(is_open cabinet_27)") == "team_0"

    def test_from_task_data_competitive(self):
        """Full round-trip: task_data → PDDLGoalChecker with Or branches and owners."""
        task_data = {
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:init)\n"
                "  (:goal (or (and (is_inside trophy_1 cabinet_10) (is_closed cabinet_10))"
                "            (and (is_inside trophy_1 cabinet_20) (is_closed cabinet_20))))\n"
                "  (:goal-owners\n"
                "    (team_0 (and (is_inside trophy_1 cabinet_10) (is_closed cabinet_10)))\n"
                "    (team_1 (and (is_inside trophy_1 cabinet_20) (is_closed cabinet_20))))\n"
                ")"
            ),
        }
        checker = PDDLGoalChecker.from_task_data(task_data)
        assert checker is not None
        assert checker.is_or_goal
        assert checker.num_branches == 2
        assert checker.get_branch_for_team("team_0") == 0
        assert checker.get_branch_for_team("team_1") == 1
        assert len(checker.conjuncts) == 4
