"""Tests for Fast Downward solver and domain fixes."""

import pytest

from emtom.pddl.dsl import (
    And,
    ForallEffect,
    Knows,
    Literal,
    Param,
    Problem,
    parse_goal_string,
)
from emtom.pddl.domain import EMTOM_DOMAIN
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.fd_solver import (
    FastDownwardSolver,
    HAS_UP,
    _deduplicate_conjuncts,
    _strip_epistemic,
)
from emtom.pddl.problem_pddl import (
    ParsedProblemPDDL,
    parse_problem_pddl,
    strip_goal_owners_pddl,
)
from emtom.pddl.goal_checker import PDDLGoalChecker


# ---------------------------------------------------------------------------
# Epistemic stripping tests
# ---------------------------------------------------------------------------

class TestStripEpistemic:
    def test_literal_unchanged(self):
        lit = Literal("is_open", ("cabinet_27",))
        assert _strip_epistemic(lit) is lit

    def test_knows_unwrapped(self):
        k = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        result = _strip_epistemic(k)
        assert isinstance(result, Literal)
        assert result.predicate == "is_open"

    def test_nested_knows_fully_unwrapped(self):
        inner = Knows("agent_1", Literal("is_open", ("cabinet_27",)))
        outer = Knows("agent_0", inner)
        result = _strip_epistemic(outer)
        assert isinstance(result, Literal)
        assert result.predicate == "is_open"

    def test_and_with_epistemic(self):
        goal = And((
            Knows("agent_0", Literal("is_open", ("cabinet_27",))),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        result = _strip_epistemic(goal)
        assert isinstance(result, And)
        assert isinstance(result.operands[0], Literal)
        assert result.operands[0].predicate == "is_open"
        assert isinstance(result.operands[1], Literal)

    def test_dedup_conjuncts(self):
        """Multiple K() goals wrapping same literal → deduplicated."""
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_open", ("cabinet_27",)),
            Literal("is_on_top", ("bottle_4", "table_13")),
        ))
        result = _deduplicate_conjuncts(goal)
        assert isinstance(result, And)
        assert len(result.operands) == 2


# ---------------------------------------------------------------------------
# ForallEffect tests
# ---------------------------------------------------------------------------

class TestForallEffect:
    def test_simple_forall(self):
        eff = ForallEffect(
            variable=Param("g", "furniture"),
            condition=Literal("mirrors", ("?f", "?g")),
            effect=Literal("is_open", ("?g",)),
        )
        pddl = eff.to_pddl()
        assert "(forall (?g - furniture)" in pddl
        assert "(when (mirrors ?f ?g) (is_open ?g))" in pddl

    def test_forall_with_negative(self):
        eff = ForallEffect(
            variable=Param("old", "room"),
            condition=Literal("agent_in_room", ("?a", "?old")),
            effect=Literal("agent_in_room", ("?a", "?r")),
            negative_effect=Literal("agent_in_room", ("?a", "?old")),
        )
        pddl = eff.to_pddl()
        assert "(forall (?old - room)" in pddl
        assert "(and (agent_in_room ?a ?r) (not (agent_in_room ?a ?old)))" in pddl


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

class TestDomainFixes:
    def test_domain_has_can_communicate(self):
        pred_names = {p.name for p in EMTOM_DOMAIN.predicates}
        assert "can_communicate" in pred_names

    def test_communicate_has_precondition(self):
        comm = next(a for a in EMTOM_DOMAIN.actions if a.name == "communicate")
        assert comm.preconditions is not None
        assert "can_communicate" in comm.preconditions.to_pddl()

    def test_navigate_has_forall(self):
        nav = next(a for a in EMTOM_DOMAIN.actions if a.name == "navigate")
        has_forall = any(isinstance(e, ForallEffect) for e in nav.effects)
        assert has_forall

    def test_open_has_forall_mirrors(self):
        open_action = next(a for a in EMTOM_DOMAIN.actions if a.name == "open")
        forall_effects = [e for e in open_action.effects if isinstance(e, ForallEffect)]
        mirror_foralls = [e for e in forall_effects if "mirrors" in e.condition.to_pddl()]
        assert len(mirror_foralls) == 1

    def test_open_inverse_negates(self):
        """Opening an inverse furniture should also negate is_open."""
        from emtom.pddl.dsl import Effect
        open_action = next(a for a in EMTOM_DOMAIN.actions if a.name == "open")
        inverse_effects = [
            e for e in open_action.effects
            if isinstance(e, Effect) and e.condition and "is_inverse" in e.condition.to_pddl()
        ]
        # Should have both: set is_closed AND negate is_open
        assert len(inverse_effects) == 2

    def test_planning_pddl_no_epistemic(self):
        pddl = EMTOM_DOMAIN.to_planning_pddl()
        assert ":conditional-effects" in pddl
        assert ":epistemic" not in pddl
        assert "do not edit manually" in pddl


# ---------------------------------------------------------------------------
# Goal owners parser tests
# ---------------------------------------------------------------------------

class TestGoalOwners:
    def test_parse_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain emtom)\n"
            "  (:objects agent_0 agent_1 - agent trophy_1 - object cabinet_10 cabinet_20 - furniture)\n"
            "  (:init)\n"
            "  (:goal (and (is_inside trophy_1 cabinet_10) (is_inside trophy_1 cabinet_20)))\n"
            "  (:goal-owners\n"
            "    (team_0 (is_inside trophy_1 cabinet_10))\n"
            "    (team_1 (is_inside trophy_1 cabinet_20)))\n"
            ")"
        )
        parsed = parse_problem_pddl(pddl)
        assert "(is_inside trophy_1 cabinet_10)" in parsed.owners
        assert parsed.owners["(is_inside trophy_1 cabinet_10)"] == "team_0"
        assert parsed.owners["(is_inside trophy_1 cabinet_20)"] == "team_1"

    def test_no_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain emtom)\n"
            "  (:objects)\n"
            "  (:init)\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        parsed = parse_problem_pddl(pddl)
        assert parsed.owners == {}

    def test_strip_goal_owners(self):
        pddl = (
            "(define (problem test)\n"
            "  (:domain emtom)\n"
            "  (:objects)\n"
            "  (:init)\n"
            "  (:goal (is_open cabinet_27))\n"
            "  (:goal-owners\n"
            "    (team_0 (is_open cabinet_27)))\n"
            ")"
        )
        stripped = strip_goal_owners_pddl(pddl)
        assert ":goal-owners" not in stripped
        assert ":goal" in stripped

    def test_goal_checker_from_task_data_with_owners(self):
        task_data = {
            "problem_pddl": (
                "(define (problem test)\n"
                "  (:domain emtom)\n"
                "  (:objects agent_0 agent_1 - agent trophy_1 - object cabinet_10 cabinet_20 - furniture)\n"
                "  (:init)\n"
                "  (:goal (and (is_inside trophy_1 cabinet_10) (is_inside trophy_1 cabinet_20) (is_open cabinet_10)))\n"
                "  (:goal-owners\n"
                "    (team_0 (is_inside trophy_1 cabinet_10))\n"
                "    (team_1 (is_inside trophy_1 cabinet_20)))\n"
                ")"
            ),
        }
        checker = PDDLGoalChecker.from_task_data(task_data)
        assert checker is not None
        assert len(checker.conjuncts) == 3
        assert checker.get_all_teams() == ["team_0", "team_1"]
        assert len(checker.get_required_conjuncts()) == 1  # is_open unowned
        assert len(checker.get_team_conjuncts("team_0")) == 1
        assert len(checker.get_team_conjuncts("team_1")) == 1


# ---------------------------------------------------------------------------
# FastDownwardSolver integration tests
# ---------------------------------------------------------------------------

def _make_problem(goal, objects=None, init=None):
    return Problem(
        name="test",
        domain_name="emtom",
        objects=objects or {
            "cabinet_27": "furniture",
            "agent_0": "agent",
            "agent_1": "agent",
            "kitchen_1": "room",
        },
        init=init or [
            Literal("agent_in_room", ("agent_0", "kitchen_1")),
            Literal("agent_in_room", ("agent_1", "kitchen_1")),
            Literal("is_in_room", ("cabinet_27", "kitchen_1")),
            Literal("is_closed", ("cabinet_27",)),
            Literal("can_communicate", ("agent_0", "agent_1")),
            Literal("can_communicate", ("agent_1", "agent_0")),
        ],
        goal=goal,
    )


@pytest.mark.skipif(not HAS_UP, reason="unified-planning not installed")
class TestFastDownwardSolver:
    def test_solvable_simple(self):
        """Simple: cabinet in init is closed, goal is to open it."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(goal)
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable
        assert result.plan is not None
        assert len(result.plan) > 0

    def test_unsolvable_restricted(self):
        """Agent restricted from only room with target, no other agents can help
        because communicate has no physical effects."""
        goal = Literal("is_open", ("cabinet_27",))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "cabinet_27": "furniture",
                "bedroom_1": "room",
                "kitchen_1": "room",
            },
            init=[
                Literal("agent_in_room", ("agent_0", "bedroom_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_closed", ("cabinet_27",)),
                Literal("is_restricted", ("agent_0", "kitchen_1")),
            ],
        )
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert not result.solvable

    def test_epistemic_stripped(self):
        """K() goals unwrapped, physical goal checked."""
        goal = Knows("agent_0", Literal("is_open", ("cabinet_27",)))
        problem = _make_problem(goal)
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable

    def test_no_goal_solvable(self):
        problem = _make_problem(None)
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable

    def test_place_achieves_on_top(self):
        """Place action achieves is_on_top."""
        goal = Literal("is_on_top", ("cabinet_27", "cabinet_27"))
        problem = _make_problem(
            goal,
            objects={
                "agent_0": "agent",
                "agent_1": "agent",
                "cabinet_27": "furniture",
                "kitchen_1": "room",
            },
            init=[
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_held_by", ("cabinet_27", "agent_0")),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
            ],
        )
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable

    def test_conjunction_solvable(self):
        goal = And((
            Literal("is_open", ("cabinet_27",)),
            Literal("is_held_by", ("cabinet_27", "agent_0")),
        ))
        problem = _make_problem(
            goal,
            init=[
                Literal("agent_in_room", ("agent_0", "kitchen_1")),
                Literal("agent_in_room", ("agent_1", "kitchen_1")),
                Literal("is_in_room", ("cabinet_27", "kitchen_1")),
                Literal("is_closed", ("cabinet_27",)),
                Literal("can_communicate", ("agent_0", "agent_1")),
                Literal("can_communicate", ("agent_1", "agent_0")),
            ],
        )
        result = FastDownwardSolver().solve(EMTOM_DOMAIN, problem)
        assert result.solvable


class TestFastDownwardFallback:
    def test_fallback_when_no_up(self):
        """When HAS_UP is False, falls back to PDKBSolver."""
        import emtom.pddl.fd_solver as mod
        original = mod.HAS_UP
        try:
            mod.HAS_UP = False
            goal = Literal("is_open", ("cabinet_27",))
            problem = _make_problem(goal)
            solver = FastDownwardSolver()
            result = solver.solve(EMTOM_DOMAIN, problem)
            # PDKBSolver does structural check — should still say solvable
            assert result.solvable
        finally:
            mod.HAS_UP = original


# ---------------------------------------------------------------------------
# Compiler can_communicate tests
# ---------------------------------------------------------------------------

class TestCompilerTypeInference:
    """The compiler now requires self-contained raw PDDL instead of inference."""

    def _make_task(self, goal_str, objects_str=""):
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
            f"(define (problem test_001)\n"
            f"  (:domain emtom)\n"
            f"  (:objects {objects_str})\n" if objects_str else
            f"(define (problem test_001)\n"
            f"  (:domain emtom)\n"
        ) + (
            f"  (:init)\n"
            f"  (:goal {goal_str})\n"
            f")"
        )
        return task

    def test_furniture_inferred_from_is_open(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task("(is_open cabinet_27)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_furniture_inferred_from_is_on_top_second_arg(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task("(is_on_top laptop_0 table_29)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_room_inferred_from_agent_in_room(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task("(agent_in_room agent_0 kitchen_1)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_item_inferred_from_has_item(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task("(has_item agent_0 item_key_1)")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_epistemic_goal_objects_inferred(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task("(K agent_0 (is_open drawer_5))")
        with pytest.raises(ValueError, match="problem_pddl"):
            compile_task(task)

    def test_default_closed_added_for_furniture(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task(
            "(is_open cabinet_27)",
            "agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture",
        )
        task.problem_pddl = task.problem_pddl.replace(
            "  (:init)\n",
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
            "(is_in_room cabinet_27 kitchen_1))\n",
        )
        problem = compile_task(task)
        closed_lits = [l for l in problem.init if l.predicate == "is_closed"]
        assert any(l.args == ("cabinet_27",) for l in closed_lits)

    def test_no_default_closed_if_already_open(self):
        """If init has is_open, don't add is_closed."""
        from unittest.mock import MagicMock
        from emtom.pddl.compiler import compile_task
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 1
        task.items = []
        task.initial_states = {}
        task.mechanic_bindings = []
        task.locked_containers = {}
        task.message_targets = None
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain emtom)\n"
            "  (:objects agent_0 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (is_in_room cabinet_27 kitchen_1) (is_open cabinet_27))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        problem = compile_task(task)
        closed_lits = [l for l in problem.init
                       if l.predicate == "is_closed" and l.args == ("cabinet_27",)]
        assert len(closed_lits) == 0


class TestCompilerCanCommunicate:
    def _make_task(self, mechanic_bindings=None, message_targets=None):
        from unittest.mock import MagicMock
        task = MagicMock()
        task.task_id = "test_001"
        task.num_agents = 2
        task.items = []
        task.initial_states = {}
        task.mechanic_bindings = mechanic_bindings or []
        task.locked_containers = {}
        task.message_targets = message_targets
        task.problem_pddl = (
            "(define (problem test_001)\n"
            "  (:domain emtom)\n"
            "  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) (is_in_room cabinet_27 kitchen_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        return task

    def test_default_all_pairs(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task()
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        # 2 agents → 2 pairs (0→1, 1→0)
        assert len(can_comm) == 2

    def test_message_targets_restricts(self):
        from emtom.pddl.compiler import compile_task
        task = self._make_task(
            message_targets={"agent_0": ["agent_1"]}  # only 0→1
        )
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        assert len(can_comm) == 1
        assert can_comm[0].args == ("agent_0", "agent_1")

    def test_existing_can_communicate_not_duplicated(self):
        """If problem_pddl already has can_communicate, don't add more."""
        from unittest.mock import MagicMock
        from emtom.pddl.compiler import compile_task
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
            "  (:objects agent_0 agent_1 - agent kitchen_1 - room cabinet_27 - furniture)\n"
            "  (:init (agent_in_room agent_0 kitchen_1) (agent_in_room agent_1 kitchen_1) "
            "         (is_in_room cabinet_27 kitchen_1) (can_communicate agent_0 agent_1))\n"
            "  (:goal (is_open cabinet_27))\n"
            ")"
        )
        problem = compile_task(task)
        can_comm = [
            l for l in problem.init if l.predicate == "can_communicate"
        ]
        # Should only have the one from init, not add defaults
        assert len(can_comm) == 1
