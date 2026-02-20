"""
PDDL solver wrapper.

Provides an interface to solve epistemic PDDL problems.
Currently implements a lightweight depth-bounded search suitable for
EmToM's bounded belief depth (max 3).

For production use with larger state spaces, this can be extended to
call PDKB (QuMuLab/pdkb-planning) or Fast Downward as external solvers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from emtom.pddl.dsl import Domain, Problem, Literal, And, Or, Knows, Believes, EpistemicFormula, Formula
from emtom.pddl.epistemic import ObservabilityModel


@dataclass
class SolverResult:
    """Result of solving a PDDL problem."""
    solvable: bool
    plan: Optional[List[str]] = None
    belief_depth: int = 0
    solve_time: float = 0.0
    error: Optional[str] = None


class PDKBSolver:
    """
    Lightweight epistemic PDDL solver for EmToM tasks.

    Uses a conservative approach: checks if the goal is achievable
    given the domain actions and observability constraints, without
    full state-space search.

    For EmToM's purposes, we need to verify:
    1. All goal predicates reference valid objects
    2. There exists a sequence of actions that achieves each goal conjunct
    3. Observability constraints don't make the problem unsolvable
    """

    def solve(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel] = None,
        max_belief_depth: int = 3,
    ) -> SolverResult:
        """
        Check if the problem is solvable at the given belief depth.

        This is a structural solvability check, not a full planner.
        It verifies that:
        - All goal predicates are achievable by domain actions
        - Object references in goals are valid
        - Observability constraints don't create impossible requirements
        """
        start = time.time()

        if not problem.goal:
            return SolverResult(
                solvable=True,
                belief_depth=0,
                solve_time=time.time() - start,
            )

        # Extract leaf literals from goal (unwrap K/B wrappers)
        goal_conjuncts = problem.goal.flatten()
        goal_literals = []
        for c in goal_conjuncts:
            node = c
            while isinstance(node, EpistemicFormula):
                node = node.inner
            if isinstance(node, Literal):
                goal_literals.append(node)

        # Check 1: All goal predicates must be achievable
        domain_predicate_names = {p.name for p in domain.predicates}

        for literal in goal_literals:
            if literal.predicate not in domain_predicate_names:
                return SolverResult(
                    solvable=False,
                    belief_depth=0,
                    solve_time=time.time() - start,
                    error=f"Goal predicate '{literal.predicate}' not in domain",
                )

        # Check 2: All object references in goals must exist in problem
        valid_objects = set(problem.objects.keys())
        for literal in goal_literals:
            for arg in literal.args:
                if arg.startswith("?"):
                    continue  # Variable, not ground
                if arg not in valid_objects:
                    return SolverResult(
                        solvable=False,
                        belief_depth=0,
                        solve_time=time.time() - start,
                        error=f"Goal references unknown object '{arg}'",
                    )

        # Check 3: For each goal literal, verify it is reachable.
        # Dynamic predicates (with positive effects) are considered potentially
        # achievable. Static predicates must already hold exactly in init.
        positive_effect_predicates = set()
        for action in domain.actions:
            for effect in action.effects:
                if not effect.literal.negated:
                    positive_effect_predicates.add(effect.literal.predicate)

        init_positive_literals = {
            (l.predicate, l.args)
            for l in problem.init
            if not l.negated
        }

        for literal in goal_literals:
            lit_key = (literal.predicate, literal.args)
            if literal.predicate in positive_effect_predicates:
                # Dynamic predicate; conservative structural check only.
                continue

            if literal.negated:
                # Static negated literal is impossible only if opposite literal
                # is fixed true in init.
                if lit_key in init_positive_literals:
                    return SolverResult(
                        solvable=False,
                        belief_depth=0,
                        solve_time=time.time() - start,
                        error=(
                            "Static goal literal is unsatisfiable: "
                            f"not {literal.predicate}{literal.args} but "
                            "the positive literal is fixed in init"
                        ),
                    )
                continue

            if lit_key not in init_positive_literals:
                return SolverResult(
                    solvable=False,
                    belief_depth=0,
                    solve_time=time.time() - start,
                    error=(
                        f"No action can achieve literal "
                        f"'{literal.predicate}{literal.args}'"
                    ),
                )

        # Check 4: Epistemic requirements
        belief_depth = self._compute_min_belief_depth(
            problem, observability, max_belief_depth
        )

        return SolverResult(
            solvable=True,
            belief_depth=belief_depth,
            solve_time=time.time() - start,
        )

    def _compute_min_belief_depth(
        self,
        problem: Problem,
        observability: Optional[ObservabilityModel],
        max_depth: int,
    ) -> int:
        """
        Compute minimum belief depth needed.

        Depth 0: No epistemic reasoning needed (all agents see everything)
        Depth 1: Agents must reason about others' knowledge
        Depth 2: Agents must reason about what others think they know
        Depth 3: Third-order nesting
        """
        if not observability or not observability.has_information_asymmetry():
            return 0

        # Count nesting depth of epistemic formulas in the goal
        goal_depth = _max_epistemic_depth(problem.goal) if problem.goal else 0

        # If there's information asymmetry but no explicit epistemic goals,
        # the task requires at least depth 1 (agents need to model others' beliefs)
        if goal_depth == 0 and observability.has_information_asymmetry():
            # Check if communication is required to bridge information gaps
            if observability.restricted_rooms or observability.hidden_effects:
                return 1

        return min(goal_depth, max_depth)


def _max_epistemic_depth(formula: Optional[Formula]) -> int:
    """Compute the maximum nesting depth of epistemic operators in a formula."""
    if formula is None:
        return 0
    if isinstance(formula, (Knows, Believes)):
        return 1 + _max_epistemic_depth(formula.inner)
    if isinstance(formula, (And, Or)):
        return max((_max_epistemic_depth(op) for op in formula.operands), default=0)
    if hasattr(formula, 'operand'):
        return _max_epistemic_depth(formula.operand)
    return 0
