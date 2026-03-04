"""
Fast Downward solver wrapping unified-planning.

Performs real state-space search for PDDL solvability, unlike PDKBSolver's
structural checks.

When epistemic goals (K/B) are present AND observability data is available,
uses epistemic compilation to verify both physical and epistemic solvability
in a single FD call. Otherwise falls back to stripping K/B and running FD
on the physical goal only.

Falls back to PDKBSolver when unified-planning is not installed.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple

from emtom.pddl.dsl import (
    And,
    Believes,
    Domain,
    EpistemicFormula,
    Formula,
    Knows,
    Literal,
    Not,
    Or,
    Problem,
)
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.solver import PDKBSolver, SolverResult

logger = logging.getLogger(__name__)

try:
    from unified_planning.io import PDDLReader
    from unified_planning.shortcuts import OneshotPlanner, get_environment
    get_environment().credits_stream = None
    HAS_UP = True
except ImportError:
    HAS_UP = False


# ---------------------------------------------------------------------------
# Epistemic stripping
# ---------------------------------------------------------------------------

def _strip_epistemic(formula: Formula) -> Formula:
    """Unwrap K()/B() to get physical goal for classical planner."""
    if isinstance(formula, (Knows, Believes)):
        return _strip_epistemic(formula.inner)
    if isinstance(formula, And):
        return And(tuple(_strip_epistemic(op) for op in formula.operands))
    if isinstance(formula, Or):
        return Or(tuple(_strip_epistemic(op) for op in formula.operands))
    if isinstance(formula, Not) and formula.operand is not None:
        return Not(operand=_strip_epistemic(formula.operand))
    return formula  # Literal unchanged


def _has_epistemic_goals(formula: Formula) -> bool:
    """Check if a formula contains any K()/B() operators."""
    if isinstance(formula, (Knows, Believes)):
        return True
    if isinstance(formula, (And, Or)):
        return any(_has_epistemic_goals(op) for op in formula.operands)
    if isinstance(formula, Not) and formula.operand is not None:
        return _has_epistemic_goals(formula.operand)
    return False


def _deduplicate_conjuncts(formula: Formula) -> Formula:
    """Remove duplicate conjuncts from an And formula."""
    if not isinstance(formula, And):
        return formula
    seen: Set[str] = set()
    unique = []
    for op in formula.operands:
        key = op.to_pddl()
        if key not in seen:
            seen.add(key)
            unique.append(op)
    if len(unique) == 1:
        return unique[0]
    return And(tuple(unique))


# ---------------------------------------------------------------------------
# PDDL serialization for unified-planning
# ---------------------------------------------------------------------------

def _problem_to_pddl(problem: Problem, domain_pddl: str) -> str:
    """Serialize a Problem to a PDDL problem string for classical planning.

    Strips :goal-owners section if present (not standard PDDL).
    """
    obj_lines = []
    by_type: Dict[str, List[str]] = {}
    for name, typ in problem.objects.items():
        by_type.setdefault(typ, []).append(name)
    for typ, names in sorted(by_type.items()):
        obj_lines.append(f"    {' '.join(sorted(names))} - {typ}")
    objects_str = "\n".join(obj_lines)

    init_str = "\n    ".join(l.to_pddl() for l in problem.init if not l.negated)

    goal_str = problem.goal.to_pddl() if problem.goal else "()"

    return (
        f"(define (problem {problem.name})\n"
        f"  (:domain {problem.domain_name})\n"
        f"  (:objects\n{objects_str}\n  )\n"
        f"  (:init\n    {init_str}\n  )\n"
        f"  (:goal {goal_str})\n"
        f")"
    )


def _strip_goal_owners_from_pddl(pddl_str: str) -> str:
    """Remove (:goal-owners ...) section from a PDDL string."""
    return re.sub(r'\(\s*:goal-owners\s.*?\)\s*\)', '', pddl_str, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# FastDownwardSolver
# ---------------------------------------------------------------------------

class FastDownwardSolver:
    """
    Real state-space PDDL solver using Fast Downward via unified-planning.

    For epistemic goals: strips K()/B() to get a physical goal, solves that
    with a classical planner, then validates epistemic requirements separately.

    Falls back to PDKBSolver if unified-planning is not installed.
    """

    def __init__(self):
        self._fallback = PDKBSolver()

    def solve(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel] = None,
        max_belief_depth: int = 3,
        timeout: float = 30.0,
    ) -> SolverResult:
        """
        Solve a PDDL problem using Fast Downward.

        When epistemic goals are present and observability data is available,
        uses epistemic compilation to verify both physical and epistemic
        solvability in a single FD call. Otherwise strips K/B and checks
        the physical goal only.

        Args:
            domain: The PDDL domain.
            problem: The PDDL problem instance.
            observability: Epistemic observability model.
            max_belief_depth: Max epistemic nesting depth.
            timeout: Planner timeout in seconds.

        Returns:
            SolverResult with solvability, plan, and belief depth.
        """
        if not HAS_UP:
            logger.warning(
                "unified-planning not installed; falling back to PDKBSolver "
                "(structural checks only, no real state-space search). "
                "Install with: pip install unified-planning up-fast-downward"
            )
            return self._fallback.solve(domain, problem, observability, max_belief_depth)

        start = time.time()

        if not problem.goal:
            return SolverResult(
                solvable=True,
                belief_depth=0,
                solve_time=time.time() - start,
            )

        # Epistemic compilation path: unified FD call for physical + epistemic
        has_epistemic = _has_epistemic_goals(problem.goal)
        if has_epistemic and observability and observability.object_rooms:
            return self._solve_epistemic(
                domain, problem, observability, timeout, start
            )

        # Non-epistemic or no observability data: strip K/B and solve physical
        return self._solve_physical(
            domain, problem, observability, max_belief_depth, timeout, start
        )

    def _solve_epistemic(
        self,
        domain: Domain,
        problem: Problem,
        observability: ObservabilityModel,
        timeout: float,
        start: float,
    ) -> SolverResult:
        """Epistemic compilation path — single FD call for physical + epistemic."""
        from emtom.pddl.epistemic_compiler import compile_epistemic

        try:
            compilation = compile_epistemic(
                problem.goal, domain, problem, observability
            )
        except Exception as e:
            logger.error("Epistemic compilation error: %s", e)
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=f"Epistemic compilation error: {e}",
            )

        try:
            result = self._run_planner(
                compilation.domain_pddl, compilation.problem_pddl, timeout
            )
        except Exception as e:
            err_str = str(e)
            if "expression false" in err_str.lower():
                # Contradictory goal (e.g. physical (not X) + epistemic K(a, X))
                # Do NOT fall back — the problem is genuinely unsolvable.
                logger.error(
                    "Fast Downward detected contradictory goal (expression false): %s", e
                )
                return SolverResult(
                    solvable=False,
                    solve_time=time.time() - start,
                    error=(
                        "Contradictory goal: epistemic K() fact conflicts with "
                        "a negated physical goal. Ensure K() inner facts are "
                        "consistent with physical goal literals."
                    ),
                )
            logger.error(
                "Fast Downward planner error (epistemic): %s — falling back to PDKBSolver", e
            )
            return self._fallback.solve(
                domain, problem, observability, max_belief_depth=3
            )

        if not result["solvable"]:
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=result.get(
                    "error",
                    "No plan found — goal is unreachable "
                    "(physical or epistemic requirements unsatisfiable)",
                ),
            )

        return SolverResult(
            solvable=True,
            plan=result.get("plan"),
            belief_depth=compilation.belief_depth,
            solve_time=time.time() - start,
            trivial_k_goals=compilation.trivial_k_goals,
        )

    def _solve_physical(
        self,
        domain: Domain,
        problem: Problem,
        observability: Optional[ObservabilityModel],
        max_belief_depth: int,
        timeout: float,
        start: float,
    ) -> SolverResult:
        """Strip K/B and solve the physical goal only (fallback path)."""
        physical_goal = _strip_epistemic(problem.goal)
        physical_goal = _deduplicate_conjuncts(physical_goal)

        planning_problem = Problem(
            name=problem.name,
            domain_name=problem.domain_name,
            objects=problem.objects,
            init=problem.init,
            goal=physical_goal,
        )

        domain_pddl = domain.to_planning_pddl()
        problem_pddl = _problem_to_pddl(planning_problem, domain_pddl)

        try:
            result = self._run_planner(domain_pddl, problem_pddl, timeout)
        except Exception as e:
            logger.error("Fast Downward planner error: %s — falling back to PDKBSolver", e)
            return self._fallback.solve(
                domain, problem, observability, max_belief_depth
            )

        if not result["solvable"]:
            return SolverResult(
                solvable=False,
                solve_time=time.time() - start,
                error=result.get("error", "No plan found — goal is unreachable from init state"),
            )

        # Run epistemic checks (belief depth, trivial K, budget)
        epistemic_result = self._fallback._compute_min_belief_depth(
            problem, observability, max_belief_depth
        )
        belief_depth, trivial_goals = epistemic_result

        return SolverResult(
            solvable=True,
            plan=result.get("plan"),
            belief_depth=belief_depth,
            solve_time=time.time() - start,
            trivial_k_goals=trivial_goals,
        )

    def _run_planner(
        self,
        domain_pddl: str,
        problem_pddl: str,
        timeout: float,
    ) -> dict:
        """Run Fast Downward via unified-planning.

        Returns dict with 'solvable' (bool), 'plan' (list of str), 'error' (str).
        """
        reader = PDDLReader()

        # Write temp files for PDDLReader
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pddl', prefix='fd_domain_', delete=False
        ) as df:
            df.write(domain_pddl)
            domain_path = df.name

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pddl', prefix='fd_problem_', delete=False
        ) as pf:
            pf.write(problem_pddl)
            problem_path = pf.name

        import os
        try:
            up_problem = reader.parse_problem(domain_path, problem_path)
        except Exception as e:
            # Dump PDDL for debugging before deleting temp files
            logger.debug("PDDL parse failure — problem PDDL:\n%s", problem_pddl[:500])
            logger.debug("PDDL parse failure — domain PDDL (first 200 chars):\n%s", domain_pddl[:200])
            os.unlink(domain_path)
            os.unlink(problem_path)
            raise RuntimeError(f"PDDL parse error: {e}") from e

        os.unlink(domain_path)
        os.unlink(problem_path)

        # Solve with Fast Downward (default heuristic search)
        with OneshotPlanner(name="fast-downward") as planner:
            up_result = planner.solve(up_problem, timeout=timeout)

        if up_result.status in (
            up_result.status.__class__.SOLVED_SATISFICING,
            up_result.status.__class__.SOLVED_OPTIMALLY,
        ):
            plan_steps = []
            if up_result.plan:
                for action in up_result.plan.actions:
                    plan_steps.append(str(action))
            return {"solvable": True, "plan": plan_steps}

        return {
            "solvable": False,
            "error": f"Planner status: {up_result.status.name}",
        }

    def check_communication_budget(
        self,
        problem: Problem,
        observability: ObservabilityModel,
    ) -> Optional[str]:
        """Delegate to PDKBSolver's budget check."""
        return self._fallback.check_communication_budget(problem, observability)
