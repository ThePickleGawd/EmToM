"""
Theory of Mind depth verifier.

Computes the minimum ToM depth required to solve a task by analyzing
the epistemic structure of the PDDL problem.

Inspired by DAEDALUS (Bolander et al., 2025) iterative deepening approach.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from emtom.pddl.compiler import compile_task
from emtom.pddl.domain import EMTOM_DOMAIN
from emtom.pddl.epistemic import ObservabilityModel
from emtom.pddl.solver import PDKBSolver

if TYPE_CHECKING:
    from emtom.task_gen.task_generator import GeneratedTask


def compute_tom_depth(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
    max_depth: int = 3,
) -> int:
    """
    Compute the minimum Theory of Mind depth for a task.

    This replaces the manually-assigned tom_level field.

    ToM depth meanings:
    - 0: No belief reasoning needed (all information is shared)
    - 1: Agent must reason about what another agent knows/sees
         ("Agent B knows where the key is")
    - 2: Agent must reason about what another agent thinks a third knows
         ("Agent A thinks Agent B believes the safe is on the left")
    - 3: Third-order nesting

    Args:
        task: The generated task
        scene_data: Optional scene data for object resolution
        max_depth: Maximum depth to check (default 3)

    Returns:
        Minimum ToM depth (0-3), or -1 if unsolvable at any depth
    """
    problem = compile_task(task, scene_data)
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)
    solver = PDKBSolver()

    # Base check: is the problem structurally solvable at all?
    result = solver.solve(EMTOM_DOMAIN, problem, observability, max_depth)

    # The PDKBSolver provides structural checks. For actual plan-based
    # verification, use solve_with_epistemic_planner() which does BFS search.
    # The solver's belief_depth gives syntactic depth; actual difficulty
    # may differ based on communication constraints.

    if not result.solvable:
        # If it failed due to missing scene data (unknown objects),
        # fall back to observability-based estimate
        if result.error and "unknown object" in result.error and scene_data is None:
            if observability.has_information_asymmetry():
                return 1
            return 0
        return -1

    # The solver's belief depth computation handles the iterative check
    return result.belief_depth


def explain_tom_depth(
    task: "GeneratedTask",
    scene_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Explain why a task requires a specific ToM depth.

    Returns:
        Dict with:
        - tom_level: int (0-3)
        - tom_reasoning: str explaining why
        - information_gaps: list of asymmetries
        - communication_required: bool
        - trivial_k_goals: list of trivially satisfied K() goals (if any)
    """
    depth = compute_tom_depth(task, scene_data)
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)

    # Analyze information gaps
    gaps = []
    for agent, rooms in observability.restricted_rooms.items():
        gaps.append(f"{agent} cannot see rooms: {sorted(rooms)}")
    for trigger, agents in observability.hidden_effects.items():
        gaps.append(f"Effect of {trigger} hidden from: {sorted(agents)}")

    # Determine if communication is required
    comm_required = bool(observability.restricted_rooms or observability.hidden_effects)

    # Check for trivial K() goals
    trivial_goals = []
    problem = compile_task(task, scene_data)
    result = PDKBSolver().solve(EMTOM_DOMAIN, problem, observability)
    if result.trivial_k_goals:
        trivial_goals = result.trivial_k_goals

    # Build reasoning explanation
    if depth == 0:
        reasoning = "All agents have full observability. No belief reasoning needed."
    elif depth == 1:
        reasoning = (
            f"Information asymmetry requires first-order belief reasoning. "
            f"Agents must reason about what others know. "
            f"Gaps: {'; '.join(gaps) if gaps else 'agent secrets create private knowledge'}."
        )
    elif depth == 2:
        reasoning = (
            f"Task requires second-order belief reasoning. "
            f"Agents must model what others think about third parties' knowledge. "
            f"Gaps: {'; '.join(gaps)}."
        )
    elif depth == 3:
        reasoning = (
            f"Task requires third-order belief reasoning. "
            f"Complex nested beliefs about others' models of others. "
            f"Gaps: {'; '.join(gaps)}."
        )
    else:
        reasoning = f"Task appears unsolvable at belief depth <= 3."

    if trivial_goals:
        reasoning += (
            f" WARNING: {len(trivial_goals)} K() goal(s) are trivially satisfied "
            f"(agent can directly observe the fact)."
        )

    return {
        "tom_level": depth,
        "tom_reasoning": reasoning,
        "information_gaps": gaps,
        "communication_required": comm_required,
        "trivial_k_goals": trivial_goals,
    }
