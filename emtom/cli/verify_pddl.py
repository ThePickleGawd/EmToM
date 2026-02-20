"""
Verify PDDL goal solvability and compute ToM depth.

Checks:
1. pddl_goal syntax is valid
2. Goal compiles against scene objects
3. Goal is solvable by the PDDL solver
4. Computes ToM depth from epistemic structure

Usage:
    # CLI
    python -m emtom.cli.verify_pddl task.json [--working-dir DIR]

    # Programmatic
    from emtom.cli.verify_pddl import run
    result = run("task.json", working_dir="/tmp/work")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from emtom.cli import CLIResult, failure, success


def run(task_file: str, working_dir: str = None) -> CLIResult:
    """
    Verify PDDL goal solvability and compute ToM depth.

    Args:
        task_file: Path to task JSON file.
        working_dir: Optional working directory (used to find current_scene.json).

    Returns:
        CLIResult with data keys: valid, solvable, tom_level, tom_reasoning,
        goal_description, num_conjuncts, solve_time, pddl_goal.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    pddl_goal = task_data.get("pddl_goal")
    if not pddl_goal:
        return failure("No pddl_goal field in task. Add a pddl_goal string.")

    # Parse goal
    try:
        from emtom.pddl.dsl import parse_goal_string

        goal = parse_goal_string(pddl_goal)
    except Exception as e:
        return failure(f"Invalid PDDL goal syntax: {e}")

    # Validate predicate arities against domain
    from emtom.pddl.dsl import validate_goal_predicates
    from emtom.pddl.domain import EMTOM_DOMAIN

    arity_errors = validate_goal_predicates(goal, EMTOM_DOMAIN)
    if arity_errors:
        return failure(
            "Predicate validation errors:\n" + "\n".join(arity_errors),
            data={"valid": False, "pddl_goal": pddl_goal},
        )

    # Build task object
    from emtom.task_gen.task_generator import GeneratedTask

    task = GeneratedTask.from_dict(task_data)

    # Load scene data from current_scene.json if available
    scene_data: Optional[Dict[str, Any]] = None
    if working_dir:
        scene_file = Path(working_dir) / "current_scene.json"
        if scene_file.exists():
            try:
                with open(scene_file) as sf:
                    scene_data = json.load(sf)
            except (json.JSONDecodeError, IOError):
                pass

    # Compile and solve
    from emtom.pddl.compiler import compile_task
    from emtom.pddl.epistemic import ObservabilityModel
    from emtom.pddl.solver import PDKBSolver

    problem = compile_task(task, scene_data)
    solver = PDKBSolver()
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)
    result = solver.solve(EMTOM_DOMAIN, problem, observability)

    if not result.solvable:
        return failure(
            f"PDDL goal is not solvable: {result.error}",
            data={"valid": False, "pddl_goal": pddl_goal},
        )

    # Check communication budget
    budget_warning = solver.check_communication_budget(problem, observability)

    # Compute ToM depth
    from emtom.pddl.tom_verifier import explain_tom_depth

    tom_info = explain_tom_depth(task, scene_data)

    # Goal description
    from emtom.pddl.describe import goal_to_natural_language

    description = goal_to_natural_language(goal)

    output = {
        "valid": True,
        "pddl_goal": pddl_goal,
        "solvable": True,
        "tom_level": tom_info["tom_level"],
        "tom_reasoning": tom_info["tom_reasoning"],
        "goal_description": description,
        "num_conjuncts": len(goal.flatten()),
        "solve_time": result.solve_time,
    }
    if result.trivial_k_goals:
        output["trivial_k_warnings"] = (
            f"These K() goals are trivially satisfied (agent can directly observe the fact): "
            f"{result.trivial_k_goals}. Consider removing them or adding room_restriction "
            f"to create real information asymmetry."
        )
    if budget_warning:
        output["budget_warning"] = budget_warning

    return success(output)


if __name__ == "__main__":
    import argparse

    from emtom.cli import print_result

    parser = argparse.ArgumentParser(description="Verify PDDL goal solvability")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--working-dir", default=None, help="Working directory (for current_scene.json)")
    args = parser.parse_args()

    result = run(args.task_file, working_dir=args.working_dir)
    print_result(result)
