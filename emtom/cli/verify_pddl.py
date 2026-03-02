"""
Verify PDDL goal solvability and compute ToM depth.

Checks:
1. `problem_pddl` syntax is valid
2. Goal compiles against scene objects
3. Goal is structurally solvable by PDKBSolver
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
import re
import time
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
    total_start = time.perf_counter()
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    # Build GoalSpec from canonical inline problem_pddl.
    from emtom.pddl.goal_spec import GoalSpec
    from emtom.pddl.domain import EMTOM_DOMAIN
    from emtom.pddl.problem_pddl import parse_problem_pddl

    problem_pddl = task_data.get("problem_pddl")
    legacy_goal_fields = [k for k in ("goals", "pddl_goal", "pddl_ordering", "pddl_owners") if k in task_data]
    parsed_problem = None

    parse_start = time.perf_counter()
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        return failure("Task must define non-empty 'problem_pddl'.")
    if legacy_goal_fields:
        return failure(
            "Legacy goal fields are not supported. "
            f"Remove {legacy_goal_fields} and encode goals in problem_pddl only."
        )
    try:
        parsed_problem = parse_problem_pddl(problem_pddl)
    except ValueError as e:
        return failure(f"Invalid problem_pddl: {e}")

    declared_domain = task_data.get("pddl_domain")
    if isinstance(declared_domain, str) and declared_domain:
        if parsed_problem.domain_name != declared_domain:
            return failure(
                "problem_pddl domain mismatch: "
                f":domain is '{parsed_problem.domain_name}' but pddl_domain is '{declared_domain}'"
            )
    if parsed_problem.domain_name != EMTOM_DOMAIN.name:
        return failure(
            f"Unsupported problem domain '{parsed_problem.domain_name}'. "
            f"Expected '{EMTOM_DOMAIN.name}'."
        )
    if re.search(r"\bteam_[a-zA-Z0-9_]+\b", parsed_problem.goal_pddl):
        return failure(
            "Invalid problem_pddl :goal: found team_* identifier(s). "
            "Use world-state predicates in :goal and put ownership in :goal-owners."
        )

    goal_spec = GoalSpec.from_legacy(parsed_problem.goal_pddl, [], {})
    parse_time_s = time.perf_counter() - parse_start

    # Validate goal spec against domain
    num_agents = task_data.get("num_agents", 2)
    valid_agents = {f"agent_{i}" for i in range(num_agents)}
    validation_errors = goal_spec.validate(EMTOM_DOMAIN, valid_agents)
    if validation_errors:
        return failure(
            "Goal validation errors:\n" + "\n".join(validation_errors),
            data={"valid": False, "pddl_goal": goal_spec.to_pddl_string()},
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

    # Compile and solve with FastDownwardSolver (real state-space search)
    from emtom.pddl.compiler import compile_task
    from emtom.pddl.epistemic import ObservabilityModel
    from emtom.pddl.fd_solver import FastDownwardSolver

    compile_start = time.perf_counter()
    problem = compile_task(task, scene_data)
    compile_time_s = time.perf_counter() - compile_start
    solver = FastDownwardSolver()
    observability = ObservabilityModel.from_task_with_scene(task, scene_data)
    solve_start = time.perf_counter()
    result = solver.solve(EMTOM_DOMAIN, problem, observability)
    solve_wall_time_s = time.perf_counter() - solve_start

    if not result.solvable:
        diagnostic = result.error or "unknown"
        # When epistemic goals are present, diagnose whether the physical
        # layer or the epistemic layer is the root cause.
        from emtom.pddl.fd_solver import _has_epistemic_goals, _strip_epistemic, _deduplicate_conjuncts
        if _has_epistemic_goals(problem.goal):
            try:
                physical_goal = _deduplicate_conjuncts(_strip_epistemic(problem.goal))
                phys_problem = Problem(
                    name=problem.name,
                    domain_name=problem.domain_name,
                    objects=problem.objects,
                    init=problem.init,
                    goal=physical_goal,
                )
                from emtom.pddl.fd_solver import _problem_to_pddl
                domain_pddl = EMTOM_DOMAIN.to_planning_pddl()
                problem_pddl = _problem_to_pddl(phys_problem, domain_pddl)
                phys_result = solver._run_planner(domain_pddl, problem_pddl, timeout=15.0)
                if not phys_result["solvable"]:
                    diagnostic += (
                        "\n  Diagnostic: Physical goals are not achievable from "
                        "initial state (check room restrictions, object placement, "
                        "and action preconditions)"
                    )
                else:
                    diagnostic += (
                        "\n  Diagnostic: Physical goals are achievable but "
                        "epistemic K() requirements are unsatisfiable (check "
                        "communication paths, observability, and K() goal structure)"
                    )
            except Exception:
                pass  # Diagnostic is best-effort
        return failure(
            f"PDDL goal is not solvable: {diagnostic}",
            data={"valid": False, "pddl_goal": goal_spec.to_pddl_string()},
        )

    # Check communication budget
    budget_warning = solver.check_communication_budget(problem, observability)

    # Compute ToM depth (use FD solver result for authoritative belief_depth)
    from emtom.pddl.tom_verifier import explain_tom_depth
    tom_info = explain_tom_depth(task, scene_data, solver_result=result)

    # Goal description
    formula = goal_spec.to_formula()
    from emtom.pddl.describe import goal_to_natural_language
    description = goal_to_natural_language(formula)

    output = {
        "valid": True,
        "pddl_goal": parsed_problem.goal_pddl if parsed_problem else goal_spec.to_pddl_string(),
        "solvable": True,
        "tom_level": tom_info["tom_level"],
        "tom_reasoning": tom_info["tom_reasoning"],
        "goal_description": description,
        "num_conjuncts": len(goal_spec),
        "solve_time": result.solve_time,
        "timing": {
            "parse_time_ms": round(parse_time_s * 1000, 3),
            "compile_time_ms": round(compile_time_s * 1000, 3),
            "solve_time_ms": round(solve_wall_time_s * 1000, 3),
            "total_time_ms": round((time.perf_counter() - total_start) * 1000, 3),
        },
    }
    if result.plan:
        output["plan"] = result.plan
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
