"""
Submit a validated task to the output directory.

Validates structure, checks goal/agent counts, generates filename,
and copies to output directories.

Gate checks (verify/judge/test passed) are NOT enforced here — they are
agent-level state managed by agent.py. This module only handles the
file operations and final validation.

Usage:
    # CLI
    python -m emtom.cli.submit_task task.json --output-dir DIR [--working-dir DIR]

    # Programmatic
    from emtom.cli.submit_task import run
    result = run("task.json", output_dir="data/emtom/tasks")
"""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from emtom.cli import CLIResult, failure, success


def _load_scene_data(working_dir: Optional[str], scene_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load raw scene JSON for object typing/init synthesis."""
    scene_path = Path(scene_file) if scene_file else None
    if scene_path is None and working_dir:
        scene_path = Path(working_dir) / "current_scene.json"
    if not scene_path or not scene_path.exists():
        return None
    try:
        with open(scene_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _ensure_domain_pddl_file(domain_name: str) -> Path:
    """
    Ensure a concrete shared domain PDDL file exists on disk.

    Writes `data/emtom/pddl/domains/<domain_name>/domain.pddl` when missing.
    """
    project_root = Path(__file__).resolve().parents[2]
    domain_dir = project_root / "emtom" / "pddl" / "domains" / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_path = domain_dir / "domain.pddl"

    if not domain_path.exists():
        from emtom.pddl.domain import EMTOM_DOMAIN

        # For now we support a single baked-in EMTOM domain.
        if domain_name != EMTOM_DOMAIN.name:
            raise ValueError(
                f"Unknown pddl_domain '{domain_name}'. Expected '{EMTOM_DOMAIN.name}'."
            )
        domain_path.write_text(EMTOM_DOMAIN.to_pddl() + "\n", encoding="utf-8")

    return domain_path


def _compute_tom_metadata(
    task_data: Dict[str, Any],
    scene_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute authoritative ToM metadata from canonical problem_pddl.

    Returns a dict with at least:
      - tom_level (int, clamped to >= 1 for benchmark compatibility)
      - tom_reasoning (optional str)
    """
    from emtom.pddl.compiler import compile_task
    from emtom.pddl.domain import EMTOM_DOMAIN
    from emtom.pddl.epistemic import ObservabilityModel
    from emtom.pddl.fd_solver import FastDownwardSolver
    from emtom.pddl.tom_verifier import explain_tom_depth
    from emtom.task_gen.task_generator import GeneratedTask

    generated = GeneratedTask.from_dict(task_data)
    problem = compile_task(generated, scene_data=scene_data)
    observability = ObservabilityModel.from_task_with_scene(generated, scene_data)

    solver = FastDownwardSolver()
    solver_result = solver.solve(EMTOM_DOMAIN, problem, observability)
    if not solver_result.solvable:
        raise ValueError(f"PDDL goal is not solvable: {solver_result.error or 'unknown reason'}")

    tom_info = explain_tom_depth(generated, scene_data, solver_result=solver_result)
    tom_level = tom_info.get("tom_level")
    if not isinstance(tom_level, int):
        raise ValueError(f"Invalid computed tom_level: {tom_level!r}")

    result: Dict[str, Any] = {
        "tom_level": max(tom_level, 1),
    }
    tom_reasoning = tom_info.get("tom_reasoning")
    if isinstance(tom_reasoning, str) and tom_reasoning.strip():
        result["tom_reasoning"] = tom_reasoning
    return result


def run(
    task_file: str,
    output_dir: str,
    working_dir: str = None,
    scene_file: str = None,
    submitted_dir: str = None,
    subtasks_min: int = 3,
    subtasks_max: int = 20,
    agents_min: int = 2,
    agents_max: int = 10,
) -> CLIResult:
    """
    Submit a task to the output directory.

    Args:
        task_file: Path to task JSON file.
        output_dir: Permanent output directory.
        working_dir: Optional working directory (for scene data).
        scene_file: Optional explicit scene data JSON file.
        submitted_dir: Optional session-scoped submitted_tasks directory.
        subtasks_min: Minimum PDDL conjuncts or subtasks.
        subtasks_max: Maximum PDDL conjuncts or subtasks.
        agents_min: Minimum agent count.
        agents_max: Maximum agent count.

    Returns:
        CLIResult with data keys: output_path, filename, submitted_path.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")
    scene_data = _load_scene_data(working_dir=working_dir, scene_file=scene_file)

    # Validate task structure
    from emtom.cli.validate_task import run as validate_run

    validation = validate_run(task_file, working_dir=working_dir, scene_file=scene_file)
    if not validation["success"]:
        return validation

    # Canonicalize to inline problem_pddl when missing.
    if not isinstance(task_data.get("problem_pddl"), str) or not task_data.get("problem_pddl", "").strip():
        from emtom.pddl.compiler import compile_task
        from emtom.task_gen.task_generator import GeneratedTask

        generated = GeneratedTask.from_dict(task_data)
        compiled = compile_task(generated, scene_data=scene_data)
        task_data["problem_pddl"] = compiled.to_pddl()
        task_data["pddl_domain"] = compiled.domain_name

        # Persist canonicalized artifact back to working file so follow-up tools
        # operate on the same spec.
        with open(task_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")

    # Compute and persist ToM metadata from canonical PDDL at submit time.
    try:
        tom_meta = _compute_tom_metadata(task_data, scene_data=scene_data)
        task_data["tom_level"] = tom_meta["tom_level"]
        if "tom_reasoning" in tom_meta:
            task_data["tom_reasoning"] = tom_meta["tom_reasoning"]
        else:
            task_data.pop("tom_reasoning", None)
    except Exception as e:
        return failure(f"Failed to compute tom_level during submit: {e}")

    # Always regenerate golden trajectory from authoritative task spec.
    try:
        from emtom.pddl.planner import regenerate_golden_trajectory

        regenerate_golden_trajectory(
            task_data,
            scene_data=scene_data,
            source="submit",
            task_file=str(task_path),
        )
    except Exception as e:
        return failure(f"Failed to regenerate golden trajectory from task spec: {e}")

    # Validate goal count from canonical problem_pddl.
    if task_data.get("problem_pddl"):
        from emtom.pddl.problem_pddl import parse_problem_pddl
        from emtom.pddl.dsl import collect_leaf_literals

        try:
            parsed_problem = parse_problem_pddl(task_data["problem_pddl"])
            num_goals = len(collect_leaf_literals(parsed_problem.goal_formula))
        except Exception:
            num_goals = 0
        if num_goals < subtasks_min:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, minimum is {subtasks_min}.",
                data={"current": num_goals, "required_min": subtasks_min},
            )
        if num_goals > subtasks_max:
            return failure(
                f"Task has {num_goals} PDDL goal conjuncts, maximum is {subtasks_max}.",
                data={"current": num_goals, "required_max": subtasks_max},
            )

    # Validate agent count
    num_agents = task_data.get("num_agents", 2)
    if num_agents < agents_min:
        return failure(
            f"Task has {num_agents} agents, minimum is {agents_min}.",
            data={"current": num_agents, "required_min": agents_min},
        )
    if num_agents > agents_max:
        return failure(
            f"Task has {num_agents} agents, maximum is {agents_max}.",
            data={"current": num_agents, "required_max": agents_max},
        )

    # Canonical task_id naming (stable hash over problem + key metadata).
    title = task_data.get("title", "untitled")
    title_slug = re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-")[:40] or "untitled"
    category = str(task_data.get("category", "cooperative")).lower()
    scene_id = str(task_data.get("scene_id", "scene"))
    episode_id = str(task_data.get("episode_id", "episode"))
    problem_hash = hashlib.sha256(
        (task_data.get("problem_pddl", "") + "|" + category + "|" + scene_id + "|" + episode_id).encode("utf-8")
    ).hexdigest()[:8]
    canonical_task_id = f"emtom-{scene_id}-{episode_id}-{category}-{title_slug}-{problem_hash}"
    task_data["task_id"] = canonical_task_id
    with open(task_path, "w") as f:
        json.dump(task_data, f, indent=2)
        f.write("\n")

    # Generate filename: {datetime}_{title_slug}.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"{timestamp}_{canonical_task_id}.json"

    # Copy to main output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(task_data, f, indent=2)
        f.write("\n")

    # Ensure shared domain PDDL exists.
    pddl_domain = task_data.get("pddl_domain", "emtom")
    try:
        domain_path = _ensure_domain_pddl_file(pddl_domain)
    except ValueError as e:
        return failure(str(e))

    result_data = {
        "output_path": str(output_path),
        "filename": output_filename,
        "task_id": canonical_task_id,
        "title": task_data.get("title", "untitled"),
        "domain_path": str(domain_path),
        "pddl_domain": pddl_domain,
    }

    # Also copy to submitted_tasks/ for session tracking
    if submitted_dir:
        sub_dir = Path(submitted_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        submitted_path = sub_dir / output_filename
        with open(submitted_path, "w") as f:
            json.dump(task_data, f, indent=2)
            f.write("\n")
        result_data["submitted_path"] = str(submitted_path)

    return success(result_data)


if __name__ == "__main__":
    import argparse

    from emtom.cli import print_result

    parser = argparse.ArgumentParser(description="Submit a validated task")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--output-dir", required=True, help="Permanent output directory")
    parser.add_argument("--working-dir", default=None, help="Working directory (for scene data)")
    parser.add_argument("--scene-file", default=None, help="Scene data JSON file")
    parser.add_argument("--submitted-dir", default=None, help="Session submitted_tasks directory")
    parser.add_argument("--subtasks-min", type=int, default=3)
    parser.add_argument("--subtasks-max", type=int, default=20)
    parser.add_argument("--agents-min", type=int, default=2)
    parser.add_argument("--agents-max", type=int, default=10)
    args = parser.parse_args()

    result = run(
        args.task_file,
        output_dir=args.output_dir,
        working_dir=args.working_dir,
        scene_file=args.scene_file,
        submitted_dir=args.submitted_dir,
        subtasks_min=args.subtasks_min,
        subtasks_max=args.subtasks_max,
        agents_min=args.agents_min,
        agents_max=args.agents_max,
    )
    print_result(result)
