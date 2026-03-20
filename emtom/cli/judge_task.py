"""
Evaluate task quality using multi-model council (Kimi K2.5 + GPT-5.2).

Scores task on category-specific criteria and returns pass/fail with suggestions.

Usage:
    # CLI
    python -m emtom.cli.judge_task task.json [--working-dir DIR] [--threshold 0.65]

    # Programmatic
    from emtom.cli.judge_task import run
    result = run("task.json", working_dir="/tmp/work")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from emtom.cli import CLIResult, failure, success
from emtom.cli.task_metadata import compute_strict_tom_metadata


def run(
    task_file: str,
    working_dir: str = None,
    scene_file: str = None,
    trajectory_dir: str = None,
    models: Optional[List[str]] = None,
    threshold: float = 0.65,
    difficulty: Optional[str] = None,
    user_query: Optional[str] = None,
    required_tom_level: Optional[int] = None,
) -> CLIResult:
    """
    Evaluate task quality using multi-model council.

    Args:
        task_file: Path to task JSON file.
        working_dir: Optional working directory (for scene data, trajectory lookup).
        scene_file: Optional explicit scene data JSON file.
        trajectory_dir: Optional path to benchmark rollout data.
        models: Council model names (default: ["kimi-k2.5", "gpt-5.2"]).
        threshold: Overall score threshold for passing.
        difficulty: Difficulty level context (easy/medium/hard).
        user_query: Optional user query the task should align with.
        required_tom_level: Optional strict tom_level the task must satisfy.

    Returns:
        CLIResult with data keys: passed, overall_score, threshold,
        models, model_results, suggestions, disagreements.
    """
    task_path = Path(task_file)
    if not task_path.exists():
        return failure(f"Task file not found: {task_file}")

    try:
        with open(task_path) as f:
            task_data = json.load(f)
    except json.JSONDecodeError as e:
        return failure(f"Invalid JSON: {e}")

    scene_data = _load_scene_data(working_dir, scene_file)

    # Deterministic strict PDDL verification is the first judge gate.
    from emtom.cli.verify_pddl import run as verify_pddl_run

    verify_result = verify_pddl_run(task_file, working_dir=working_dir)
    if not verify_result["success"]:
        return verify_result

    try:
        strict_tom = compute_strict_tom_metadata(task_data, scene_data=scene_data)
    except Exception as e:
        return failure(f"Strict ToM verification failed: {e}")

    strict_tom_level = strict_tom.get("tom_level")
    if required_tom_level is not None and strict_tom_level != required_tom_level:
        return failure(
            f"Strict tom_level is {strict_tom_level} but required tom_level is {required_tom_level}.",
            data={
                "required_tom_level": required_tom_level,
                "computed_tom_level": strict_tom_level,
                "strict_tom_verification": strict_tom,
            },
        )

    # Always regenerate golden trajectory from authoritative task spec.
    try:
        from emtom.pddl.planner import regenerate_golden_trajectory

        regenerate_golden_trajectory(
            task_data,
            scene_data=scene_data,
            source="judge_auto",
            task_file=task_file,
        )
    except Exception as e:
        return failure(f"Failed to regenerate golden trajectory from task spec: {e}")

    # Validate task structure before expensive LLM calls
    from emtom.cli.validate_task import validate

    validation = validate(task_data, scene_data)
    if not validation["success"]:
        return validation

    # Find latest trajectory dir if not explicitly provided
    traj_path = None
    if trajectory_dir:
        traj_path = Path(trajectory_dir)
    elif working_dir:
        trajectories_dir = Path(working_dir) / "agent_trajectories"
        if trajectories_dir.exists():
            task_dirs = sorted(trajectories_dir.glob("task_*"), key=lambda p: p.name)
            if task_dirs:
                run_dirs = sorted(task_dirs[-1].glob("run_*"), key=lambda p: p.name)
                if run_dirs:
                    traj_path = run_dirs[-1]

    # Create judge and evaluate
    from emtom.task_gen.judge import Judge

    judge = Judge(
        models=models,
        overall_threshold=threshold,
        difficulty=difficulty,
        user_query=user_query if not difficulty else None,
    )

    try:
        verdict = judge.evaluate(
            task_data,
            scene_data=scene_data,
            trajectory_dir=traj_path,
        )
    except RuntimeError as e:
        # RuntimeError from _aggregate means judge council is incomplete due to
        # infrastructure (429, timeout, etc.). Re-raise to kill the process
        # rather than letting the agent waste iterations on a non-task issue.
        if "council incomplete" in str(e) or "All judge models failed" in str(e):
            raise
        return failure(f"Evaluation failed: {e}")
    except Exception as e:
        return failure(f"Evaluation failed: {e}")

    # Build result
    result_data: Dict[str, Any] = {
        "passed": verdict.passed,
        "overall_score": verdict.overall_score,
        "threshold": judge.overall_threshold,
        "models": list(verdict.judgments.keys()),
        "pddl_verification": verify_result["data"],
        "strict_tom_verification": strict_tom,
        "model_results": {
            model: {
                "passed": j.is_valid,
                "score": j.overall_score,
            }
            for model, j in verdict.judgments.items()
        },
    }

    if verdict.disagreements:
        result_data["disagreements"] = verdict.disagreements

    if verdict.passed:
        result_data["summary"] = (
            f"PASS - All models agree task is valid (score: {verdict.overall_score:.2f})"
        )
    else:
        result_data["suggestions"] = verdict.suggestions
        result_data["summary"] = (
            f"FAIL - Task did not pass council (score: {verdict.overall_score:.2f})"
        )

    # Save verdict JSON
    verdict_dict = verdict.to_dict()

    if working_dir:
        judgments_dir = Path(working_dir) / "judgments"
        judgments_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judgment_file = judgments_dir / f"judgment_{timestamp}.json"
        with open(judgment_file, "w") as f:
            json.dump(verdict_dict, f, indent=2)
        result_data["judgment_file"] = str(judgment_file)

    # Print status to stderr (human-readable)
    from emtom.task_gen.judge import Colors

    status = "PASS" if verdict.passed else "FAIL"
    color = Colors.GREEN if verdict.passed else Colors.RED
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}=== Task Evaluation (Council) ==={Colors.RESET}",
        file=sys.stderr,
    )
    print(
        f"{Colors.BOLD}{color}{status}{Colors.RESET} - Score: {verdict.overall_score:.2f} "
        f"(threshold: {judge.overall_threshold})",
        file=sys.stderr,
    )
    print(f"Models: {', '.join(verdict.judgments.keys())}", file=sys.stderr)

    if verdict.disagreements:
        print(f"\n{Colors.YELLOW}Model disagreements:{Colors.RESET}", file=sys.stderr)
        for d in verdict.disagreements:
            print(f"  - {d}", file=sys.stderr)

    if not verdict.passed and verdict.suggestions:
        print(f"\n{Colors.YELLOW}Suggestions for improvement:{Colors.RESET}", file=sys.stderr)
        for i, suggestion in enumerate(verdict.suggestions, 1):
            print(f"  {i}. {suggestion}", file=sys.stderr)

    return success(result_data)


def _load_scene_data(working_dir: Optional[str], scene_file: Optional[str]):
    """Load SceneData from file if available."""
    scene_path = Path(scene_file) if scene_file else None
    if scene_path is None and working_dir:
        scene_path = Path(working_dir) / "current_scene.json"

    if scene_path and scene_path.exists():
        try:
            from emtom.task_gen.scene_loader import SceneData

            with open(scene_path) as sf:
                sd = json.load(sf)
            return SceneData(
                episode_id=sd["episode_id"],
                scene_id=sd["scene_id"],
                rooms=sd.get("rooms", []),
                furniture=sd.get("furniture", []),
                objects=sd.get("objects", []),
                articulated_furniture=sd.get("articulated_furniture", []),
                furniture_in_rooms=sd.get("furniture_in_rooms", {}),
                objects_on_furniture=sd.get("objects_on_furniture", {}),
                agent_spawns=sd.get("agent_spawns", {}),
            )
        except Exception:
            pass
    return None


if __name__ == "__main__":
    import argparse

    from emtom.cli import print_result

    parser = argparse.ArgumentParser(description="Evaluate task quality with multi-model council")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--working-dir", default=None, help="Working directory")
    parser.add_argument("--scene-file", default=None, help="Scene data JSON file")
    parser.add_argument("--trajectory-dir", default=None, help="Benchmark rollout data directory")
    parser.add_argument("--models", default=None, help="Comma-separated council models")
    parser.add_argument("--threshold", type=float, default=0.65, help="Score threshold (default: 0.65)")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--required-tom-level", type=int, default=None)
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")] if args.models else None
    result = run(
        args.task_file,
        working_dir=args.working_dir,
        scene_file=args.scene_file,
        trajectory_dir=args.trajectory_dir,
        models=model_list,
        threshold=args.threshold,
        difficulty=args.difficulty,
        required_tom_level=args.required_tom_level,
    )
    print_result(result)
