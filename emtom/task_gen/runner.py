#!/usr/bin/env python3
"""
Entry point for the agentic task generator.

Usage:
    python emtom/task_gen/runner.py --config-name examples/emtom_2_robots +llm_provider=openai_chat +model=gpt-5

Or via shell script:
    ./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5
    ./emtom/run_emtom.sh generate --llm anthropic_claude --model sonnet
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

# Suppress httpx logging (OpenAI client HTTP requests)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Ensure project root is in Python path (for imports like emtom.*)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import hydra
from omegaconf import DictConfig

from habitat_llm.utils import cprint, setup_config, fix_config


def parse_extra_args():
    """Parse extra CLI arguments before Hydra."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query", type=str, default=None,
                        help="Seed query to guide task generation")
    parser.add_argument("--retry-verification", type=str, default=None,
                        help="Path to failed ToM verification JSON to retry with suggestions")
    parser.add_argument("--calibration-model", type=str, default="gpt-5.2",
                        help="Model to calibrate dataset difficulty against (default: gpt-5.2)")
    parser.add_argument("--target-pass-rate", type=float, default=0.20,
                        help="Target pass rate for calibration model (default: 0.20 = 20%%)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["cooperative", "competitive", "mixed"],
                        help="Task category to generate (default: random)")
    parser.add_argument("--seed-task", type=str, default=None,
                        help="Path to existing task JSON to use as seed (instead of blank template)")
    parser.add_argument("--sampled-tasks-dir", type=str, default=None,
                        help="Pre-built sampled_tasks directory (skips random sampling)")
    parser.add_argument("--judge-threshold", type=float, default=None,
                        help="Override judge overall_threshold (default: judge's built-in default)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"],
                        help="Difficulty level for judge context (easy/medium/hard)")
    parser.add_argument("--test-model", type=str, default=None,
                        help="Override model used for test_task calibration (e.g. gpt-5-mini)")
    parser.add_argument("--tom-target-l1", type=float, default=0.40,
                        help="Target ratio for ToM level 1 tasks (default: 0.40)")
    parser.add_argument("--tom-target-l2", type=float, default=0.40,
                        help="Target ratio for ToM level 2 tasks (default: 0.40)")
    parser.add_argument("--tom-target-l3", type=float, default=0.20,
                        help="Target ratio for ToM level 3 tasks (default: 0.20)")
    parser.add_argument("--tom-ratio-tolerance", type=float, default=0.08,
                        help="Tolerance for ToM ratio calibration guidance (default: 0.08)")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    return args


def _infer_task_tom_level(task_data: dict) -> Optional[int]:
    """Infer tom_level from stored field or computed PDDL semantics."""
    stored = task_data.get("tom_level")
    if isinstance(stored, int) and 1 <= stored <= 3:
        return stored

    # Fallback for historical tasks missing persisted tom_level.
    try:
        from emtom.task_gen.task_generator import GeneratedTask

        task = GeneratedTask.from_dict(task_data)
        level = task.compute_tom_level(scene_data=None)
        if isinstance(level, int):
            # Keep calibration focused on levels 1-3.
            return min(max(level, 1), 3)
    except Exception:
        return None
    return None


def _is_task_like_json(path: Path) -> bool:
    """Return True when a JSON file looks like an EMTOM task spec."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    # Minimal structural check.
    required = ("title", "task", "agent_actions")
    return all(k in data for k in required)


def _copy_sample_with_aliases(src_path: Path, sampled_tasks_dir: Path, index: int) -> None:
    """Copy sampled task with both unpadded and zero-padded filenames."""
    shutil.copy(src_path, sampled_tasks_dir / f"task_{index}.json")
    shutil.copy(src_path, sampled_tasks_dir / f"task_{index:03d}.json")


def populate_sampled_tasks_dir(
    sampled_tasks_dir: Path,
    primary_output_dir: str,
    sample_count: int = 10,
) -> tuple[Optional[Path], int]:
    """
    Populate sampled_tasks from the best available source.

    Priority order:
    1) current output_dir (when it already has tasks)
    2) canonical curated tasks dir
    3) historical fallback curated dir
    """
    candidate_dirs = [
        Path(primary_output_dir),
        Path("data/emtom/tasks"),
        Path("data/emtom/very_old_tasks/old_calibration_format_2_25_26"),
    ]

    # Dedupe while preserving order.
    seen = set()
    ordered_candidates = []
    for d in candidate_dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(d)

    for source_dir in ordered_candidates:
        if not source_dir.exists():
            continue
        task_files = [p for p in source_dir.glob("*.json") if _is_task_like_json(p)]
        if not task_files:
            continue
        selected = random.sample(task_files, min(sample_count, len(task_files)))
        for i, task_path in enumerate(selected, 1):
            _copy_sample_with_aliases(task_path, sampled_tasks_dir, i)
        return source_dir, len(selected)

    return None, 0


def compute_calibration_stats(tasks_dir: str, model: str) -> dict:
    """Compute pass-rate and ToM-ratio stats from existing tasks."""
    from emtom.evolve.benchmark_wrapper import find_calibration_entry, cal_passed

    stats = {
        "passed": 0,
        "failed": 0,
        "untested": 0,
        "model": model,
        "tom_counts": {1: 0, 2: 0, 3: 0},
        "tom_total": 0,
        "tom_unknown": 0,
        "tom_ratios": {1: None, 2: None, 3: None},
    }
    tasks_path = Path(tasks_dir)

    if not tasks_path.exists():
        stats["total"] = 0
        stats["rate"] = None
        return stats

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task = json.load(f)

            tom_level = _infer_task_tom_level(task)
            if tom_level in (1, 2, 3):
                stats["tom_counts"][tom_level] += 1
                stats["tom_total"] += 1
            else:
                stats["tom_unknown"] += 1

            cal = find_calibration_entry(task.get("calibration", []), model=model)
            if cal is None:
                stats["untested"] += 1
            elif cal_passed(cal):
                stats["passed"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            continue

    stats["total"] = stats["passed"] + stats["failed"]
    stats["rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else None
    if stats["tom_total"] > 0:
        for level in (1, 2, 3):
            stats["tom_ratios"][level] = stats["tom_counts"][level] / stats["tom_total"]
    return stats


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    extra_args = getattr(main, "_extra_args", None)

    # Get Hydra output directory for logging
    from hydra.core.hydra_config import HydraConfig
    try:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_output_dir = None

    # Extract custom args from config (passed as +arg=value)
    num_tasks = config.get("num_tasks", 1)
    model = config.get("model", None)
    llm_provider = config.get("llm_provider", None)

    # Validate required parameters
    if not llm_provider:
        cprint(
            "Error: llm_provider is required. Use +llm_provider=openai_chat, +llm_provider=anthropic_claude, or +llm_provider=bedrock_claude",
            "red",
        )
        sys.exit(1)
    if not model:
        cprint("Error: model is required. Use +model=<model_name>", "red")
        sys.exit(1)
    output_dir = config.get("output_dir", "data/emtom/tasks")  # Final output location
    iterations_per_task = config.get("iterations_per_task", 100)
    quiet = config.get("quiet", False)
    subtasks_min = config.get("subtasks_min", 2)
    subtasks_max = config.get("subtasks_max", 5)
    agents_min = config.get("agents_min", 2)
    agents_max = config.get("agents_max", 2)
    seed = config.get("seed", None)

    # Get query from extra_args (parsed before Hydra to handle quoted strings)
    query = extra_args.query if extra_args else None
    retry_verification = extra_args.retry_verification if extra_args else None
    calibration_model = extra_args.calibration_model if extra_args else "gpt-5.2"
    target_pass_rate = extra_args.target_pass_rate if extra_args else 0.10
    category = extra_args.category if extra_args else None
    seed_task = extra_args.seed_task if extra_args else None
    judge_threshold = extra_args.judge_threshold if extra_args else None
    difficulty = extra_args.difficulty if extra_args else None
    test_model = extra_args.test_model if extra_args else None
    tom_target_l1 = extra_args.tom_target_l1 if extra_args else 0.40
    tom_target_l2 = extra_args.tom_target_l2 if extra_args else 0.40
    tom_target_l3 = extra_args.tom_target_l3 if extra_args else 0.20
    tom_ratio_tolerance = extra_args.tom_ratio_tolerance if extra_args else 0.08

    tom_target_sum = tom_target_l1 + tom_target_l2 + tom_target_l3
    if abs(tom_target_sum - 1.0) > 1e-6:
        cprint(
            f"Error: --tom-target-l1/2/3 must sum to 1.0, got {tom_target_sum:.6f}",
            "red",
        )
        sys.exit(1)
    if tom_ratio_tolerance < 0:
        cprint("Error: --tom-ratio-tolerance must be non-negative", "red")
        sys.exit(1)

    # Validate seed task path
    if seed_task:
        seed_task_path = Path(seed_task)
        if not seed_task_path.exists():
            cprint(f"ERROR: Seed task file not found: {seed_task_path}", "red")
            sys.exit(1)
        cprint(f"Seed task: {seed_task_path}", "green")

    # Load failed verification suggestions if retrying
    verification_feedback = None
    if retry_verification:
        retry_path = Path(retry_verification)
        cprint(f"[DEBUG] Looking for verification file: {retry_path}", "blue")
        cprint(f"[DEBUG] File exists: {retry_path.exists()}", "blue")
        if retry_path.exists():
            try:
                with open(retry_path) as f:
                    verification_data = json.load(f)
                cprint(f"[DEBUG] is_valid_tom in file: {verification_data.get('is_valid_tom', 'MISSING')}", "blue")
                if not verification_data.get("is_valid_tom", True):
                    verification_feedback = {
                        "suggestions": verification_data.get("suggestions", []),
                        "criteria": verification_data.get("criteria", {}),
                        "overall_reasoning": verification_data.get("overall_reasoning", ""),
                    }
                    cprint(f"Loaded failed verification from: {retry_path}", "yellow")
                    cprint(f"Suggestions to incorporate ({len(verification_feedback['suggestions'])}):", "yellow")
                    for i, s in enumerate(verification_feedback["suggestions"], 1):
                        cprint(f"  {i}. {s}", "yellow")
                else:
                    cprint(f"Verification already passed, ignoring: {retry_path}", "green")
            except Exception as e:
                cprint(f"Warning: Could not load verification file: {e}", "red")
        else:
            cprint(f"ERROR: Verification file not found: {retry_path}", "red")

    # Create unique temp working directory for this instance (allows parallel runs)
    instance_id = uuid.uuid4().hex[:8]
    working_dir = Path(tempfile.gettempdir()) / f"emtom_taskgen_{instance_id}"
    working_dir.mkdir(parents=True, exist_ok=True)

    # Sample example tasks for agent inspiration
    sampled_tasks_dir = working_dir / "sampled_tasks"
    sampled_tasks_dir.mkdir(parents=True, exist_ok=True)

    sampled_tasks_override = extra_args.sampled_tasks_dir if extra_args else None

    if sampled_tasks_override:
        override_path = Path(sampled_tasks_override)
        if override_path.exists():
            override_files = [p for p in override_path.glob("*.json") if _is_task_like_json(p)]
            if override_files:
                selected = sorted(override_files)[:10]
                for i, f in enumerate(selected, 1):
                    shutil.copy(f, sampled_tasks_dir / f.name)
                    _copy_sample_with_aliases(f, sampled_tasks_dir, i)
                cprint(
                    f"Using pre-built sampled_tasks: {override_path} "
                    f"({len(override_files)} files)",
                    "green",
                )
            else:
                cprint(
                    f"WARNING: --sampled-tasks-dir has no task JSONs: {override_path}, "
                    "falling back to defaults",
                    "yellow",
                )
                sampled_tasks_override = None
        else:
            cprint(f"WARNING: --sampled-tasks-dir not found: {override_path}, falling back to random", "yellow")
            sampled_tasks_override = None

    if not sampled_tasks_override:
        source_dir, count = populate_sampled_tasks_dir(
            sampled_tasks_dir=sampled_tasks_dir,
            primary_output_dir=output_dir,
            sample_count=10,
        )
        if source_dir is not None:
            cprint(f"Sampled {count} task examples from: {source_dir}", "green")
        else:
            cprint(
                "WARNING: No task examples found for sampled_tasks/. "
                "Generation will proceed without examples.",
                "yellow",
            )

    # Sample exploration trajectories for agent inspiration
    sampled_trajectories_dir = working_dir / "sampled_trajectories"
    sampled_trajectories_dir.mkdir(parents=True, exist_ok=True)
    trajectories_source = Path("outputs/emtom")
    if trajectories_source.exists():
        # Find all trajectory files in exploration output directories
        trajectory_files = list(trajectories_source.glob("*-exploration/results/trajectory_*.json"))
        if trajectory_files:
            sample_count = min(5, len(trajectory_files))
            sampled = random.sample(trajectory_files, sample_count)
            for i, traj_path in enumerate(sampled, 1):
                dest = sampled_trajectories_dir / f"trajectory_{i}.json"
                shutil.copy(traj_path, dest)

    # Setup config (registers Habitat plugins, sets seed, etc.)
    fix_config(config)
    config = setup_config(config, seed=seed or 47668090)

    # Compute calibration stats from existing dataset
    calibration_stats = compute_calibration_stats(output_dir, calibration_model)
    calibration_stats["target_rate"] = target_pass_rate
    calibration_stats["tom_target"] = {
        1: tom_target_l1,
        2: tom_target_l2,
        3: tom_target_l3,
    }
    calibration_stats["tom_tolerance"] = tom_ratio_tolerance

    cprint("=" * 60, "blue")
    cprint("EMTOM Task Generator (Live Scene Mode)", "blue")
    cprint("=" * 60, "blue")
    cprint(f"Instance: {instance_id}", "blue")
    cprint(f"Target tasks: {num_tasks}", "blue")
    cprint(f"Agents: {agents_min} - {agents_max}", "blue")
    cprint(f"LLM: {llm_provider} ({model})", "blue")
    cprint(f"Category: {category or 'random'}", "blue")
    if query:
        cprint(f"Query: {query}", "green")
    cprint(f"Working dir: {working_dir}", "blue")
    cprint(f"Output: {output_dir}", "blue")

    # Display calibration stats
    cprint(f"Calibration: {calibration_model} (target: {target_pass_rate:.0%})", "blue")
    if calibration_stats["rate"] is not None:
        cprint(f"  Current rate: {calibration_stats['rate']:.1%} ({calibration_stats['passed']}/{calibration_stats['total']})", "yellow")
    else:
        cprint(f"  No calibration data yet (untested: {calibration_stats['untested']})", "yellow")
    cprint(
        "ToM target mix: "
        f"L1 {tom_target_l1:.0%}, L2 {tom_target_l2:.0%}, L3 {tom_target_l3:.0%} "
        f"(tol +/-{tom_ratio_tolerance:.0%})",
        "blue",
    )
    if calibration_stats["tom_total"] > 0:
        cprint(
            "  Current ToM mix: "
            f"L1 {calibration_stats['tom_counts'][1]}/{calibration_stats['tom_total']} "
            f"({calibration_stats['tom_ratios'][1]:.1%}), "
            f"L2 {calibration_stats['tom_counts'][2]}/{calibration_stats['tom_total']} "
            f"({calibration_stats['tom_ratios'][2]:.1%}), "
            f"L3 {calibration_stats['tom_counts'][3]}/{calibration_stats['tom_total']} "
            f"({calibration_stats['tom_ratios'][3]:.1%})",
            "yellow",
        )
        if calibration_stats["tom_unknown"] > 0:
            cprint(f"  ToM unknown: {calibration_stats['tom_unknown']}", "yellow")
    else:
        cprint("  No ToM-labeled tasks yet for ratio calibration", "yellow")
    cprint("=" * 60, "blue")
    print()

    # Note: Scene is NOT loaded at startup
    # Agent will call new_scene[num_agents] to load scene with desired agent count
    cprint("Scene loading deferred to agent (call new_scene[num_agents] first)", "yellow")
    print()

    # Initialize LLM
    cprint("Initializing LLM client...", "blue")
    try:
        from habitat_llm.llm import instantiate_llm
        llm_client = instantiate_llm(llm_provider, generation_params={"model": model})
        cprint(f"Using model: {llm_client.generation_params.model}", "green")
    except Exception as e:
        cprint(f"Error initializing LLM: {e}", "red")
        sys.exit(1)

    # Create and run agent (scene_data=None - agent will call new_scene[num_agents] first)
    from emtom.task_gen.agent import TaskGeneratorAgent

    agent = TaskGeneratorAgent(
        llm_client=llm_client,
        config=config,
        working_dir=str(working_dir),  # Unique temp dir for this instance
        output_dir=output_dir,  # Shared output for all instances
        iterations_per_task=iterations_per_task,
        verbose=not quiet,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        agents_min=agents_min,
        agents_max=agents_max,
        scene_data=None,  # Agent will call new_scene[num_agents] to load scene
        log_dir=hydra_output_dir,  # Pass Hydra output directory for logs
        query=query,  # Optional seed query for task generation
        verification_feedback=verification_feedback,  # Failed ToM verification suggestions
        calibration_stats=calibration_stats,  # Dataset calibration stats for difficulty guidance
        category=category,  # Task category: cooperative, competitive, or mixed
        seed_task=seed_task,  # Existing task to use as seed instead of blank template
        judge_threshold=judge_threshold,  # Override judge threshold (None = use default)
        difficulty=difficulty,  # Difficulty context for judge: easy/medium/hard
        test_model=test_model,  # Override model for test_task calibration
    )

    # Run agent
    try:
        submitted_tasks = agent.run(num_tasks_target=num_tasks)
    finally:
        # Clean up temp working directory
        agent.close()
        shutil.rmtree(working_dir, ignore_errors=True)
        cprint(f"Cleaned up temp directory: {working_dir}", "blue")

    # Summary
    print()
    cprint("=" * 60, "green")
    cprint("Generation Complete", "green")
    cprint("=" * 60, "green")
    cprint(f"Tasks generated: {len(submitted_tasks)}", "green")
    for task_path in submitted_tasks:
        cprint(f"  - {task_path}", "green")
    print()


if __name__ == "__main__":
    extra_args = parse_extra_args()
    main._extra_args = extra_args
    main()
