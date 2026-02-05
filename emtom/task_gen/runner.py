#!/usr/bin/env python3
"""
Entry point for the agentic task generator.

Usage:
    python emtom/task_gen/runner.py --config-name examples/emtom_2_robots +llm_provider=openai_chat +model=gpt-5

Or via shell script:
    ./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5
    ./emtom/run_emtom.sh generate --llm bedrock_claude --model sonnet
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

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
    parser.add_argument("--target-pass-rate", type=float, default=0.10,
                        help="Target pass rate for calibration model (default: 0.10 = 10%%)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["cooperative", "competitive", "mixed"],
                        help="Task category to generate (default: random)")
    parser.add_argument("--seed-task", type=str, default=None,
                        help="Path to existing task JSON to use as seed (instead of blank template)")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    return args


def compute_calibration_stats(tasks_dir: str, model: str) -> dict:
    """Compute pass rate statistics for a given model from existing tasks."""
    stats = {"passed": 0, "failed": 0, "untested": 0, "model": model}
    tasks_path = Path(tasks_dir)

    if not tasks_path.exists():
        stats["total"] = 0
        stats["rate"] = None
        return stats

    for task_file in tasks_path.glob("*.json"):
        try:
            with open(task_file) as f:
                task = json.load(f)
            cal = task.get("calibration", {}).get(model)
            if cal is None:
                stats["untested"] += 1
            elif cal.get("passed"):
                stats["passed"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            continue

    stats["total"] = stats["passed"] + stats["failed"]
    stats["rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else None
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
        cprint("Error: llm_provider is required. Use +llm_provider=openai_chat or +llm_provider=bedrock_claude", "red")
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

    # Sample reference tasks from existing pool for agent inspiration
    import random
    sampled_tasks_dir = working_dir / "sampled_tasks"
    sampled_tasks_dir.mkdir(parents=True, exist_ok=True)
    tasks_source = Path(output_dir)
    if tasks_source.exists():
        existing_tasks = list(tasks_source.glob("*.json"))
        if existing_tasks:
            sample_count = min(10, len(existing_tasks))
            sampled = random.sample(existing_tasks, sample_count)
            for i, task_path in enumerate(sampled, 1):
                dest = sampled_tasks_dir / f"task_{i}.json"
                shutil.copy(task_path, dest)

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
