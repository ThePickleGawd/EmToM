#!/usr/bin/env python3
"""
Run EMTOM benchmark with Habitat integration.

This script runs EMTOM tasks in the Habitat simulator with LLM planners
for multi-agent evaluation.

By default, runs ALL tasks in data/emtom/tasks/. Use --task to run a single task.

Usage:
    ./emtom/run_emtom.sh benchmark                    # Run all tasks
    ./emtom/run_emtom.sh benchmark --model sonnet     # Run all tasks with Claude Sonnet
    ./emtom/run_emtom.sh benchmark --task task.json   # Run single task
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, open_dict

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, setup_config, fix_config

from emtom.task_gen import GeneratedTask


def load_tasks_from_file(task_file: str) -> Tuple[List[GeneratedTask], List[Dict]]:
    """Load tasks from a single JSON file.

    Supports two formats:
    - Bundle format: {"tasks": [task1, task2, ...]}
    - Single task format: {task_id, title, ...}

    Returns:
        Tuple of (tasks list, raw data list) - raw data includes golden_trajectory
    """
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    raw_data = []

    # Check if it's a bundle (has "tasks" array) or single task
    if "tasks" in data:
        # Bundle format
        for task_data in data["tasks"]:
            task = GeneratedTask.from_dict(task_data)
            tasks.append(task)
            raw_data.append(task_data)
    elif "task_id" in data:
        # Single task format
        task = GeneratedTask.from_dict(data)
        tasks.append(task)
        raw_data.append(data)

    return tasks, raw_data


def load_all_tasks(task_dir: Path) -> Tuple[List[GeneratedTask], List[Dict]]:
    """Load all tasks from a directory.

    Returns:
        Tuple of (tasks list, raw data list)
    """
    tasks = []
    raw_data = []

    # Find all JSON files in the directory
    json_files = sorted(task_dir.glob("*.json"))

    for task_file in json_files:
        try:
            file_tasks, file_raw = load_tasks_from_file(str(task_file))
            tasks.extend(file_tasks)
            raw_data.extend(file_raw)
        except Exception as e:
            cprint(f"Warning: Could not load {task_file.name}: {e}", "yellow")

    return tasks, raw_data


def run_single_task(
    config: DictConfig,
    env_interface: EnvironmentInterface,
    task: GeneratedTask,
    task_raw: Dict[str, Any],
    output_dir: str,
    task_index: int = 0,
    total_tasks: int = 1,
) -> Dict[str, Any]:
    """Run benchmark on a single task.

    Returns:
        Results dict with success, steps, turns, etc.
    """
    from emtom.runner import BenchmarkRunner
    from emtom.runner.benchmark import task_to_instruction

    task_id = task.task_id
    prefix = f"[{task_index + 1}/{total_tasks}]" if total_tasks > 1 else ""

    cprint(f"\n{'=' * 60}", "blue")
    cprint(f"{prefix} TASK: {task.title}", "blue")
    cprint(f"{'=' * 60}", "blue")
    print(f"Task ID: {task_id}")
    print(f"Episode ID: {task.episode_id} (Scene: {task.scene_id})")
    if task.task:
        print(f"\n[Task]: {task.task}\n")
    print(f"Mechanics: {task.active_mechanics}")
    if task.mechanic_bindings:
        print(f"Mechanic bindings: {len(task.mechanic_bindings)} active")
        for b in task.mechanic_bindings:
            print(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}")

    # Validate num_agents matches config
    task_num_agents = task.num_agents
    config_num_agents = len(config.evaluation.agents)
    if task_num_agents != config_num_agents:
        cprint(f"SKIP: Task requires {task_num_agents} agents but config has {config_num_agents}", "yellow")
        return {
            "task_id": task_id,
            "title": task.title,
            "skipped": True,
            "skip_reason": f"Agent count mismatch: task needs {task_num_agents}, config has {config_num_agents}",
            "success": False,
        }

    # Reset environment to the correct episode for this task
    if task.episode_id and task.episode_id != "unknown":
        cprint(f"Resetting environment to episode: {task.episode_id}", "blue")
        try:
            env_interface.reset_environment(episode_id=task.episode_id)
            cprint(f"Successfully loaded episode {task.episode_id}", "green")
        except (ValueError, IndexError) as e:
            cprint(f"SKIP: Could not load episode {task.episode_id}: {e}", "yellow")
            return {
                "task_id": task_id,
                "title": task.title,
                "skipped": True,
                "skip_reason": f"Episode not found: {task.episode_id}",
                "success": False,
            }

    # Create task-specific output directory
    task_output_dir = f"{output_dir}/{task_id}"
    Path(task_output_dir).mkdir(parents=True, exist_ok=True)

    # Create and setup benchmark runner
    runner = BenchmarkRunner(config)
    runner.setup(
        env_interface=env_interface,
        task=task,
        output_dir=task_output_dir,
    )

    # Build instruction
    instruction = task_to_instruction(task)

    print(f"\nPer-agent instructions:")
    for agent_id, instr in instruction.items():
        print(f"\n--- {agent_id} ---")
        print(instr)

    # Print agent info
    cprint(f"\nAgents: {list(runner.agents.keys())}", "blue")
    for uid, agent in runner.agents.items():
        cprint(f"  agent_{uid} tools: {list(agent.tools.keys())}", "blue")

    # Get max steps from config
    max_steps = config.habitat.environment.get("max_episode_steps", 2000)

    # Calculate max turns as 5x golden trajectory length
    golden_trajectory = task_raw.get("golden_trajectory", [])
    if "max_turns" in config:
        max_turns = config.max_turns
    else:
        max_turns = max(len(golden_trajectory) * 5, 20)  # Minimum 20 turns

    cprint(f"\nMax simulation steps: {max_steps}", "blue")
    cprint(f"Max LLM turns: {max_turns} (golden trajectory: {len(golden_trajectory)} steps)", "blue")

    # Run benchmark
    results = {
        "task_id": task_id,
        "title": task.title,
        "skipped": False,
        "success": False,
        "steps": 0,
        "turns": 0,
        "error": None,
    }

    try:
        cprint("Starting task execution with LLM planners...", "blue")
        run_results = runner.run(instruction=instruction, max_steps=max_steps, max_turns=max_turns)

        results["success"] = run_results.get("success", False)
        results["steps"] = run_results.get("steps", 0)
        results["turns"] = run_results.get("turns", 0)
        results["done"] = run_results.get("done", False)
        results["episode_over"] = run_results.get("episode_over", False)
        results["evaluation"] = run_results.get("evaluation", {})

        if results["success"]:
            cprint(f"\n✓ TASK PASSED: {task.title}", "green")
        else:
            cprint(f"\n✗ TASK FAILED: {task.title}", "red")

        print(f"Steps: {results['steps']}, Turns: {results['turns']}")

    except Exception as e:
        error_str = str(e)
        is_timeout = "Episode over" in error_str or "call reset before calling step" in error_str

        if is_timeout:
            cprint(f"\nTask timed out (max simulation steps reached)", "yellow")
            results["error"] = "timeout"
        else:
            cprint(f"Error during task execution: {e}", "red")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)

    return results


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Get model and llm_provider from config (passed via +model=X +llm_provider=Y)
    model = config.get("model", "gpt-5.2")
    llm_provider = config.get("llm_provider", "openai_chat")

    # Override LLM config for all agents
    with open_dict(config):
        if not hasattr(config.evaluation, 'save_video'):
            config.evaluation.save_video = True

        # Override each agent's planner LLM configuration
        if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'agents'):
            import habitat_llm
            import os
            from omegaconf import OmegaConf

            # Load the base LLM config
            habitat_llm_dir = os.path.dirname(habitat_llm.__file__)
            llm_config_path = f"{habitat_llm_dir}/conf/llm/{llm_provider}.yaml"

            if os.path.exists(llm_config_path):
                base_llm_config = OmegaConf.load(llm_config_path)
                base_llm_config.generation_params.model = model

                # Apply to all agents
                for agent_key in config.evaluation.agents:
                    agent_conf = config.evaluation.agents[agent_key]
                    if hasattr(agent_conf, 'planner') and hasattr(agent_conf.planner, 'plan_config'):
                        agent_conf.planner.plan_config.llm = base_llm_config

    cprint("\n" + "=" * 60, "blue")
    cprint("EMTOM Habitat Benchmark", "blue")
    cprint("=" * 60, "blue")
    cprint(f"LLM: {llm_provider} ({model})", "blue")

    # Register Habitat components
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    cprint(f"Loaded dataset with {len(dataset.episodes)} episodes", "green")

    # Create environment interface
    cprint("Initializing Habitat environment...", "blue")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        cprint(f"Warning: Failed to initialize world graph: {e}", "yellow")

    cprint("Environment initialized!", "green")

    # Determine which tasks to run
    task_file_arg = config.get("task", None)
    num_agents_filter = config.get("num_agents_filter", None)
    task_dir = Path("data/emtom/tasks")

    if task_file_arg:
        # Single task mode: run only the specified task
        task_file = Path(task_file_arg)
        if not task_file.exists():
            cprint(f"ERROR: Task file not found: {task_file}", "red")
            sys.exit(1)
        cprint(f"Single task mode: {task_file}", "blue")
        tasks, raw_data = load_tasks_from_file(str(task_file))
    else:
        # All tasks mode: run all tasks in the directory
        if not task_dir.exists():
            cprint(f"ERROR: Task directory not found: {task_dir}", "red")
            cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
            sys.exit(1)

        tasks, raw_data = load_all_tasks(task_dir)

        if not tasks:
            cprint(f"ERROR: No tasks found in {task_dir}", "red")
            cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
            sys.exit(1)

        # Filter by agent count if specified
        if num_agents_filter:
            filtered_tasks = []
            filtered_raw = []
            for task, raw in zip(tasks, raw_data):
                if task.num_agents == num_agents_filter:
                    filtered_tasks.append(task)
                    filtered_raw.append(raw)
            tasks, raw_data = filtered_tasks, filtered_raw

            if not tasks:
                cprint(f"No tasks found with {num_agents_filter} agents", "yellow")
                return

            cprint(f"Running {len(tasks)} tasks with {num_agents_filter} agents", "blue")
        else:
            cprint(f"All tasks mode: {len(tasks)} tasks found", "blue")

    output_dir = config.paths.results_dir

    # Run all tasks
    all_results = []
    for i, (task, task_raw) in enumerate(zip(tasks, raw_data)):
        result = run_single_task(
            config=config,
            env_interface=env_interface,
            task=task,
            task_raw=task_raw,
            output_dir=output_dir,
            task_index=i,
            total_tasks=len(tasks),
        )
        all_results.append(result)

    # Close environment after all tasks are done
    try:
        env_interface.env.close()
    except Exception:
        pass

    # Print summary
    cprint("\n" + "=" * 60, "blue")
    cprint("BENCHMARK SUMMARY", "blue")
    cprint("=" * 60, "blue")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("success"))
    failed = sum(1 for r in all_results if not r.get("skipped") and not r.get("success"))
    skipped = sum(1 for r in all_results if r.get("skipped"))

    cprint(f"Total tasks: {total}", "blue")
    cprint(f"  Passed:  {passed}", "green" if passed > 0 else "blue")
    cprint(f"  Failed:  {failed}", "red" if failed > 0 else "blue")
    cprint(f"  Skipped: {skipped}", "yellow" if skipped > 0 else "blue")

    if total > 0:
        pass_rate = passed / (total - skipped) * 100 if (total - skipped) > 0 else 0
        cprint(f"\nPass rate: {pass_rate:.1f}%", "green" if pass_rate > 50 else "red")

    # Print per-task results
    print("\nPer-task results:")
    for r in all_results:
        status = "✓ PASS" if r.get("success") else ("SKIP" if r.get("skipped") else "✗ FAIL")
        color = "green" if r.get("success") else ("yellow" if r.get("skipped") else "red")
        reason = f" ({r.get('skip_reason', r.get('error', ''))})" if not r.get("success") else ""
        cprint(f"  [{status}] {r['task_id']}: {r['title']}{reason}", color)

    # Save summary to file
    summary_file = Path(output_dir) / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "model": model,
            "llm_provider": llm_provider,
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": pass_rate if total > 0 else 0,
            "results": all_results,
        }, f, indent=2)

    cprint(f"\nResults saved to: {summary_file}", "blue")
    cprint("Benchmark complete!", "green")


if __name__ == "__main__":
    cprint("\nEMTOM Habitat Benchmark Runner", "blue")
    cprint("This script runs EMTOM tasks in Habitat with LLM planners.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/emtom_2_robots", "yellow")
        sys.exit(1)

    main()
