#!/usr/bin/env python3
"""
Run EMTOM benchmark with Habitat integration.

This script runs EMTOM tasks in the Habitat simulator with LLM planners
for multi-agent evaluation.

Usage:
    ./emtom/run_emtom.sh benchmark --max-sim-steps 1000
"""

import json
import sys
from pathlib import Path

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


def load_tasks(task_file: str) -> list:
    """Load tasks from JSON file."""
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    for task_data in data.get("tasks", []):
        task = GeneratedTask.from_dict(task_data)
        tasks.append(task)

    return tasks


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Ensure video saving is enabled
    with open_dict(config):
        config.evaluation.save_video = True

    cprint("\n" + "=" * 60, "blue")
    cprint("EMTOM Habitat Benchmark", "blue")
    cprint("=" * 60, "blue")

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

    # Load EMTOM tasks
    task_dir = Path("data/emtom/tasks")
    if not task_dir.exists():
        cprint(f"ERROR: Task directory not found: {task_dir}", "red")
        cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
        sys.exit(1)

    # Use most recent generated task file
    task_files = list(task_dir.glob("emtom_challenges_*.json"))
    if not task_files:
        cprint(f"ERROR: No task files found in {task_dir}", "red")
        cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
        sys.exit(1)
    task_file = sorted(task_files)[-1]
    cprint(f"Using task file: {task_file}", "blue")

    tasks = load_tasks(str(task_file))
    if not tasks:
        cprint(f"ERROR: No tasks found in {task_file}", "red")
        sys.exit(1)

    cprint(f"Loaded {len(tasks)} EMTOM tasks", "green")

    # Run first task
    task = tasks[0]
    output_dir = config.paths.results_dir

    # Create and setup benchmark runner
    from emtom.runner import BenchmarkRunner
    from emtom.runner.benchmark import task_to_instruction

    runner = BenchmarkRunner(config)
    runner.setup(
        env_interface=env_interface,
        task=task,
        output_dir=output_dir,
    )

    # Print task info
    instruction = task_to_instruction(task)

    cprint(f"\n{'=' * 60}", "blue")
    cprint(f"EMTOM TASK: {task.title}", "blue")
    cprint(f"{'=' * 60}", "blue")
    print(f"Public Goal: {task.public_goal}")
    print(f"Mechanics: {task.active_mechanics}")
    if task.mechanic_bindings:
        print(f"Mechanic bindings: {len(task.mechanic_bindings)} active")
        for b in task.mechanic_bindings:
            print(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}")
    print(f"\nPer-agent instructions:")
    for agent_id, instr in instruction.items():
        print(f"\n--- {agent_id} ---")
        print(instr)

    # Print agent info
    cprint(f"\nAgents: {list(runner.agents.keys())}", "blue")
    for uid, agent in runner.agents.items():
        cprint(f"  agent_{uid} tools: {list(agent.tools.keys())}", "blue")

    # Run benchmark
    try:
        cprint("\nStarting task execution with LLM planners...", "blue")
        results = runner.run(instruction=instruction)

        cprint("\nTask execution completed!", "green")
        print(f"Steps: {results['steps']}")
        print(f"Done: {results['done']}")
        print(f"Episode over: {results.get('episode_over', False)}")

    except Exception as e:
        error_str = str(e)

        # Check if this is a timeout (episode over) vs a real error
        is_timeout = "Episode over" in error_str or "call reset before calling step" in error_str

        if is_timeout:
            cprint(f"\nTask timed out (max simulation steps reached)", "yellow")
            cprint("This is normal - the task just needs more steps to complete.", "yellow")
        else:
            cprint(f"Error during task execution: {e}", "red")
            import traceback
            traceback.print_exc()

    # Cleanup
    runner.cleanup()
    cprint("\nBenchmark complete!", "green")
    cprint(f"Check {output_dir} for videos and logs", "blue")


if __name__ == "__main__":
    cprint("\nEMTOM Habitat Benchmark Runner", "blue")
    cprint("This script runs EMTOM tasks in Habitat with LLM planners and video recording.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/emtom_two_robots", "yellow")
        sys.exit(1)

    main()
