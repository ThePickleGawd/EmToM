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
    """Load tasks from JSON file.

    Supports two formats:
    - Bundle format: {"tasks": [task1, task2, ...]}
    - Single task format: {task_id, title, ...}
    """
    with open(task_file) as f:
        data = json.load(f)

    tasks = []

    # Check if it's a bundle (has "tasks" array) or single task
    if "tasks" in data:
        # Bundle format
        for task_data in data["tasks"]:
            task = GeneratedTask.from_dict(task_data)
            tasks.append(task)
    elif "task_id" in data:
        # Single task format
        task = GeneratedTask.from_dict(data)
        tasks.append(task)

    return tasks


def load_all_tasks_from_dir(task_dir: Path) -> list:
    """Load all tasks from a directory.

    Looks for:
    - Individual task files: task_*.json
    - Bundle files: emtom_challenges_*.json, tasks_*.json
    """
    tasks = []

    # Find all JSON files in the directory
    json_files = list(task_dir.glob("*.json"))

    for task_file in json_files:
        try:
            file_tasks = load_tasks(str(task_file))
            tasks.extend(file_tasks)
        except Exception as e:
            print(f"Warning: Could not load {task_file}: {e}")

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

    # Load EMTOM task - either from --task argument or most recent in task_dir
    task_file_arg = config.get("task", None)

    if task_file_arg:
        # Use explicitly specified task file
        task_file = Path(task_file_arg)
        if not task_file.exists():
            cprint(f"ERROR: Task file not found: {task_file}", "red")
            sys.exit(1)
        cprint(f"Loading specified task: {task_file}", "blue")
    else:
        # Fall back to most recent task in task_dir
        task_dir = Path("data/emtom/tasks")

        if not task_dir.exists():
            cprint(f"ERROR: Task directory not found: {task_dir}", "red")
            cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
            sys.exit(1)

        # Find all task JSON files and sort by modification time (most recent first)
        task_files = list(task_dir.glob("*.json"))
        if not task_files:
            cprint(f"ERROR: No task files found in {task_dir}", "red")
            cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
            sys.exit(1)

        task_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        task_file = task_files[0]
        cprint(f"Loading most recent task: {task_file.name}", "blue")

    tasks = load_tasks(str(task_file))

    if not tasks:
        cprint(f"ERROR: Failed to parse task file: {task_file}", "red")
        sys.exit(1)

    task = tasks[0]
    cprint(f"Loaded task: {task.title}", "green")
    output_dir = config.paths.results_dir

    # Validate num_agents matches config
    task_num_agents = task.num_agents
    config_num_agents = len(config.evaluation.agents)
    if task_num_agents != config_num_agents:
        cprint(f"ERROR: Task requires {task_num_agents} agents but config has {config_num_agents}", "red")
        cprint(f"Use: ./emtom/run_emtom.sh benchmark --num-agents {task_num_agents} --task {task_file}", "yellow")
        sys.exit(1)

    # Reset environment to the correct episode for this task
    if task.episode_id and task.episode_id != "unknown":
        cprint(f"Resetting environment to episode: {task.episode_id}", "blue")
        try:
            env_interface.reset_environment(episode_id=task.episode_id)
            cprint(f"Successfully loaded episode {task.episode_id}", "green")
        except (ValueError, IndexError) as e:
            cprint(f"Warning: Could not load episode {task.episode_id}: {e}", "yellow")
            cprint("Continuing with current episode...", "yellow")

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
    print(f"Episode ID: {task.episode_id} (Scene: {task.scene_id})")
    if task.task:
        print(f"\n[Task]: {task.task}\n")
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

    # Get max steps from config (set via --max-sim-steps or Hydra)
    max_steps = config.habitat.environment.get("max_episode_steps", 2000)

    # Get max turns from config (set via +max_turns=N)
    max_turns = config.get("max_turns", 20)

    cprint(f"\nMax simulation steps: {max_steps}", "blue")
    cprint(f"Max LLM turns: {max_turns}", "blue")

    # Run benchmark
    try:
        cprint("Starting task execution with LLM planners...", "blue")
        results = runner.run(instruction=instruction, max_steps=max_steps, max_turns=max_turns)

        cprint("\nTask execution completed!", "green")
        print(f"Steps: {results['steps']}")
        print(f"LLM turns: {results['turns']}")
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
        cprint("Example: python run_habitat_benchmark.py --config-name examples/emtom_2_robots", "yellow")
        sys.exit(1)

    main()
