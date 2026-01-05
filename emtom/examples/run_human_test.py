#!/usr/bin/env python3
# isort: skip_file
"""
Human-in-the-loop (HITL) testing mode for EMTOM benchmark.

Uses the unified BenchmarkRunner with human_agents parameter to enable
human control of agents instead of (or alongside) LLM control.

Usage:
    # Run with task file (human controls all agents)
    ./emtom/run_emtom.sh test --task data/emtom/tasks/emtom_challenges_xxx.json

    # Run with one LLM agent
    ./emtom/run_emtom.sh test --task data/emtom/tasks/emtom_challenges_xxx.json --llm-agents agent_1

    # Run with specific mechanics (no task file)
    ./emtom/run_emtom.sh test --mechanics inverse_state remote_control
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, open_dict

from habitat_llm.utils import setup_config, fix_config, cprint
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

from emtom import list_mechanics
from emtom.task_gen import GeneratedTask


def print_task_info(task: GeneratedTask) -> None:
    """Pretty print task information for human readability."""
    cprint("\n" + "=" * 66, "blue")
    cprint("TASK INFORMATION", "blue")
    cprint("=" * 66, "blue")

    if task.title:
        cprint(f"\nTitle: {task.title}", "blue")
    if task.task_id:
        cprint(f"Task ID: {task.task_id}", "gray")

    if task.task:
        cprint(f"\nTask:", "blue")
        cprint(f"  {task.task}", "gray")

    if task.agent_roles:
        cprint("\nAgent Roles:", "blue")
        for agent_id, role in task.agent_roles.items():
            cprint(f"  {agent_id}: {role}", "gray")

    if task.agent_secrets:
        cprint("\nAgent Secrets:", "yellow")
        for agent_id, secrets in task.agent_secrets.items():
            cprint(f"  {agent_id}:", "yellow")
            for secret in secrets:
                cprint(f"    - {secret}", "gray")

    if task.agent_actions:
        cprint("\nAgent Actions:", "blue")
        for agent_id, actions in task.agent_actions.items():
            cprint(f"  {agent_id}: {', '.join(actions)}", "gray")

    cprint("\n" + "-" * 66, "blue")


def print_mechanics_info(task: GeneratedTask) -> None:
    """Print mechanics from task."""
    if not task.mechanic_bindings:
        cprint("Mechanics: none", "gray")
        return

    cprint("\nMechanics:", "blue")
    for binding in task.mechanic_bindings:
        mech_type = binding.mechanic_type
        trigger = binding.trigger_object or "?"
        target = binding.target_object or "self"
        cprint(f"  - {mech_type}: {trigger} -> {target}", "gray")


def load_task(task_file: str) -> Optional[GeneratedTask]:
    """Load task from JSON file.

    Supports two formats:
    - Bundle format: {"tasks": [task1, task2, ...]}
    - Single task format: {task_id, title, ...}
    """
    with open(task_file) as f:
        data = json.load(f)

    # Check if it's a bundle (has "tasks" array) or single task
    if "tasks" in data:
        tasks = data["tasks"]
        if tasks:
            return GeneratedTask.from_dict(tasks[0])
    elif "task_id" in data:
        return GeneratedTask.from_dict(data)

    return None


def parse_extra_args():
    """Parse extra CLI arguments before Hydra."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default=None,
                        help="Task file to load")
    parser.add_argument("--mechanics", type=str, nargs="*", default=None,
                        help="Mechanics to enable (e.g., inverse_state remote_control)")
    parser.add_argument("--llm-agents", type=str, nargs="*", default=None,
                        help="Agents to make LLM-controlled (e.g., agent_0 agent_1)")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    return args


@hydra.main(config_path="../../habitat_llm/conf", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    extra_args = getattr(main, "_extra_args", None)

    # Setup config
    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Ensure video saving is enabled
    with open_dict(config):
        config.evaluation.save_video = True

    cprint("\n" + "=" * 60, "blue")
    cprint("EMTOM Human Test Mode", "blue")
    cprint("=" * 60, "blue")

    # Register Habitat components
    cprint("Registering Habitat components...", "blue")
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = None
    try:
        dataset = CollaborationDatasetV0(config.habitat.dataset)
        cprint(f"Loaded dataset with {len(dataset.episodes)} episodes", "green")
    except Exception as e:
        cprint(f"Warning: Could not load dataset: {e}", "yellow")

    # Create environment interface
    cprint("Initializing Habitat environment...", "blue")
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        cprint(f"Warning: Failed to initialize world graph: {e}", "yellow")

    cprint("Environment initialized!", "green")

    # Determine which agents are human vs LLM
    llm_agents = []
    if extra_args and extra_args.llm_agents:
        llm_agents = extra_args.llm_agents

    # Calculate human agents (all agents minus LLM agents)
    all_agent_ids = ["agent_0", "agent_1"]
    human_agents = [a for a in all_agent_ids if a not in llm_agents]

    cprint(f"Human-controlled: {human_agents}", "green")
    cprint(f"LLM-controlled: {llm_agents}", "green")

    # Load task or setup mechanics
    task: Optional[GeneratedTask] = None
    task_data = {"mechanics": []}
    is_task_mode = False

    if extra_args and extra_args.task:
        task = load_task(extra_args.task)
        if task:
            is_task_mode = True
            print_task_info(task)

            # Reset environment to the correct episode
            if task.episode_id and task.episode_id != "unknown":
                cprint(f"Resetting environment to episode: {task.episode_id}", "blue")
                try:
                    env_interface.reset_environment(episode_id=task.episode_id)
                    cprint(f"Successfully loaded episode {task.episode_id}", "green")
                except (ValueError, IndexError) as e:
                    cprint(f"Warning: Could not load episode {task.episode_id}: {e}", "yellow")

            print_mechanics_info(task)

    elif extra_args and extra_args.mechanics:
        cprint(f"Using mechanics from CLI: {extra_args.mechanics}", "green")
        for mech_name in extra_args.mechanics:
            task_data["mechanics"].append({"mechanic_type": mech_name})

    else:
        # Default exploration mode: use all mechanics
        all_mechanics = list_mechanics()
        cprint(f"Using all mechanics: {all_mechanics}", "yellow")
        for mech_name in all_mechanics:
            task_data["mechanics"].append({"mechanic_type": mech_name})

    # Output directory
    output_dir = config.paths.results_dir if hasattr(config, 'paths') else "outputs/emtom/human_test"

    # Create and setup unified BenchmarkRunner with human_agents
    from emtom.runner import BenchmarkRunner
    from emtom.runner.benchmark import task_to_instruction

    runner = BenchmarkRunner(config)
    runner.setup(
        env_interface=env_interface,
        task_data=task_data if not task else None,
        output_dir=output_dir,
        task=task,
        human_agents=human_agents,
        save_video=True,
    )

    # Auto-bind mechanics only in exploration mode (not task mode)
    if not is_task_mode and runner.game_manager:
        state, bindings = runner.game_manager.auto_bind_mechanics()
        if bindings:
            cprint("\nAuto-bound mechanics:", "green")
            for mech_type, info in bindings.items():
                cprint(f"  - {mech_type}: {info}", "gray")

    if runner.game_manager:
        active = runner.game_manager.get_debug_info()['active_mechanics']
        cprint(f"Active mechanics: {', '.join(active)}", "green")

    # Build instruction
    if task:
        instruction = task_to_instruction(task)
    else:
        instruction = {
            "agent_0": "Explore the environment and discover mechanics.",
            "agent_1": "Explore the environment and discover mechanics.",
        }

    # Run interactive test
    try:
        results = runner.run(instruction=instruction)
        cprint(f"\nCompleted: {results['turns']} turns, {results['steps']} steps", "green")
        if results.get("success"):
            cprint("Task SUCCESS!", "green")
    except KeyboardInterrupt:
        cprint("\nInterrupted by user", "yellow")
    except Exception as e:
        cprint(f"\nError: {e}", "red")
        import traceback
        traceback.print_exc()

    runner.cleanup()


if __name__ == "__main__":
    extra_args = parse_extra_args()
    main._extra_args = extra_args
    main()
