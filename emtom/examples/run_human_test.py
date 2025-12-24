#!/usr/bin/env python3
# isort: skip_file
"""
Human-in-the-loop (HITL) testing mode for EMTOM benchmark.

This script mirrors the benchmark setup but allows human control
of agents instead of (or alongside) LLM control.

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


def print_task_info(task_info: Dict[str, Any]) -> None:
    """Pretty print task information for human readability."""
    cprint("\n╔══════════════════════════════════════════════════════════════╗", "blue")
    cprint("║                       TASK INFORMATION                        ║", "blue")
    cprint("╚══════════════════════════════════════════════════════════════╝", "blue")

    # Title and ID
    if task_info.get("title"):
        cprint(f"\nTitle: {task_info['title']}", "blue")
    if task_info.get("task_id"):
        cprint(f"Task ID: {task_info['task_id']}", "gray")

    # Story / atmosphere
    if task_info.get("story"):
        cprint(f"\nStory:", "blue")
        cprint(f"  {task_info['story']}", "gray")

    # Public goal and context
    if task_info.get("public_goal"):
        cprint(f"\nPublic Goal: {task_info['public_goal']}", "blue")
    if task_info.get("public_context"):
        cprint(f"Context: {task_info['public_context']}", "gray")

    # Agent roles
    agent_roles = task_info.get("agent_roles", {})
    if agent_roles:
        cprint("\nAgent Roles:", "blue")
        for agent_id, role in agent_roles.items():
            cprint(f"  {agent_id}: {role}", "gray")

    # Agent secrets (the vital info!)
    agent_secrets = task_info.get("agent_secrets", {})
    if agent_secrets:
        cprint("\n⚠️  Agent Secrets:", "yellow")
        for agent_id, secrets in agent_secrets.items():
            cprint(f"  {agent_id}:", "yellow")
            for secret in secrets:
                cprint(f"    • {secret}", "gray")

    cprint("\n" + "-" * 66, "blue")


def print_bindings_pretty(bindings: Dict[str, Any]) -> None:
    """Pretty print auto-bound mechanics for human readability."""
    cprint("\n╔══════════════════════════════════════════════════════════════╗", "green")
    cprint("║                    AUTO-BOUND MECHANICS                       ║", "green")
    cprint("╚══════════════════════════════════════════════════════════════╝", "green")

    # Standard mechanics
    mechanics_printed = False
    for mech in ["inverse_state", "remote_control", "state_mirroring", "conditional_unlock"]:
        if mech in bindings:
            if not mechanics_printed:
                cprint("\nMechanics:", "blue")
                mechanics_printed = True
            info = bindings[mech]
            if mech == "inverse_state":
                cprint(f"   - Inverse State: {info.get('target')}", "gray")
            elif mech == "remote_control":
                cprint(f"   - Remote Control: {info.get('trigger')} -> {info.get('target')}", "gray")
            elif mech == "state_mirroring":
                pair = info.get('pair', [])
                cprint(f"   - State Mirroring: {pair[0] if pair else '?'} <-> {pair[1] if len(pair) > 1 else '?'}", "gray")
            elif mech == "conditional_unlock":
                cprint(f"   - Conditional Unlock: {info.get('prerequisite')} unlocks -> {info.get('target')}", "gray")

    # Hidden items
    if "hidden_items" in bindings:
        cprint("Hidden Items:", "blue")
        for container, item in bindings["hidden_items"].items():
            item_name = bindings.get("item_definitions", {}).get(item, item)
            cprint(f"   - {item_name} hidden in {container}", "gray")

    # Clues
    if "clues" in bindings:
        cprint("\nClues:", "blue")
        for i, clue in enumerate(bindings["clues"], 1):
            clue_type = clue.get("type", "unknown").capitalize()
            cprint(f"   {i}. [{clue_type}] \"{clue.get('text', '')}\"", "gray")

    # Suggested locations
    if "suggested_locations" in bindings:
        locs = bindings["suggested_locations"]
        cprint(f"\nSuggested Locations: {', '.join(locs)}", "blue")

    cprint("\n" + "-" * 66, "green")


def load_task(task_file: str) -> Optional[Dict[str, Any]]:
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
            return tasks[0]
    elif "task_id" in data:
        # Single task format - return as-is
        return data

    return None


def task_to_instruction(task_info: Dict[str, Any]) -> Dict[str, str]:
    """Convert task info to per-agent instructions."""
    instructions = {}

    agent_roles = task_info.get("agent_roles", {})
    if not agent_roles:
        agent_roles = {"agent_0": "Agent 0", "agent_1": "Agent 1"}

    for agent_id in agent_roles.keys():
        parts = []

        # Add atmospheric story first (sets the scene)
        if task_info.get("story"):
            parts.append(task_info["story"])
            parts.append("")  # blank line

        parts.append(f"Goal: {task_info.get('public_goal', 'Complete the task')}")

        if task_info.get("public_context"):
            parts.append(task_info["public_context"])

        # Per-agent secrets
        secrets = task_info.get("agent_secrets", {}).get(agent_id, [])
        if secrets:
            parts.append("\nSecret Knowledge:")
            for s in secrets:
                parts.append(f"- {s}")

        instructions[agent_id] = "\n".join(parts)

    return instructions


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
    all_agent_ids = ["agent_0", "agent_1"]  # Assuming 2 agents
    human_agents = [a for a in all_agent_ids if a not in llm_agents]

    cprint(f"Human-controlled: {human_agents}", "green")
    cprint(f"LLM-controlled: {llm_agents}", "green")

    # Load task or setup mechanics
    task_info = None
    task_data = {"mechanics": []}
    is_task_mode = False  # Track if we're running a specific task

    if extra_args and extra_args.task:
        task_info = load_task(extra_args.task)
        if task_info:
            is_task_mode = True  # Running a specific task

            # Print task information including agent secrets
            print_task_info(task_info)

            # Reset environment to the correct episode for this task
            task_episode_id = task_info.get("episode_id")
            if task_episode_id and task_episode_id != "unknown":
                cprint(f"Resetting environment to episode: {task_episode_id}", "blue")
                try:
                    env_interface.reset_environment(episode_id=task_episode_id)
                    cprint(f"Successfully loaded episode {task_episode_id}", "green")
                except (ValueError, IndexError) as e:
                    cprint(f"Warning: Could not load episode {task_episode_id}: {e}", "yellow")
                    cprint("Continuing with current episode...", "yellow")

            # Use mechanics from task (even if empty - we respect the task definition)
            if task_info.get("mechanic_bindings"):
                task_data["mechanics"] = task_info["mechanic_bindings"]
                cprint(f"  Mechanics: {len(task_data['mechanics'])} from task", "green")
                for binding in task_info["mechanic_bindings"]:
                    mech_type = binding.get("mechanic_type", "unknown")
                    trigger = binding.get("trigger_object", "?")
                    target = binding.get("target_object", "self")
                    print(f"    - {mech_type}: {trigger} -> {target}")
            else:
                cprint("  Mechanics: none (task has no mechanics)", "green")

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

    # Create and setup human test runner
    from emtom.runner import HumanTestRunner

    runner = HumanTestRunner(config)
    runner.setup(
        env_interface=env_interface,
        # In task mode, always pass task_data (even if empty) to prevent auto-bind in base.py
        # In exploration mode, pass task_data with all mechanics enabled
        task_data=task_data,
        output_dir=output_dir,
        task_info=task_info,
        human_agents=human_agents,
    )

    # Only auto-bind mechanics if NOT in task mode (exploration mode only)
    # When running a specific task, use ONLY the mechanics defined in the task
    if not is_task_mode:
        state, bindings = runner.game_manager.auto_bind_mechanics()
        if bindings:
            print_bindings_pretty(bindings)

    active = runner.game_manager.get_debug_info()['active_mechanics']
    cprint(f"⚙️  Active mechanics: {', '.join(active)}", "green")

    # Build instruction
    if task_info:
        instruction = task_to_instruction(task_info)
    else:
        instruction = {
            "agent_0": "Explore the environment and discover mechanics.",
            "agent_1": "Explore the environment and discover mechanics.",
        }

    # Run interactive test
    try:
        results = runner.run(instruction=instruction)
        cprint(f"\nTest completed: {results['steps']} steps", "green")
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
