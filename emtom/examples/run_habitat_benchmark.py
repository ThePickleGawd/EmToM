#!/usr/bin/env python3
"""
Example script for running the EMTOM benchmark with Habitat integration.

This script demonstrates running EMTOM tasks in the Habitat simulator
with proper video recording (third-person and first-person views) and
LLM-driven agent actions.

Usage:
    # Run with Habitat (requires proper config and scene data)
    python emtom/examples/run_habitat_benchmark.py \
        --config-name examples/planner_multi_agent_demo_config \
        evaluation.save_video=True

    # The script uses the existing habitat_llm configuration system
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.evaluation import DecentralizedEvaluationRunner
from habitat_llm.utils import cprint, setup_config, fix_config

from emtom.task_gen import GeneratedTask
from emtom.actions import get_emtom_tools
from emtom import GameStateManager, list_mechanics
from emtom.exploration.habitat_explorer import HabitatWorldAdapter

# Import CommunicationTool (used as "Communicate" action)
from habitat_llm.tools.perception.communication_tool import CommunicationTool


def load_tasks(task_file: str) -> list:
    """Load tasks from JSON file."""
    with open(task_file) as f:
        data = json.load(f)

    tasks = []
    for task_data in data.get("tasks", []):
        task = GeneratedTask.from_dict(task_data)
        tasks.append(task)

    return tasks


def task_to_instruction(task: GeneratedTask) -> Dict[str, str]:
    """Convert an EMTOM task to per-agent instructions.

    Returns a dict mapping agent_id -> instruction string.
    Each agent gets the public goal plus their own secrets (if any).
    """
    instructions = {}

    for agent_id in task.agent_roles.keys():
        parts = [f"Goal: {task.public_goal}"]

        if task.public_context:
            parts.append(task.public_context)

        # Per-agent actions available
        actions = task.agent_actions.get(agent_id, [])
        if actions:
            parts.append(f"\nYour available actions: {', '.join(actions)}")

        # Per-agent secrets (only for ToM tasks)
        secrets = task.agent_secrets.get(agent_id, [])
        if secrets:
            parts.append("\nSecret Knowledge:")
            for s in secrets:
                parts.append(f"- {s}")

        if task.theory_of_mind_required:
            parts.append("\nUse Communicate[message] to coordinate with your teammate.")

        instructions[agent_id] = "\n".join(parts)

    return instructions


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Ensure video saving is enabled
    with open_dict(config):
        config.evaluation.save_video = True

    cprint("\n" + "="*60, "blue")
    cprint("EMTOM Habitat Benchmark", "blue")
    cprint("="*60, "blue")

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

    # Create evaluation runner - this handles planners, agents, video recording
    cprint("Initializing evaluation runner with LLM planners...", "blue")
    eval_runner = DecentralizedEvaluationRunner(config.evaluation, env_interface)
    cprint(f"Evaluation runner created: {eval_runner}", "green")

    # Setup output directory
    output_dir = config.paths.results_dir
    os.makedirs(output_dir, exist_ok=True)
    cprint(f"Output directory: {output_dir}", "blue")

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

    # Setup GameStateManager with mechanics from task
    cprint("\nSetting up GameStateManager with mechanics...", "blue")
    game_manager = GameStateManager(env_interface)

    # Convert task to mechanics initialization format
    if task.mechanic_bindings:
        task_data = {
            "mechanics": [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        }
        game_manager.initialize_from_task(task_data)
        cprint(f"Mechanics loaded from task bindings:", "green")
        for b in task.mechanic_bindings:
            cprint(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}", "green")
    else:
        # Use all mechanics and auto-bind
        all_mechanics = list_mechanics()
        task_data = {"mechanics": [{"mechanic_type": m} for m in all_mechanics]}
        game_manager.initialize_from_task(task_data)
        cprint(f"No mechanic bindings in task - using auto-bind with all mechanics", "yellow")

    # Get entities and auto-bind if needed
    world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)
    entities = world_adapter.get_interactable_entities()
    state = game_manager.get_state()
    state.entities = entities
    game_manager.set_state(state)

    if not task.mechanic_bindings:
        state, bindings = game_manager.auto_bind_mechanics()
        if bindings:
            cprint(f"Auto-bound mechanics: {bindings}", "green")

    # Inject EMTOM tools and Communicate into each agent
    cprint("\nInjecting tools into agents...", "blue")
    for agent in eval_runner.agents.values():
        agent_uid = agent.uid

        # Add EMTOM tools (Use, Inspect)
        emtom_tools = get_emtom_tools(agent_uid=agent_uid)
        for tool_name, tool in emtom_tools.items():
            tool.set_environment(env_interface)
            tool.set_game_manager(game_manager)
            agent.tools[tool_name] = tool
            cprint(f"  Added {tool_name} to agent_{agent_uid}", "green")

        # Add Communicate tool for agent-to-agent messaging
        comm_config = OmegaConf.create({
            "name": "Communicate",
            "description": "Send a message to the other agent. Usage: Communicate[your message]. The other agent will see your message in their context. IMPORTANT: Keep your message on a single line - do not use newlines."
        })
        comm_tool = CommunicationTool(comm_config)
        comm_tool.agent_uid = agent_uid
        comm_tool.set_environment(env_interface)
        agent.tools["Communicate"] = comm_tool
        cprint(f"  Added Communicate to agent_{agent_uid}", "green")

    # Print agent info
    cprint(f"\nAgents: {eval_runner.agent_list}", "blue")
    for agent in eval_runner.agents.values():
        cprint(f"  agent_{agent.uid} tools: {list(agent.tools.keys())}", "blue")

    instruction = task_to_instruction(task)

    cprint(f"\n{'='*60}", "blue")
    cprint(f"EMTOM TASK: {task.title}", "blue")
    cprint(f"{'='*60}", "blue")
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

    # Run the instruction using the evaluation runner
    try:
        cprint("\nStarting task execution with LLM planners...", "blue")
        info = eval_runner.run_instruction(
            instruction=instruction,
            output_name=f"emtom_{task.task_id}"
        )

        cprint("\nTask execution completed!", "green")
        print(f"Results: {json.dumps(info, indent=2, default=str)}")

    except Exception as e:
        error_str = str(e)

        # Check if this is a timeout (episode over) vs a real error
        is_timeout = "Episode over" in error_str or "call reset before calling step" in error_str

        if is_timeout:
            cprint(f"\nTask timed out (max simulation steps reached)", "yellow")
            cprint("This is normal - the task just needs more steps to complete.", "yellow")
            video_suffix = f"emtom_{task.task_id}_timeout"
        else:
            cprint(f"Error during task execution: {e}", "red")
            traceback.print_exc()
            video_suffix = f"emtom_{task.task_id}_error"

        # Save video
        cprint("Saving video...", "yellow")
        try:
            if hasattr(eval_runner, 'dvu') and eval_runner.dvu is not None:
                eval_runner.dvu._make_video(play=False, postfix=video_suffix)
                cprint("Third-person video saved!", "green")
            if hasattr(eval_runner, '_fpv_recorder') and eval_runner._fpv_recorder is not None:
                eval_runner._make_first_person_videos()
                cprint("First-person videos saved!", "green")
        except Exception as ve:
            cprint(f"Could not save video: {ve}", "red")

    # Cleanup
    env_interface.env.close()
    cprint("\nBenchmark complete!", "green")
    cprint(f"Check {output_dir} for videos and logs", "blue")


if __name__ == "__main__":
    cprint("\nEMTOM Habitat Benchmark Runner", "blue")
    cprint("This script runs EMTOM tasks in Habitat with LLM planners and video recording.\n", "blue")

    if len(sys.argv) < 2:
        cprint("Usage: python run_habitat_benchmark.py --config-name <config>", "yellow")
        cprint("Example: python run_habitat_benchmark.py --config-name examples/planner_multi_agent_demo_config", "yellow")
        sys.exit(1)

    main()
