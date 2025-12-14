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
from emtom.tools import get_emtom_tools
from emtom.mechanics import MechanicRegistry, wrap_tools_with_mechanics

# Import CommunicationTool for agent-to-agent messaging
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


def task_to_instruction(task: GeneratedTask) -> str:
    """Convert an EMTOM task to a natural language instruction for the agents.

    Note: We don't include global mechanic hints here since both agents see this instruction.
    Agent-specific knowledge (including mechanic knowledge for agent_0) is in the task's
    agent_knowledge section.
    """
    # Build an instruction that includes the task context
    instruction_parts = [
        f"Task: {task.title}",
        f"Description: {task.description}",
        f"Goal: {task.success_condition.description}",
    ]

    # Add any special knowledge that agents have
    # Note: This is where agent_0 gets mechanic knowledge and agent_1 doesn't
    if task.agent_knowledge:
        instruction_parts.append("\nAgent-specific knowledge (only visible to that agent):")
        for agent_id, knowledge in task.agent_knowledge.items():
            if knowledge:
                instruction_parts.append(f"  {agent_id}: {', '.join(knowledge)}")

    return "\n".join(instruction_parts)


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

    # Load EMTOM tasks first (needed for mechanics setup)
    task_dir = Path("data/emtom/tasks")
    if not task_dir.exists():
        cprint(f"ERROR: Task directory not found: {task_dir}", "red")
        cprint("Run task generation first: ./emtom/run_emtom.sh generate", "yellow")
        sys.exit(1)

    # Prefer test files with mechanic bindings over generated challenge files
    test_file = task_dir / "emtom_tom_test.json"
    if test_file.exists():
        task_file = test_file
        cprint(f"Using test task file with mechanic bindings: {task_file}", "green")
    else:
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

    # Setup mechanics from task bindings using unified mechanics system
    mechanics = []
    if task.mechanic_bindings:
        # Convert MechanicBinding objects to dicts for the registry
        bindings_dicts = [b.to_dict() for b in task.mechanic_bindings]
        mechanics = MechanicRegistry.instantiate_from_bindings(bindings_dicts)
        cprint(f"\nMechanics loaded from task bindings:", "green")
        for b in task.mechanic_bindings:
            cprint(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}", "green")
    else:
        cprint(f"\nNo mechanic bindings in task - mechanics will not be active", "yellow")

    # Inject EMTOM tools and CommunicationTool into each agent
    cprint("\nInjecting tools into agents...", "blue")
    for agent in eval_runner.agents.values():
        agent_uid = agent.uid

        # Add EMTOM tools (Hide, Inspect, WriteMessage)
        emtom_tools = get_emtom_tools(agent_uid=agent_uid)
        for tool_name, tool in emtom_tools.items():
            tool.set_environment(env_interface)
            agent.tools[tool_name] = tool
            cprint(f"  Added {tool_name} to agent_{agent_uid}", "green")

        # Add CommunicationTool renamed to "Communicate"
        comm_config = OmegaConf.create({
            "name": "Communicate",
            "description": "Send a message to the other agent. Usage: Communicate[your message here]. The other agent will see your message in their context."
        })
        comm_tool = CommunicationTool(comm_config)
        comm_tool.agent_uid = agent_uid
        comm_tool.set_environment(env_interface)
        agent.tools["Communicate"] = comm_tool
        cprint(f"  Added Communicate to agent_{agent_uid}", "green")

        # Wrap tools with unified mechanics system if mechanics are active
        if mechanics:
            agent.tools = wrap_tools_with_mechanics(
                agent_tools=agent.tools,
                mechanics=mechanics,
                world_state_adapter=None,  # TODO: Add world state adapter if needed
            )

    # Print agent info
    cprint(f"\nAgents: {eval_runner.agent_list}", "blue")
    for agent in eval_runner.agents.values():
        cprint(f"  agent_{agent.uid} tools: {list(agent.tools.keys())}", "blue")

    instruction = task_to_instruction(task)

    cprint(f"\n{'='*60}", "blue")
    cprint(f"EMTOM TASK: {task.title}", "blue")
    cprint(f"{'='*60}", "blue")
    print(f"Description: {task.description}")
    print(f"Mechanics: {task.required_mechanics}")
    if task.mechanic_bindings:
        print(f"Mechanic bindings: {len(task.mechanic_bindings)} active")
        for b in task.mechanic_bindings:
            print(f"  - {b.mechanic_type}: {b.trigger_object} -> {b.target_object or 'self'}")
    print(f"Goal: {task.success_condition.description}")
    print(f"\nInstruction to agents:\n{instruction}")

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
