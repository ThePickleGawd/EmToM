#!/usr/bin/env python3
"""
Entry point for the agentic task generator.

Usage:
    python emtom/task_gen/runner.py --config-name examples/emtom_two_robots +num_tasks=5 +model=gpt-5

Or via shell script:
    ./emtom/run_emtom.sh generate --num-tasks 5 --model gpt-5
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in Python path (for imports like emtom.*)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import hydra
from omegaconf import DictConfig, open_dict

from habitat_llm.utils import cprint, setup_config, fix_config


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Extract custom args from config (passed as +arg=value)
    num_tasks = config.get("num_tasks", 1)
    model = config.get("model", "gpt-5")
    trajectory_dir = config.get("trajectory_dir", "data/emtom/trajectories")
    output_dir = config.get("output_dir", "data/emtom/tasks/curated")
    max_iterations = config.get("max_iterations", 100)
    quiet = config.get("quiet", False)
    difficulty = config.get("difficulty", 3)
    min_subtasks = config.get("min_subtasks", 3)
    max_subtasks = config.get("max_subtasks", 5)

    # Setup config (registers Habitat plugins, sets seed, etc.)
    fix_config(config)
    config = setup_config(config, seed=47668090)

    cprint("=" * 60, "blue")
    cprint("EMTOM Agentic Task Generator", "blue")
    cprint("=" * 60, "blue")
    cprint(f"Target tasks: {num_tasks}", "blue")
    cprint(f"Model: {model}", "blue")
    cprint(f"Trajectories: {trajectory_dir}", "blue")
    cprint(f"Output: {output_dir}", "blue")
    cprint("=" * 60, "blue")
    print()

    # Check trajectory directory
    if not Path(trajectory_dir).exists():
        cprint(f"Error: Trajectory directory not found: {trajectory_dir}", "red")
        cprint("Run exploration first: ./emtom/run_emtom.sh explore", "yellow")
        sys.exit(1)

    # Check for trajectories
    trajectories = list(Path(trajectory_dir).glob("*.json"))
    if not trajectories:
        cprint(f"Error: No trajectory files found in {trajectory_dir}", "red")
        cprint("Run exploration first: ./emtom/run_emtom.sh explore", "yellow")
        sys.exit(1)

    cprint(f"Found {len(trajectories)} trajectory files", "green")
    print()

    # Initialize LLM
    cprint("Initializing LLM client...", "blue")
    try:
        from habitat_llm.llm import instantiate_llm
        llm_client = instantiate_llm("openai_chat", model=model)
        cprint(f"Using model: {llm_client.generation_params.model}", "green")
    except Exception as e:
        cprint(f"Error initializing LLM: {e}", "red")
        sys.exit(1)

    # Create and run agent
    from emtom.task_gen.agent import TaskGeneratorAgent

    agent = TaskGeneratorAgent(
        llm_client=llm_client,
        config=config,
        working_dir="data/emtom/tasks",
        trajectory_dir=trajectory_dir,
        output_dir=output_dir,
        max_iterations=max_iterations,
        verbose=not quiet,
        difficulty=difficulty,
        min_subtasks=min_subtasks,
        max_subtasks=max_subtasks,
    )

    # Run agent
    submitted_tasks = agent.run(num_tasks_target=num_tasks)

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
    main()
