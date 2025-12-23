#!/usr/bin/env python3
"""
Entry point for the agentic task generator.

Usage:
    python emtom/task_gen/runner.py --config-name examples/emtom_2_robots +num_tasks=5 +model=gpt-5

Or via shell script:
    ./emtom/run_emtom.sh generate --num-tasks 5 --model gpt-5
"""

from __future__ import annotations

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


@hydra.main(version_base=None, config_path="../../habitat_llm/conf")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Get Hydra output directory for logging
    from hydra.core.hydra_config import HydraConfig
    try:
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_output_dir = None

    # Extract custom args from config (passed as +arg=value)
    num_tasks = config.get("num_tasks", 1)
    model = config.get("model", "gpt-5")
    output_dir = config.get("output_dir", "data/emtom/tasks")  # Final output location
    max_iterations = config.get("max_iterations", 100)
    quiet = config.get("quiet", False)
    subtasks = config.get("subtasks", 3)
    seed = config.get("seed", None)

    # Create unique temp working directory for this instance (allows parallel runs)
    instance_id = uuid.uuid4().hex[:8]
    working_dir = Path(tempfile.gettempdir()) / f"emtom_taskgen_{instance_id}"
    working_dir.mkdir(parents=True, exist_ok=True)

    # Setup config (registers Habitat plugins, sets seed, etc.)
    fix_config(config)
    config = setup_config(config, seed=seed or 47668090)

    cprint("=" * 60, "blue")
    cprint("EMTOM Task Generator (Live Scene Mode)", "blue")
    cprint("=" * 60, "blue")
    cprint(f"Instance: {instance_id}", "blue")
    cprint(f"Target tasks: {num_tasks}", "blue")
    cprint(f"Model: {model}", "blue")
    cprint(f"Working dir: {working_dir}", "blue")
    cprint(f"Output: {output_dir}", "blue")
    cprint("=" * 60, "blue")
    print()

    # Load random scene from PARTNR dataset
    cprint("Loading random scene from PARTNR dataset...", "blue")
    from emtom.task_gen.scene_loader import load_random_scene

    try:
        scene_data = load_random_scene(config, seed=seed)
        cprint(f"Loaded scene {scene_data.scene_id} (episode {scene_data.episode_id})", "green")
        cprint(f"  Rooms: {len(scene_data.rooms)}", "green")
        cprint(f"  Furniture: {len(scene_data.furniture)}", "green")
        cprint(f"  Objects: {len(scene_data.objects)}", "green")
        cprint(f"  Articulated: {len(scene_data.articulated_furniture)}", "green")
    except Exception as e:
        cprint(f"Error loading scene: {e}", "red")
        import traceback
        traceback.print_exc()
        # Clean up temp dir on error
        shutil.rmtree(working_dir, ignore_errors=True)
        sys.exit(1)

    print()

    # Save scene data to working directory
    scene_file = working_dir / "current_scene.json"
    with open(scene_file, "w") as f:
        json.dump(scene_data.to_dict(), f, indent=2)
    cprint(f"Scene data saved to {scene_file}", "green")
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

    # Create and run agent with scene data
    from emtom.task_gen.agent import TaskGeneratorAgent

    agent = TaskGeneratorAgent(
        llm_client=llm_client,
        config=config,
        working_dir=str(working_dir),  # Unique temp dir for this instance
        output_dir=output_dir,  # Shared output for all instances
        max_iterations=max_iterations,
        verbose=not quiet,
        subtasks=subtasks,
        scene_data=scene_data,  # Pass live scene data
        log_dir=hydra_output_dir,  # Pass Hydra output directory for logs
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
    main()
