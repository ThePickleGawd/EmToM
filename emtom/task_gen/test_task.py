#!/usr/bin/env python3
"""
Standalone script to test task with LLM agents.

This runs as a subprocess to get a fresh GL context.

Usage:
    python emtom/task_gen/test_task.py --task-file <path> --config-name <config>
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test task with LLM agents")
    parser.add_argument("--task-file", required=True, help="Path to task JSON")
    parser.add_argument("--result-file", required=True, help="Path to write result JSON")
    parser.add_argument("--config-name", default="examples/emtom_two_robots")
    parser.add_argument("--max-turns", type=int, default=20, help="Max LLM turns per agent")
    args = parser.parse_args()

    def write_result(result: dict):
        """Write result to file instead of stdout."""
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Load task
    try:
        with open(args.task_file) as f:
            task_data = json.load(f)
    except Exception as e:
        write_result({"success": False, "error": f"Failed to load task: {e}"})
        sys.exit(1)

    # Import and setup Habitat
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict

        from habitat_llm.agent.env import (
            register_actions,
            register_measures,
            register_sensors,
            remove_visual_sensors,
        )
        from habitat_llm.agent.env.dataset import CollaborationDatasetV0
        from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        from habitat_llm.utils import fix_config, setup_config

        from emtom.runner import BenchmarkRunner
        from emtom.runner.benchmark import task_to_instruction
        from emtom.task_gen import GeneratedTask
    except ImportError as e:
        write_result({"success": False, "error": f"Import error: {e}"})
        sys.exit(1)

    # Initialize Hydra config
    try:
        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=args.config_name)

        # Manually override Hydra interpolations BEFORE fix_config tries to resolve them
        output_dir = "/tmp/emtom_test"
        with open_dict(config):
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=47668090)
    except Exception as e:
        write_result({"success": False, "error": f"Config error: {e}"})
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        write_result({"success": False, "error": f"Invalid task format: {e}"})
        sys.exit(1)

    # Setup environment
    try:
        # Remove visual sensors - LLM planners don't need them
        remove_visual_sensors(config)

        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = CollaborationDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        # Load the specific episode from the task
        dataset_episode_id = task.dataset_episode_id
        print(f"Loading episode: {dataset_episode_id} (scene: {task.scene_id})", file=sys.stderr)
        env_interface.reset_environment(episode_id=dataset_episode_id)

        runner = BenchmarkRunner(config)

        task_mechanics = {
            "mechanics": [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        } if task.mechanic_bindings else None

        runner.setup(
            env_interface=env_interface,
            task_data=task_mechanics,
            output_dir=output_dir,
            task=task,
            save_video=False,
        )

        # Generate instruction and run
        instruction = task_to_instruction(task)
        results = runner.run(instruction=instruction, max_turns=args.max_turns)

        runner.cleanup()

        write_result({
            "success": True,
            "steps": results.get("steps", 0),
            "turns": results.get("turns", 0),
            "done": results.get("done", False),
            "episode_over": results.get("episode_over", False),
            "summary": f"Task {'completed' if results.get('done') else 'not completed'} in {results.get('turns', 0)} turns"
        })

    except Exception as e:
        write_result({
            "success": False,
            "steps": 0,
            "done": False,
            "error": str(e),
            "summary": f"Benchmark error: {e}"
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
