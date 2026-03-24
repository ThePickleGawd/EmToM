#!/usr/bin/env python3
"""
Standalone script to load a scene.

Runs as a subprocess to get a fresh GL context and avoid sensor registry conflicts.

Usage:
    python emtom/task_gen/load_scene.py --result-file <path> --config-name <config> [--seed N] [--scene-id ID]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Load a scene (random or specific)")
    parser.add_argument("--result-file", required=True, help="Path to write result JSON")
    parser.add_argument("--working-dir", default=None, help="Working directory for Hydra config output")
    parser.add_argument("--config-name", default="examples/emtom_2_robots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for scene selection")
    parser.add_argument("--scene-id", type=str, default=None, help="Specific scene ID to load (e.g., '102817140')")
    args = parser.parse_args()

    def write_result(result: dict):
        """Write result to file."""
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)

    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict

        from habitat_llm.utils import fix_config, setup_config

        from emtom.task_gen.scene_loader import load_scene
    except ImportError as e:
        # Minimal fallback for infra environments without Habitat/Hydra deps.
        # This allows taskgen to proceed with a tiny synthetic scene suitable
        # for unit testing/judging the task JSON format.
        write_result({
            "success": True,
            "scene_data": {
                "scene_id": "synthetic_scene",
                "episode_id": "synthetic_episode",
                "num_agents": 2,
                "agent_spawns": {"agent_0": "room_0", "agent_1": "room_1"},
                "rooms": ["room_0", "room_1"],
                "furniture_in_rooms": {"room_0": ["table_0", "cabinet_0"], "room_1": ["table_1", "cabinet_1"]},
                "objects": ["mug_0", "book_0", "key_0", "note_0", "bottle_0"],
                "objects_on_furniture": {"table_0": ["mug_0", "note_0"], "cabinet_0": ["key_0"], "table_1": ["book_0"], "cabinet_1": ["bottle_0"]},
                "valid_agent_ids": ["agent_0", "agent_1"],
            },
            "warning": f"Fallback synthetic scene used due to missing deps: {e}",
        })
        return
    # Initialize Hydra config
    try:
        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=args.config_name)

        # Set output directory to avoid Hydra interpolation errors
        # Use working_dir if provided, otherwise fall back to PID-based tmp directory
        if args.working_dir:
            output_dir = f"{args.working_dir}/hydra_scene_{os.getpid()}"
        else:
            output_dir = f"/tmp/emtom_scene_load_{os.getpid()}"
        with open_dict(config):
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=args.seed or 47668090)
    except Exception as e:
        write_result({"success": False, "error": f"Config error: {e}"})
        sys.exit(1)

    # Load scene (random or specific)
    try:
        scene_data = load_scene(config, seed=args.seed, scene_id=args.scene_id)

        write_result({
            "success": True,
            "scene_data": scene_data.to_dict(),
        })

    except Exception as e:
        import traceback
        write_result({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
