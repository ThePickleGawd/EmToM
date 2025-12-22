#!/usr/bin/env python3
# isort: skip_file
"""
Run EMTOM exploration using the Habitat simulator backend.

This script runs LLM-guided exploration in the Habitat environment,
discovering mechanics through curiosity-driven action selection.

Usage:
    ./emtom/run_emtom.sh explore --steps 50
"""

import sys
import time
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from omegaconf import DictConfig

from habitat_llm.utils import setup_config, fix_config
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


@hydra.main(config_path="../../habitat_llm/conf", version_base=None)
def main(config: DictConfig):
    """Main entry point with Hydra configuration."""
    print("=" * 60)
    print("EMTOM Habitat Exploration")
    print("=" * 60)

    # Fix and setup config
    fix_config(config)
    seed = 47668090
    config = setup_config(config, seed)

    # Get exploration parameters from config or defaults
    max_steps = config.get("exploration_steps", 50)

    # Register Habitat components
    print("Registering Habitat components...")
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = None
    if hasattr(config.habitat, 'dataset'):
        try:
            dataset = CollaborationDatasetV0(config.habitat.dataset)
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")

    # Create environment interface
    print("Creating environment...")
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Always select a random episode for diversity
    if dataset and hasattr(dataset, 'episodes') and len(dataset.episodes) > 1:
        import random as rnd
        rnd.seed(seed)
        selected_episode = rnd.choice(dataset.episodes)
        print(f"Randomly selected episode: {selected_episode.episode_id}")
        env_interface.reset_environment(episode_id=selected_episode.episode_id)

    # Get scene info
    current_episode = env_interface.env.env.env._env.current_episode
    scene_id = getattr(current_episode, 'scene_id', "unknown")
    episode_id = getattr(current_episode, 'episode_id', "unknown")

    print(f"Scene: {scene_id}")
    print(f"Episode: {episode_id}")

    # Get output directory
    output_dir = config.paths.results_dir if hasattr(config, 'paths') else "data/emtom/trajectories"

    # Create and setup exploration runner
    from emtom.runner import ExplorationRunner

    runner = ExplorationRunner(config)
    runner.setup(
        env_interface=env_interface,
        task_data=None,  # Use all mechanics and auto-bind
        output_dir=output_dir,
        max_steps=max_steps,
        save_video=getattr(config.evaluation, 'save_video', True),
        save_fpv=True,
    )

    # Run exploration
    t0 = time.time()
    print(f"\nRunning exploration for {max_steps} steps...")
    print("-" * 40)

    metadata = {
        "seed": seed,
        "mode": "llm",
        "scene_id": scene_id,
        "episode_id": episode_id,
    }

    try:
        episode_data = runner.run(max_steps=max_steps, metadata=metadata)
    except Exception as e:
        print(f"Exploration failed: {e}")
        import traceback
        traceback.print_exc()
        runner.cleanup()
        return

    elapsed = time.time() - t0

    # Print results
    print("\n" + "=" * 60)
    print("EXPLORATION RESULTS")
    print("=" * 60)

    stats = episode_data.get("statistics", {})
    print(f"Total steps: {stats.get('total_steps', 'N/A')}")
    print(f"Total surprises: {stats.get('total_surprises', 0)}")
    print(f"Actions per agent: {stats.get('actions_per_agent', {})}")
    print(f"Unique actions: {stats.get('unique_actions', 0)}")
    print(f"Time: {elapsed:.1f}s")

    # Print surprises
    if episode_data.get("surprise_summary"):
        print("\nSurprise moments:")
        for s in episode_data["surprise_summary"]:
            print(f"  Step {s['step']}: {s['action']} on {s['target']}")
            print(f"    Level: {s['surprise_level']}/5")
            print(f"    {s['explanation']}")

    # Print video paths
    if episode_data.get("video_paths"):
        print("\nSaved videos:")
        for name, path in episode_data["video_paths"].items():
            print(f"  {name}: {path}")

    # Print trajectory path
    print(f"\nTrajectory: {output_dir}/trajectory_{episode_data['episode_id']}.json")
    print(f"Copied to: data/emtom/trajectories/trajectory_{episode_data['episode_id']}.json")

    runner.cleanup()
    print(f"\nExploration completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
