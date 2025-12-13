#!/usr/bin/env python3
"""
GPT-Guided Room Exploration Script.

This script uses OpenAI's GPT to guide a robot through exploring rooms in a scene.
GPT decides which rooms to visit (no repeats), what objects to interact with,
forms hypotheses, and reports surprising findings.

Results are saved to JSON and displayed in the terminal.

Requirements:
    - OPENAI_API_KEY must be set in your .env file

Usage:
    # With real-time X11 display:
    python -m main.exploration.explore_with_gpt \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        +live_display=True

    # Limit rooms and interactions:
    python -m main.exploration.explore_with_gpt \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        +live_display=True \
        +max_rooms=3 \
        +max_interactions_per_room=2

    # Use a different GPT model:
    python -m main.exploration.explore_with_gpt \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        +gpt_model="gpt-4o"
"""

import sys
import subprocess
from typing import Any

import omegaconf
import hydra
from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)

from main.exploration.room_explorer import RoomExplorer
from main.exploration.gpt_guided_explorer import GPTGuidedExplorer


def play_video_x11(filename: str) -> None:
    """Play video using mpv which works better with X11 forwarding."""
    print(f"     ...playing video with mpv (press 'q' to skip)...")
    try:
        subprocess.run(
            [
                "mpv",
                "--keep-open=no",
                "--really-quiet",
                filename,
            ],
            check=False,
        )
    except Exception as e:
        print(f"Could not play video with mpv: {e}")
        print(f"Video saved to: {filename}")


# Monkey-patch DebugVideoUtil to use X11-friendly video player
DebugVideoUtil.play_video = lambda self, filename: play_video_x11(filename)


@hydra.main(
    config_path="../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
)
def run_gpt_guided_exploration(config: omegaconf.DictConfig) -> Any:
    """
    Main function that loads a scene and runs GPT-guided exploration.

    Args:
        config: Hydra config (uses same config as skill_runner).

    Returns:
        ExplorationSession with complete exploration data.
    """
    fix_config(config)

    # Setup seed for reproducibility
    seed = 47668090

    # Setup hardcoded config overrides
    with omegaconf.open_dict(config):
        config_dict = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(config.habitat, resolve=True)
        )
        config_dict.dataset.metadata = {"metadata_folder": "data/hssd-hab/metadata"}
        config.habitat = config_dict

    config = setup_config(config, seed)

    assert config.env == "habitat", "Only valid for Habitat."

    # Check if live display mode is enabled
    live_display = config.get("live_display", False)
    display_scale = config.get("display_scale", 1.0)

    # GPT configuration
    gpt_model = config.get("gpt_model", "gpt-4o-mini")
    max_rooms = config.get("max_rooms", None)
    max_interactions_per_room = config.get("max_interactions_per_room", 3)
    output_dir = config.get("output_dir", "./exploration_results")

    # Video configuration - disabled when using live display
    if live_display:
        show_videos = False
        make_video = False
        cprint("Live X11 display mode enabled - videos will NOT be saved", "blue")
    else:
        show_videos = config.get("skill_runner_show_videos", True)
        make_video = config.evaluation.save_video or show_videos

    if not make_video and not live_display:
        remove_visual_sensors(config)

    # Register sensors, actions, and measures
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    print(f"Loading EpisodeDataset from: {config.habitat.dataset.data_path}")

    # Initialize environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Select episode (by index or id)
    if hasattr(config, "skill_runner_episode_index"):
        episode_index = config.skill_runner_episode_index
        print(f"Loading episode_index = {episode_index}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_index(
            episode_index
        )
    elif hasattr(config, "skill_runner_episode_id"):
        episode_id = config.skill_runner_episode_id
        print(f"Loading episode_id = {episode_id}")
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )

    env_interface.reset_environment()

    # Initialize planner
    planner_conf = config.evaluation.planner
    planner = instantiate(planner_conf)
    planner = planner(env_interface=env_interface)
    agent_config = config.evaluation.agents
    planner.agents = init_agents(agent_config, env_interface)
    planner.reset()

    sim = env_interface.sim

    # Print scene information
    cprint("=== GPT-Guided Room Exploration ===", "green")
    cprint(
        f"Episode ID: {sim.ep_info.episode_id}, Scene: {sim.ep_info.scene_id}",
        "green",
    )
    cprint(f"GPT Model: {gpt_model}", "blue")

    # Print all entities in the scene
    print("\n=== Entities in Scene ===")
    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    # Get the robot agent (agent 0)
    robot_agent = planner.get_agent_from_uid(env_interface.robot_agent_uid)
    cprint(f"\nUsing Robot Agent (uid={env_interface.robot_agent_uid})", "blue")

    # Create the RoomExplorer with live display if enabled
    room_explorer = RoomExplorer(
        env_interface=env_interface,
        planner=planner,
        robot_agent=robot_agent,
        show_videos=show_videos,
        live_display=live_display,
        display_scale=display_scale,
    )

    # Print room information
    rooms = room_explorer.get_room_names()
    cprint(f"\nFound {len(rooms)} rooms in the scene:", "blue")
    for room in rooms:
        furniture_count = len(room_explorer.get_furniture_in_room(room))
        cprint(f"  - {room}: {furniture_count} furniture items", "yellow")

    # Create GPT-guided explorer
    gpt_explorer = GPTGuidedExplorer(
        room_explorer=room_explorer,
        gpt_model=gpt_model,
        output_dir=output_dir
    )

    # Run GPT-guided exploration
    cprint(f"\n{'=' * 60}", "green")
    cprint("Starting GPT-guided exploration...", "green")
    if live_display:
        cprint("  Mode: LIVE X11 DISPLAY (press 'q' in window to quit)", "blue")
    if max_rooms:
        cprint(f"  Max rooms: {max_rooms}", "blue")
    cprint(f"  Max interactions per room: {max_interactions_per_room}", "blue")
    cprint(f"{'=' * 60}\n", "green")

    session = gpt_explorer.explore_with_gpt(
        max_rooms=max_rooms,
        max_interactions_per_room=max_interactions_per_room
    )

    cprint("=== GPT-Guided Exploration Complete ===", "green")

    return session


if __name__ == "__main__":
    cprint(
        "\nStarting GPT-guided room exploration for robot agent in Habitat environment.",
        "blue",
    )

    run_gpt_guided_exploration()

    cprint(
        "\nGPT-guided exploration script finished.",
        "blue",
    )
