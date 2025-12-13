#!/usr/bin/env python3
"""
Script to make the robot agent explore all rooms in a scene.

This script systematically navigates the robot through every room,
visiting furniture within each room to thoroughly explore the environment.

Usage:
    # With real-time X11 display (no video saving):
    python -m main.exploration.explore_all_rooms \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        +live_display=True

    # With video saving (legacy mode):
    python -m main.exploration.explore_all_rooms \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        evaluation.save_video=True
"""

import sys
import subprocess
from typing import Any, List

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
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer
from habitat_llm.utils.sim import init_agents
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)

from main.exploration.room_explorer import RoomExplorer, FullExplorationResult


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
def run_room_exploration(config: omegaconf.DictConfig) -> FullExplorationResult:
    """
    Main function that loads a scene and explores all rooms.

    Args:
        config: Hydra config (uses same config as skill_runner).

    Returns:
        FullExplorationResult with complete exploration data.
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

    # Video configuration - disabled when using live display
    if live_display:
        # In live display mode, we don't save videos
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

    # Show topdown map if requested (only in non-live-display mode)
    if config.get("skill_runner_show_topdown", False) and not live_display:
        dbv = DebugVisualizer(sim, config.paths.results_dir)
        dbv.create_dbv_agent(resolution=(1000, 1000))
        top_down_map = dbv.peek("stage")
        if show_videos:
            top_down_map.show()
        if config.evaluation.save_video:
            top_down_map.save(output_path=config.paths.results_dir, prefix="topdown")
        dbv.remove_dbv_agent()
        dbv.create_dbv_agent()
        dbv.remove_dbv_agent()

    # Print scene information
    cprint("=== Room Exploration Script Started ===", "green")
    cprint(
        f"Episode ID: {sim.ep_info.episode_id}, Scene: {sim.ep_info.scene_id}",
        "green",
    )

    # Print all entities in the scene
    print("\n=== Entities in Scene ===")
    print_all_entities(env_interface.perception.gt_graph)
    print_furniture_entity_handles(env_interface.perception.gt_graph)
    print_object_entity_handles(env_interface.perception.gt_graph)

    # Get the robot agent (agent 0)
    robot_agent = planner.get_agent_from_uid(env_interface.robot_agent_uid)
    cprint(f"\nUsing Robot Agent (uid={env_interface.robot_agent_uid})", "blue")

    # Create the RoomExplorer with live display if enabled
    explorer = RoomExplorer(
        env_interface=env_interface,
        planner=planner,
        robot_agent=robot_agent,
        show_videos=show_videos,
        live_display=live_display,
        display_scale=display_scale,
    )

    # Get exploration configuration
    furniture_per_room = config.get("furniture_per_room", None)
    randomize_furniture = config.get("randomize_furniture", True)
    randomize_rooms = config.get("randomize_rooms", False)

    # Print room information
    rooms = explorer.get_room_names()
    cprint(f"\nFound {len(rooms)} rooms in the scene:", "blue")
    for room in rooms:
        furniture_count = len(explorer.get_furniture_in_room(room))
        cprint(f"  - {room}: {furniture_count} furniture items", "yellow")

    # Define optional callbacks
    def on_room_enter(room_name: str, room_idx: int, total_rooms: int) -> None:
        cprint(f"\n>>> Entering room: {room_name} ({room_idx + 1}/{total_rooms})", "yellow")

    def on_room_exit(room_name: str, result) -> None:
        cprint(
            f"<<< Finished exploring {room_name}: visited {len(result.furniture_visited)} furniture",
            "yellow",
        )

    # Run exploration
    cprint(f"\n{'=' * 60}", "green")
    cprint("Starting room exploration...", "green")
    if live_display:
        cprint("  Mode: LIVE X11 DISPLAY (press 'q' in window to quit)", "blue")
    else:
        cprint("  Mode: Video recording", "blue")
    if furniture_per_room:
        cprint(f"  Furniture per room limit: {furniture_per_room}", "blue")
    cprint(f"  Randomize furniture order: {randomize_furniture}", "blue")
    cprint(f"  Randomize room order: {randomize_rooms}", "blue")
    cprint(f"{'=' * 60}\n", "green")

    results = explorer.explore_all_rooms(
        furniture_per_room=furniture_per_room,
        randomize_furniture=randomize_furniture,
        randomize_rooms=randomize_rooms,
        make_video=make_video,
        on_room_enter=on_room_enter,
        on_room_exit=on_room_exit,
    )

    # Create cumulative video of all exploration (only if not using live display)
    if not live_display and len(results.all_frames) > 0 and make_video:
        cprint("\n" + "=" * 60, "green")
        cprint("Creating COMBINED video of all exploration...", "green")
        cprint("=" * 60, "green")

        dvu = DebugVideoUtil(env_interface, config.paths.results_dir)
        dvu.frames = results.all_frames
        dvu._make_video(postfix="all_room_exploration", play=show_videos)

        cumulative_video_path = f"{config.paths.results_dir}/videos/video-all_room_exploration.mp4"
        cprint(f"\nCombined video saved to: {cumulative_video_path}", "green")
        cprint(f"  Total frames: {len(results.all_frames)}", "blue")

    # Print summary
    cprint("\n=== Room Exploration Summary ===", "green")
    cprint(f"Total rooms explored: {len(results.rooms_explored)}", "blue")
    cprint(f"Total furniture visited: {results.total_furniture_visited}", "blue")

    for room_result in results.rooms_explored:
        successful = sum(1 for _, _, s in room_result.navigation_results if s)
        failed = len(room_result.navigation_results) - successful
        cprint(
            f"  {room_result.room_name}: {successful} successful, {failed} failed navigations",
            "yellow" if failed > 0 else "green",
        )

    if not live_display:
        cprint(f"\nVideos saved in: {config.paths.results_dir}/videos/", "blue")
    cprint("=== Room Exploration Script Completed ===", "green")

    return results


if __name__ == "__main__":
    cprint(
        "\nStarting room exploration script for robot agent in Habitat environment.",
        "blue",
    )

    run_room_exploration()

    cprint(
        "\nRoom exploration script finished.",
        "blue",
    )
