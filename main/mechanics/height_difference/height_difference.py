#!/usr/bin/env python3
# Script to make the robot agent do random walks around the scene
# Supports both random walk mode and GPT-guided exploration mode
# When a surprising finding is detected, generates tasks using LLM and ends the scene

import sys
import random
import subprocess
import os
from typing import List, Any

# Append parent directory path
sys.path.append("../../..")

# Import GPT-guided exploration modules
from main.exploration.room_explorer import RoomExplorer
from main.exploration.gpt_guided_explorer import GPTGuidedExplorer, ExplorationSession, RoomVisit, Interaction

# Import task generator
from main.llm_task_generation.task_generator import (
    TaskGenerator,
    ensure_tasks_directory,
)

import omegaconf
import hydra
from hydra.utils import instantiate

from habitat_llm.utils import cprint, setup_config, fix_config

# Patch the video player to use mpv/ffplay instead of broken OpenCV Qt
def play_video_x11(filename: str) -> None:
    """Play video using mpv which works better with X11 forwarding"""
    print(f"     ...playing video with mpv (press 'q' to skip)...")
    try:
        # Play video once with mpv (works great with X11 forwarding)
        subprocess.run([
            "mpv",
            "--keep-open=no",  # Close after playing
            "--really-quiet",  # Less verbose output
            filename
        ], check=False)
    except Exception as e:
        print(f"Could not play video with mpv: {e}")
        print(f"Video saved to: {filename}")

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
from habitat_llm.examples.example_utils import execute_skill, DebugVideoUtil
from habitat_llm.utils.world_graph import (
    print_all_entities,
    print_furniture_entity_handles,
    print_object_entity_handles,
)

# Monkey-patch DebugVideoUtil to use our X11-friendly video player
DebugVideoUtil.play_video = lambda self, filename: play_video_x11(filename)


@hydra.main(
    config_path="../../../habitat_llm/conf",
    config_name="examples/skill_runner_default_config.yaml",
)
def run_exploration(config: omegaconf.DictConfig) -> None:
    """
    Main function that loads a scene and explores it using the configured mode.

    Supports two exploration modes:
    - "random" (default): Random walks to furniture items
    - "gpt": GPT-guided exploration with hypothesis generation and interaction

    Config options:
        exploration_mode: "random" or "gpt" (default: "random")
        live_display: Enable X11 live display (default: False)
        gpt_model: GPT model for exploration (default: "gpt-4o-mini")
        max_rooms: Max rooms to explore in GPT mode (default: all)
        max_interactions_per_room: Max interactions per room (default: 3)
        num_random_walks: Number of walks in random mode (default: 5)

    Example usage:
        # GPT-guided exploration:
        python height_difference.py +exploration_mode=gpt +live_display=True

        # Random walks (default):
        python height_difference.py +num_random_walks=10

    :param config: Hydra config (use same config as skill_runner)
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

    # Video configuration
    show_videos = config.get("skill_runner_show_videos", True)
    make_video = config.evaluation.save_video or show_videos

    if not make_video:
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

    # Show topdown map if requested
    if config.get("skill_runner_show_topdown", False):
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
    cprint("=== Random Walk Script Started ===", "green")
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

    # Get all furniture in the scene
    world_graph = env_interface.world_graph[robot_agent.uid]
    all_furniture = world_graph.get_all_furnitures()

    if not all_furniture:
        cprint("No furniture found in the scene! Cannot perform random walks.", "red")
        return

    cprint(f"\nFound {len(all_furniture)} furniture items in the scene", "blue")
    cprint(f"Sample furniture: {[f.name for f in all_furniture[:5]]}", "blue")

    # Check exploration mode: "gpt" for GPT-guided, "random" for random walks
    exploration_mode = config.get("exploration_mode", "random")
    live_display = config.get("live_display", False)
    gpt_model = config.get("gpt_model", "gpt-4o-mini")

    cprint(f"\nExploration mode: {exploration_mode}", "blue")

    if exploration_mode == "gpt":
        # ========== GPT-GUIDED EXPLORATION MODE ==========
        cprint("\n" + "=" * 60, "green")
        cprint("STARTING GPT-GUIDED EXPLORATION", "green")
        cprint("(Will stop and generate tasks on first surprising finding)", "yellow")
        cprint("=" * 60 + "\n", "green")

        # Create RoomExplorer instance
        room_explorer = RoomExplorer(
            env_interface=env_interface,
            planner=planner,
            robot_agent=robot_agent,
            show_videos=show_videos,
            live_display=live_display,
            display_scale=1.0,
        )

        # Create GPT-guided explorer
        exploration_output_dir = f"{config.paths.results_dir}/gpt_exploration"
        gpt_explorer = GPTGuidedExplorer(
            room_explorer=room_explorer,
            gpt_model=gpt_model,
            output_dir=exploration_output_dir,
        )

        # Get the current mechanics directory for task output
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tasks_dir = ensure_tasks_directory(current_dir)

        # Initialize task generator for when we find a surprise
        task_generator = TaskGenerator(model=gpt_model)

        # Track if we generated tasks
        tasks_generated = False
        generated_tasks_file = None

        def on_surprise_found(
            session: ExplorationSession,
            room_visit: RoomVisit,
            interaction: Interaction,
            finding: str
        ) -> bool:
            """
            Callback called when a surprising finding is detected.
            Generates tasks based on the exploration and returns False to stop.
            """
            nonlocal tasks_generated, generated_tasks_file

            cprint("\n" + "=" * 60, "red")
            cprint("🎯 SURPRISING FINDING DETECTED!", "red")
            cprint(f"   Room: {room_visit.room_name}", "red")
            cprint(f"   Action: {interaction.action} on {interaction.target}", "red")
            cprint(f"   Finding: {finding}", "red")
            cprint("=" * 60, "red")

            cprint("\n[TaskGenerator] Generating tasks based on exploration...", "yellow")

            # Build exploration data dict from the session
            exploration_data = {
                "start_time": session.start_time,
                "end_time": session.end_time,
                "gpt_model": session.gpt_model,
                "total_interactions": session.total_interactions,
                "total_surprising_findings": session.total_surprising_findings,
                "rooms_visited": []
            }

            # Add all visited rooms so far
            for visit in session.rooms_visited:
                visit_dict = {
                    "room_name": visit.room_name,
                    "furniture_observed": visit.furniture_observed,
                    "objects_observed": visit.objects_observed,
                    "gpt_observations": visit.gpt_observations,
                    "gpt_hypotheses": visit.gpt_hypotheses,
                    "surprising_findings": visit.surprising_findings,
                    "interactions": [
                        {
                            "action": i.action,
                            "target": i.target,
                            "result": i.result,
                            "hypothesis": i.hypothesis,
                            "surprising": i.surprising,
                            "finding": i.finding
                        }
                        for i in visit.interactions
                    ]
                }
                exploration_data["rooms_visited"].append(visit_dict)

            # Add current room if not already in the list
            current_room_names = [r["room_name"] for r in exploration_data["rooms_visited"]]
            if room_visit.room_name not in current_room_names:
                exploration_data["rooms_visited"].append({
                    "room_name": room_visit.room_name,
                    "furniture_observed": room_visit.furniture_observed,
                    "objects_observed": room_visit.objects_observed,
                    "gpt_observations": room_visit.gpt_observations,
                    "gpt_hypotheses": room_visit.gpt_hypotheses,
                    "surprising_findings": room_visit.surprising_findings,
                    "interactions": [
                        {
                            "action": i.action,
                            "target": i.target,
                            "result": i.result,
                            "hypothesis": i.hypothesis,
                            "surprising": i.surprising,
                            "finding": i.finding
                        }
                        for i in room_visit.interactions
                    ]
                })
                exploration_data["total_surprising_findings"] += 1

            # Generate tasks
            num_tasks = config.get("num_tasks", 5)
            try:
                tasks = task_generator.generate_tasks_from_exploration(
                    exploration_data,
                    num_tasks=num_tasks
                )

                if tasks:
                    # Save tasks to file
                    from datetime import datetime
                    import json

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(tasks_dir, f"generated_tasks_{timestamp}.json")

                    output_data = {
                        "generation_time": datetime.now().isoformat(),
                        "triggered_by_surprise": finding,
                        "mechanics_directory": current_dir,
                        "gpt_model": gpt_model,
                        "num_tasks_requested": num_tasks,
                        "num_tasks_generated": len(tasks),
                        "exploration_summary": {
                            "rooms_visited": len(exploration_data["rooms_visited"]),
                            "total_interactions": exploration_data["total_interactions"],
                            "total_surprising_findings": exploration_data["total_surprising_findings"]
                        },
                        "tasks": tasks
                    }

                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)

                    tasks_generated = True
                    generated_tasks_file = output_file

                    cprint(f"\n[TaskGenerator] Generated {len(tasks)} tasks!", "green")
                    cprint(f"[TaskGenerator] Saved to: {output_file}", "green")

                    # Print task summary
                    cprint("\n--- Generated Tasks ---", "blue")
                    for task in tasks:
                        cprint(f"  {task.get('task_id', 'N/A')}: {task.get('name', 'Unnamed')}", "blue")
                        cprint(f"    Goal: {task.get('goal', 'N/A')[:60]}...", "blue")
                else:
                    cprint("[TaskGenerator] Warning: No tasks generated", "red")

            except Exception as e:
                cprint(f"[TaskGenerator] Error generating tasks: {e}", "red")

            # Return False to stop exploration
            cprint("\n[System] Stopping exploration and ending scene...", "yellow")
            return False

        # Run GPT-guided exploration with surprise callback
        max_rooms = config.get("max_rooms", None)  # None = explore all rooms
        max_interactions_per_room = config.get("max_interactions_per_room", 3)

        session, stopped_early = gpt_explorer.explore_with_gpt(
            max_rooms=max_rooms,
            max_interactions_per_room=max_interactions_per_room,
            on_surprise_callback=on_surprise_found,
            stop_on_first_surprise=True,
        )

        cprint("\n=== GPT-Guided Exploration Completed ===", "green")
        cprint(f"Rooms visited: {len(session.rooms_visited)}", "blue")
        cprint(f"Total interactions: {session.total_interactions}", "blue")
        cprint(f"Surprising findings: {session.total_surprising_findings}", "blue")
        cprint(f"Exploration results saved in: {exploration_output_dir}/", "blue")

        if tasks_generated:
            cprint("\n" + "=" * 60, "green")
            cprint("🎉 TASK GENERATION COMPLETE", "green")
            cprint(f"   Tasks file: {generated_tasks_file}", "green")
            cprint("=" * 60, "green")

        if stopped_early:
            cprint("\n[System] Scene ended after surprising finding detected.", "yellow")
            cprint("[System] Tasks have been generated based on the exploration.", "yellow")
            return  # End the scene

    else:
        # ========== RANDOM WALK MODE (default) ==========
        # Configuration for random walks
        num_walks = config.get("num_random_walks", 5)  # Default: 5 random walks
        cprint(f"\nPerforming {num_walks} random walks around the scene...\n", "green")

        # Collect frames for cumulative video
        cumulative_frames: List[Any] = []

        # Perform random walks
        for walk_idx in range(num_walks):
            # Pick a random furniture to navigate to
            target_furniture = random.choice(all_furniture)

            cprint(f"=== Walk {walk_idx + 1}/{num_walks} ===", "yellow")
            cprint(f"Navigating to: {target_furniture.name}", "yellow")

            # Create high-level action to navigate to this furniture
            # Format: {agent_id: (skill_name, target, None)}
            high_level_skill_actions = {
                robot_agent.uid: ("Navigate", target_furniture.name, None)
            }

            try:
                # Execute the navigate skill
                responses, _, frames = execute_skill(
                    high_level_skill_actions,
                    planner,
                    vid_postfix=f"walk_{walk_idx}_",
                    make_video=make_video,
                    play_video=show_videos,
                )

                # Print the response
                response_msg = responses[robot_agent.uid]
                cprint(f"Result: {response_msg}\n", "green")

                # Accumulate frames
                cumulative_frames.extend(frames)

            except Exception as e:
                cprint(f"Failed to execute walk {walk_idx + 1}: {str(e)}", "red")
                continue

        # Create cumulative video of all walks
        if len(cumulative_frames) > 0 and make_video:
            cprint("\n" + "="*60, "green")
            cprint("Creating COMBINED video of all walks...", "green")
            cprint("="*60, "green")

            dvu = DebugVideoUtil(env_interface, config.paths.results_dir)
            dvu.frames = cumulative_frames
            dvu._make_video(postfix="all_random_walks", play=show_videos)

            cumulative_video_path = f"{config.paths.results_dir}/videos/video-all_random_walks.mp4"
            cprint(f"\n✓ Combined video saved to: {cumulative_video_path}", "green")
            cprint(f"  Total frames: {len(cumulative_frames)}", "blue")

            if show_videos:
                cprint("\nPlaying combined video of all walks...", "yellow")

        cprint("\n=== Random Walk Script Completed ===", "green")
        cprint(f"Completed {num_walks} random walks.", "green")
        cprint(f"Videos saved in: {config.paths.results_dir}/videos/", "blue")


if __name__ == "__main__":
    cprint(
        "\nStarting exploration script for robot agent in Habitat environment.",
        "blue",
    )
    cprint(
        "Use +exploration_mode=gpt for GPT-guided exploration, or default for random walks.",
        "blue",
    )

    run_exploration()

    cprint(
        "\nExploration script finished.",
        "blue",
    )
