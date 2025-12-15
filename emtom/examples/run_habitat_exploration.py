#!/usr/bin/env python3
# isort: skip_file
"""
Run EMTOM exploration using the Habitat simulator backend.

This script runs exploration in the actual Habitat environment, ensuring
the action space and objects match what will be used in benchmark evaluation.
Videos are generated showing the exploration process.

Usage:
    python run_habitat_exploration.py --steps 50
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra

from habitat_llm.utils import setup_config, fix_config
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0


def run_exploration_loop(env_interface, config, max_steps=50, seed=42):
    """
    Run the exploration loop with the given environment.

    Args:
        env_interface: Initialized EnvironmentInterface
        config: Hydra config
        max_steps: Maximum exploration steps
        seed: Random seed
    """
    from emtom.exploration.habitat_explorer import (
        HabitatExplorer,
        HabitatExplorationConfig,
        HabitatWorldAdapter,
    )
    from emtom.exploration.curiosity import CuriosityModel
    from emtom.exploration.surprise_detector import SurpriseDetector
    from emtom import GameStateManager, list_mechanics
    from emtom.tools import get_emtom_tools
    from habitat_llm.agent import Agent

    # Get output directory from config
    output_dir = config.paths.results_dir if hasattr(config, 'paths') else "data/emtom/trajectories"
    os.makedirs(output_dir, exist_ok=True)

    # Get scene info
    current_episode = env_interface.env.env.env._env.current_episode
    scene_id = getattr(current_episode, 'scene_id', "unknown")
    episode_id = getattr(current_episode, 'episode_id', "unknown")

    print(f"Scene: {scene_id}")
    print(f"Episode: {episode_id}")
    print(f"Output: {output_dir}")

    # Create Agent with partnr tools
    print("\nCreating Agent with partnr tools...")
    agent = None
    agent_conf = None

    # Check config.agents (from root config)
    if hasattr(config, 'agents') and config.agents:
        agent_list = list(config.agents.values())
        if agent_list:
            agent_conf = agent_list[0]
            print(f"  Found agent config in config.agents")

    # Check config.evaluation.agents (from evaluation config)
    elif hasattr(config, 'evaluation') and hasattr(config.evaluation, 'agents') and config.evaluation.agents:
        agent_list = list(config.evaluation.agents.values())
        if agent_list:
            agent_conf = agent_list[0]
            print(f"  Found agent config in config.evaluation.agents")

    if agent_conf and hasattr(agent_conf, 'config'):
        try:
            agent = Agent(
                uid=agent_conf.get('uid', 0),
                agent_conf=agent_conf.config,
                env_interface=env_interface,
            )
            print(f"  partnr tools: {list(agent.tools.keys())}")

            # Inject EMTOM tools into the agent
            print("\n  Injecting EMTOM tools...")
            agent_uid = agent_conf.get('uid', 0)
            emtom_tools = get_emtom_tools(agent_uid=agent_uid)
            for tool_name, tool in emtom_tools.items():
                tool.set_environment(env_interface)
                agent.tools[tool_name] = tool
                print(f"    Added {tool_name}")

            print(f"  All tools available: {list(agent.tools.keys())}")
        except Exception as e:
            print(f"  Failed to create agent: {e}")
            agent = None
    else:
        print("  WARNING: No agent config found")

    # Setup GameStateManager with mechanics
    print("\nSetting up GameStateManager...")
    game_manager = GameStateManager(env_interface)

    # Use all available mechanics for exploration
    all_mechanics = list_mechanics()
    task_data = {
        "mechanics": [{"mechanic_type": m} for m in all_mechanics]
    }
    game_manager.initialize_from_task(task_data)
    print(f"  Active mechanics: {all_mechanics}")

    # Get entities and auto-bind mechanics
    world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)
    entities = world_adapter.get_interactable_entities()
    state = game_manager.get_state()
    state.entities = entities
    game_manager.set_state(state)

    state, bindings = game_manager.auto_bind_mechanics()
    print(f"  Bindings: {bindings}")

    # Lock random doors with codes and spawn matching keys
    state, locked_doors = game_manager.lock_random_doors(num_doors=2)
    if locked_doors:
        print()
        print("\033[93m" + "=" * 60 + "\033[0m")
        print("\033[93m  🔒 LOCKED DOORS (require matching key to open)\033[0m")
        print("\033[93m" + "=" * 60 + "\033[0m")
        for door, code in locked_doors.items():
            print(f"\033[91m    🚪 {door} \033[93;1m[#{code}]\033[0m")
        print()
        # Spawn keys for each locked door
        state, key_spawn_info_list = game_manager.spawn_keys_for_locked_doors()
        print("\033[96m" + "-" * 60 + "\033[0m")
        print("\033[96m  🗝️  KEYS SPAWNED (agent must find and collect)\033[0m")
        print("\033[96m" + "-" * 60 + "\033[0m")
        for key_info in key_spawn_info_list:
            print(f"\033[92m    🗝️  key \033[96;1m[#{key_info['code']}]\033[0m \033[90m→ on \033[95m{key_info['location']}\033[0m")
        print("\033[93m" + "=" * 60 + "\033[0m")
        print()
    else:
        print("\033[93m  Warning: No doors found to lock\033[0m")
        # Fall back to spawning a regular key
        state, key_spawn_info = game_manager.spawn_key_on_table()
        if key_spawn_info:
            print(f"\033[96m  ★ Key spawned on: {key_spawn_info['location']} ★\033[0m")

    # Pass game_manager to EMTOM tools so they can access hidden_items
    if agent is not None:
        print("  Passing game_manager to EMTOM tools...")
        for tool_name, tool in agent.tools.items():
            if hasattr(tool, 'set_game_manager'):
                tool.set_game_manager(game_manager)

    # Setup LLM client
    print("\nSetting up LLM client...")
    from habitat_llm.llm import instantiate_llm
    llm_client = instantiate_llm("openai_chat")
    print(f"  Using model: {llm_client.generation_params.model}")

    # Pass LLM to agent tools
    if agent is not None:
        print("  Passing LLM to agent tools...")
        agent.pass_llm_to_tools(llm_client)

    # Setup curiosity model
    print("\nSetting up curiosity model...")
    llm_config = None
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'agents'):
        agent_list = list(config.evaluation.agents.values())
        if agent_list and hasattr(agent_list[0], 'planner') and hasattr(agent_list[0].planner, 'llm'):
            llm_config = agent_list[0].planner.llm

    curiosity = CuriosityModel(llm_client, llm_config=llm_config)
    print("  LLM-guided exploration enabled")

    # Setup surprise detector
    print("\nSetting up surprise detector...")
    surprise = SurpriseDetector(llm_client)
    print("  LLM-based surprise detection enabled")

    # Check if video saving is enabled
    save_video = True
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'save_video'):
        save_video = config.evaluation.save_video

    # Setup exploration config
    exploration_config = HabitatExplorationConfig(
        max_steps=max_steps,
        agent_ids=["agent_0"],
        log_path=output_dir,
        save_video=save_video,
        play_video=False,
        save_fpv=True,
    )

    # Create explorer with new GameStateManager
    print("\nCreating Habitat explorer...")
    explorer = HabitatExplorer(
        env_interface=env_interface,
        game_manager=game_manager,
        curiosity_model=curiosity,
        surprise_detector=surprise,
        agent=agent,
        config=exploration_config,
    )

    # Run exploration
    print(f"\nRunning exploration for {max_steps} steps...")
    print("-" * 40)

    metadata = {
        "seed": seed,
        "mode": "llm",
        "scene_id": scene_id,
        "episode_id": episode_id,
    }

    episode_data = explorer.run(metadata=metadata)

    # Print results
    print("\n" + "=" * 60)
    print("EXPLORATION RESULTS")
    print("=" * 60)

    stats = episode_data.get("statistics", {})
    print(f"Total steps: {stats.get('total_steps', 'N/A')}")
    print(f"Total surprises: {stats.get('total_surprises', 0)}")
    print(f"Actions per agent: {stats.get('actions_per_agent', {})}")
    print(f"Unique actions: {stats.get('unique_actions', 0)}")

    # Print mechanic bindings
    if episode_data.get("messages"):
        print("\nMechanic binding info:")
        for msg in episode_data["messages"][:10]:
            print(f"  {msg}")

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
    trajectory_file = f"{output_dir}/trajectory_{episode_data['episode_id']}.json"
    print(f"\nTrajectory saved to: {trajectory_file}")

    # Copy trajectory to data/emtom/trajectories for task generation
    import shutil
    data_traj_dir = Path("data/emtom/trajectories")
    data_traj_dir.mkdir(parents=True, exist_ok=True)
    dest_file = data_traj_dir / f"trajectory_{episode_data['episode_id']}.json"
    shutil.copy2(trajectory_file, dest_file)
    print(f"Copied to: {dest_file}")

    return episode_data


@hydra.main(config_path="../../habitat_llm/conf", version_base=None)
def main(config):
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

    # Create dataset if needed
    dataset = None
    if hasattr(config.habitat, 'dataset'):
        try:
            dataset = CollaborationDatasetV0(config.habitat.dataset)
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")

    # Create environment interface
    print("Creating environment...")
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Run exploration
    t0 = time.time()
    try:
        episode_data = run_exploration_loop(
            env_interface=env_interface,
            config=config,
            max_steps=max_steps,
            seed=seed,
        )
    except Exception as e:
        print(f"Exploration failed: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print(f"\nExploration completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
