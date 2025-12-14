#!/usr/bin/env python3
# isort: skip_file
"""
Human-in-the-loop (HITL) testing mode for EMTOM benchmark.

This script allows manual testing of mechanics, custom actions, and task definitions
by letting a human type commands directly instead of using LLM-generated actions.

Usage:
    # Run with specific mechanics
    python emtom/examples/run_human_test.py \
        --config-name examples/planner_multi_agent_demo_config \
        --mechanics inverse_state remote_control

    # Run with task file
    python emtom/examples/run_human_test.py \
        --config-name examples/planner_multi_agent_demo_config \
        --task data/emtom/tasks/emtom_tom_test.json

    # Make agent_1 always LLM-controlled
    python emtom/examples/run_human_test.py \
        --config-name examples/planner_multi_agent_demo_config \
        --llm-agents agent_1
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from habitat_llm.utils import setup_config, fix_config
from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.agent import Agent

from emtom.mechanics import (
    MechanicRegistry,
    Mechanic,
    wrap_tools_with_mechanics,
)
from emtom.tools import get_emtom_tools
from emtom.actions.custom_actions import EMTOMActionExecutor, EMTOM_ACTIONS
from emtom.exploration.habitat_explorer import HabitatWorldAdapter, HabitatMechanicWorldState


@dataclass
class HumanTestConfig:
    """Configuration for human test mode."""
    max_rounds: int = 100
    save_video: bool = True
    output_dir: str = "data/emtom/human_test"


class HumanTestRunner:
    """
    Interactive test runner for EMTOM mechanics and actions.

    Allows human to control agents via CLI commands while using
    the same environment and mechanics system as exploration/benchmark.
    """

    def __init__(
        self,
        env_interface: EnvironmentInterface,
        agents: Dict[int, Agent],
        mechanics: List[Mechanic],
        config: HumanTestConfig,
        task_info: Optional[Dict[str, Any]] = None,
        llm_agents: Optional[List[str]] = None,
    ):
        self.env = env_interface
        self.agents = agents
        self.mechanics = mechanics
        self.config = config
        self.task_info = task_info

        # Agent control modes: "human" or "llm"
        self.agent_modes: Dict[str, str] = {}
        for uid in agents.keys():
            agent_id = f"agent_{uid}"
            if llm_agents and agent_id in llm_agents:
                self.agent_modes[agent_id] = "llm"
            else:
                self.agent_modes[agent_id] = "human"

        # World adapter for mechanics
        self.world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)

        # Custom EMTOM action executor
        self.custom_action_executor = EMTOMActionExecutor(env_interface, mechanics)

        # LLM client for delegation (lazy loaded)
        self._llm_client = None
        self._curiosity_model = None

        # Action history
        self.action_history: List[Dict[str, Any]] = []

        # Video recording
        self._dvu = None
        self._setup_video_recording()

        # Wrap agent tools with mechanics
        self._wrap_tools_with_mechanics()

    def _setup_video_recording(self):
        """Setup video recording utilities."""
        if not self.config.save_video:
            return

        try:
            from habitat_llm.examples.example_utils import DebugVideoUtil
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._dvu = DebugVideoUtil(
                self.env,
                self.config.output_dir,
                unique_postfix=True,
            )
        except ImportError as e:
            print(f"[Warning] Video utils not available: {e}")

    def _wrap_tools_with_mechanics(self):
        """Wrap agent tools with mechanics system."""
        if not self.mechanics:
            return

        world_state = HabitatMechanicWorldState(self.world_adapter)

        for uid, agent in self.agents.items():
            agent.tools = wrap_tools_with_mechanics(
                agent_tools=agent.tools,
                mechanics=self.mechanics,
                world_state_adapter=world_state,
            )

    def _get_llm_client(self):
        """Lazy-load LLM client for delegation."""
        if self._llm_client is None:
            from habitat_llm.llm import instantiate_llm
            self._llm_client = instantiate_llm("openai_chat")
        return self._llm_client

    def _get_curiosity_model(self):
        """Lazy-load curiosity model for LLM turns."""
        if self._curiosity_model is None:
            from emtom.exploration.curiosity import CuriosityModel
            self._curiosity_model = CuriosityModel(self._get_llm_client())
        return self._curiosity_model

    def run_interactive(self):
        """Main interactive loop."""
        print("\n" + "=" * 70)
        print("EMTOM Human Test Mode")
        print("=" * 70)

        # Show task info if available
        if self.task_info:
            print(f"Task: {self.task_info.get('title', 'N/A')}")
            print(f"Description: {self.task_info.get('description', 'N/A')}")

        # Show active mechanics
        self._display_mechanics()

        # Show agent modes
        print("\nAgent Control Modes:")
        for agent_id, mode in self.agent_modes.items():
            print(f"  {agent_id}: {mode.upper()}")

        print("\nType 'help' for available commands.")
        print("=" * 70)

        # Record initial frame
        obs = self.env.get_observations()
        self._record_frame(obs, {})

        round_num = 0
        running = True

        while running and round_num < self.config.max_rounds:
            round_num += 1
            print(f"\n{'=' * 70}")
            print(f"Round {round_num}")
            print("=" * 70)

            for uid, agent in self.agents.items():
                agent_id = f"agent_{uid}"

                # Display agent state
                self._display_agent_state(agent_id, uid)

                # Get action based on mode
                if self.agent_modes[agent_id] == "llm":
                    print(f"\n[{agent_id} is LLM-controlled]")
                    result = self._run_llm_turn(agent_id, uid)
                else:
                    # Human input
                    try:
                        command = input(f"\n{agent_id}> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[Interrupted]")
                        running = False
                        break

                    if not command:
                        continue

                    # Handle special commands
                    if command.lower() == "quit":
                        running = False
                        break
                    elif command.lower() == "llm":
                        result = self._run_llm_turn(agent_id, uid)
                    elif command.lower() == "help":
                        self._display_help()
                        continue
                    elif command.lower() == "status":
                        self._display_full_status()
                        continue
                    elif command.lower() == "mechanics":
                        self._display_mechanics(verbose=True)
                        continue
                    elif command.lower() == "history":
                        self._display_history()
                        continue
                    else:
                        result = self._execute_command(agent_id, uid, command)

                # Display result
                if result:
                    self._display_result(agent_id, result)

                # Record frame
                obs = self.env.get_observations()
                self._record_frame(obs, {uid: (result.get("action", ""), result.get("target", ""))})

        # Save video and cleanup
        self._finish()

    def _display_agent_state(self, agent_id: str, uid: int):
        """Display current state for an agent."""
        mode = self.agent_modes[agent_id].upper()

        # Get agent location
        location = self.world_adapter.get_agent_location(agent_id)
        location_str = location or "unknown"

        print(f"\n--- {agent_id} ({mode}) in {location_str} ---")

        # Get nearby entities
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]

        # Show furniture with states
        if furniture:
            furniture_strs = []
            for f in furniture[:8]:
                state_parts = []
                for key, val in f.get("states", {}).items():
                    if isinstance(val, bool):
                        state_parts.append(f"{key}={val}")
                state_str = f" ({', '.join(state_parts)})" if state_parts else ""
                furniture_strs.append(f"{f['name']}{state_str}")
            print(f"Furniture: {', '.join(furniture_strs)}")

        # Show objects
        if objects:
            object_names = [o["name"] for o in objects[:8]]
            print(f"Objects: {', '.join(object_names)}")

        # Show rooms
        rooms = self.world_adapter.get_room_ids()
        if rooms:
            print(f"Rooms: {', '.join(rooms)}")

    def _display_mechanics(self, verbose: bool = False):
        """Display active mechanics."""
        print("\nMechanics Active:")
        if not self.mechanics:
            print("  (none)")
            return

        for mechanic in self.mechanics:
            if verbose:
                debug_state = mechanic.get_hidden_state_for_debug()
                print(f"  {mechanic.name}:")
                print(f"    Description: {mechanic.description}")
                print(f"    Bound targets: {debug_state.get('bound_targets', [])}")
                if "mappings" in debug_state:
                    print(f"    Mappings: {debug_state['mappings']}")
                if "target_thresholds" in debug_state:
                    print(f"    Thresholds: {debug_state['target_thresholds']}")
                if "interaction_counts" in debug_state:
                    counts = debug_state["interaction_counts"]
                    if counts:
                        print(f"    Interaction counts: {counts}")
            else:
                debug_state = mechanic.get_hidden_state_for_debug()
                targets = debug_state.get("bound_targets", [])
                print(f"  {mechanic.name}: {targets}")

    def _display_help(self):
        """Display help message."""
        print("\n" + "=" * 50)
        print("Available Commands")
        print("=" * 50)

        print("\nStandard Habitat Tools:")
        print("  Open[object]        - Open articulated furniture")
        print("  Close[object]       - Close articulated furniture")
        print("  Navigate[target]    - Move to room or furniture")
        print("  Pick[object]        - Pick up an object")
        print("  Place[receptacle]   - Place held object")
        print("  Explore[room]       - Search a room")

        print("\nCustom EMTOM Actions:")
        print("  Hide[object]        - Hide object from view")
        print("  Inspect[target]     - Examine object properties")
        print("  WriteMessage[furniture] - Leave a message")

        print("\nControl Commands:")
        print("  llm                 - Let LLM take this turn")
        print("  status              - Show full world state")
        print("  mechanics           - Show mechanic details")
        print("  history             - Show action history")
        print("  help                - Show this help")
        print("  quit                - Exit and save video")

    def _display_full_status(self):
        """Display full world status."""
        print("\n" + "=" * 50)
        print("World Status")
        print("=" * 50)

        entities = self.world_adapter.get_interactable_entities()

        print("\nAll Furniture:")
        for e in entities:
            if e["type"] == "furniture":
                states = e.get("states", {})
                state_str = ", ".join(f"{k}={v}" for k, v in states.items() if isinstance(v, bool))
                print(f"  {e['name']}: {state_str or '(no states)'}")

        print("\nAll Objects:")
        for e in entities:
            if e["type"] == "object":
                print(f"  {e['name']}")

        print("\nRooms:")
        for room in self.world_adapter.get_room_ids():
            print(f"  {room}")

    def _display_history(self):
        """Display recent action history."""
        print("\n" + "=" * 50)
        print("Action History (last 10)")
        print("=" * 50)

        for entry in self.action_history[-10:]:
            agent = entry.get("agent_id", "?")
            action = entry.get("action", "?")
            target = entry.get("target", "")
            obs = entry.get("observation", "")[:60]
            print(f"  {agent}: {action}[{target}] -> {obs}...")

    def _execute_command(self, agent_id: str, uid: int, command: str) -> Dict[str, Any]:
        """Execute a human command."""
        # Parse command: Action[target]
        match = re.match(r"(\w+)\[([^\]]*)\]", command)
        if not match:
            # Try without brackets
            match = re.match(r"(\w+)\s*(.*)", command)
            if match:
                action_name = match.group(1)
                target = match.group(2).strip()
            else:
                return {"observation": f"Invalid command format: {command}. Use Action[target]"}
        else:
            action_name = match.group(1)
            target = match.group(2)

        print(f"Executing: {action_name}[{target}]")

        # Check if custom EMTOM action
        if action_name in EMTOM_ACTIONS:
            return self._execute_custom_action(agent_id, action_name, target)

        # Execute via agent tools
        agent = self.agents.get(uid)
        if not agent:
            return {"observation": f"No agent with uid {uid}"}

        if action_name not in agent.tools:
            available = list(agent.tools.keys())
            return {"observation": f"Unknown action: {action_name}. Available: {available}"}

        # Get observations
        obs = self.env.get_observations()

        # Execute the tool
        try:
            low_level_action, response = agent.process_high_level_action(
                action_name, target, obs
            )

            # If perception tool (returns None), just return response
            if low_level_action is None:
                result = {
                    "action": action_name,
                    "target": target,
                    "observation": response,
                    "success": True,
                }
            else:
                # Execute motor skill until done
                skill_steps = 0
                max_steps = 500

                while skill_steps < max_steps:
                    raw_obs, reward, done, info = self.env.step({uid: low_level_action})
                    parsed_obs = self.env.parse_observations(raw_obs)
                    self._record_frame(parsed_obs, {uid: (action_name, target)})
                    skill_steps += 1

                    # Get next action
                    low_level_action, response = agent.process_high_level_action(
                        action_name, target, raw_obs
                    )
                    if low_level_action is None:
                        break

                result = {
                    "action": action_name,
                    "target": target,
                    "observation": response or f"Executed {action_name}[{target}]",
                    "success": True,
                    "steps": skill_steps,
                }

            # Record in history
            self.action_history.append({
                "agent_id": agent_id,
                **result,
            })

            return result

        except Exception as e:
            return {
                "action": action_name,
                "target": target,
                "observation": f"Error: {e}",
                "success": False,
            }

    def _execute_custom_action(self, agent_id: str, action_name: str, target: str) -> Dict[str, Any]:
        """Execute a custom EMTOM action."""
        entities = self.world_adapter.get_interactable_entities()
        rooms = self.world_adapter.get_room_ids()

        world_state = {
            "agent_location": self.world_adapter.get_agent_location(agent_id),
            "rooms": rooms,
            "entities": entities,
            "entity_details": {
                e["name"]: {
                    "properties": e.get("properties", {}),
                    "states": e.get("states", {}),
                }
                for e in entities
            },
        }

        custom_result = self.custom_action_executor.execute(
            action_name, agent_id, target, world_state
        )

        result = {
            "action": action_name,
            "target": target,
            "observation": custom_result.observation,
            "success": custom_result.success,
        }

        if custom_result.surprise_trigger:
            result["surprise"] = custom_result.surprise_trigger

        self.action_history.append({
            "agent_id": agent_id,
            **result,
        })

        return result

    def _run_llm_turn(self, agent_id: str, uid: int) -> Dict[str, Any]:
        """Let LLM take one turn for this agent."""
        print("[LLM TURN] Thinking...")

        curiosity = self._get_curiosity_model()

        # Build world description
        world_text = self._build_world_description(agent_id)
        available_actions = self._get_available_actions(agent_id)

        # Get action from curiosity model
        action_choice = curiosity.select_action(
            agent_id=agent_id,
            world_description=world_text,
            available_actions=available_actions,
            exploration_history=self.action_history[-5:],
        )

        print(f"[LLM] Thought: {action_choice.reasoning}")
        print(f"[LLM] Action: {action_choice.action}[{action_choice.target or ''}]")

        # Execute the action
        command = f"{action_choice.action}[{action_choice.target or ''}]"
        return self._execute_command(agent_id, uid, command)

    def _build_world_description(self, agent_id: str) -> str:
        """Build world description for LLM."""
        lines = []
        location = self.world_adapter.get_agent_location(agent_id)
        if location:
            lines.append(f"You are in {location}.")

        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]

        if furniture:
            lines.append(f"Furniture: {', '.join(f['name'] for f in furniture[:10])}")
        if objects:
            lines.append(f"Objects: {', '.join(o['name'] for o in objects[:10])}")

        rooms = self.world_adapter.get_room_ids()
        if rooms:
            lines.append(f"Rooms: {', '.join(rooms)}")

        return "\n".join(lines)

    def _get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get available actions for LLM."""
        actions = []

        rooms = self.world_adapter.get_room_ids()
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]
        articulated = [f for f in furniture if f.get("is_articulated")]

        if rooms:
            actions.append({"name": "Navigate", "targets": rooms[:5]})
        if articulated:
            actions.append({"name": "Open", "targets": [f["name"] for f in articulated[:5]]})
            actions.append({"name": "Close", "targets": [f["name"] for f in articulated[:5]]})
        if objects:
            actions.append({"name": "Pick", "targets": [o["name"] for o in objects[:5]]})

        # Add EMTOM actions
        actions.append({"name": "Inspect", "targets": [e["name"] for e in entities[:5]]})

        return actions

    def _display_result(self, agent_id: str, result: Dict[str, Any]):
        """Display action result."""
        agent_uid = agent_id.split("_")[-1]
        obs = result.get("observation", "")
        print(f"Agent_{agent_uid}_Observation: {obs}")

        # Show mechanic info if surprise triggered
        if result.get("surprise"):
            print(f"[MECHANIC] {result['surprise']}")

        # Show step count for motor skills
        if result.get("steps"):
            print(f"[Motor skill completed in {result['steps']} steps]")

    def _record_frame(self, obs: Dict[str, Any], actions: Dict[int, Any]):
        """Record video frame."""
        if self._dvu:
            try:
                self._dvu._store_for_video(obs, actions, popup_images={})
            except Exception:
                pass

    def _finish(self):
        """Cleanup and save video."""
        print("\n" + "=" * 50)
        print("Finishing...")

        if self._dvu and self._dvu.frames:
            try:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._dvu._make_video(play=False, postfix=f"human_test_{timestamp}")
                print(f"Video saved to {self.config.output_dir}")
            except Exception as e:
                print(f"Could not save video: {e}")

        print("Done!")


def load_task(task_file: str) -> Dict[str, Any]:
    """Load task from JSON file."""
    with open(task_file) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if tasks:
        return tasks[0]
    return None


def parse_extra_args():
    """Parse extra CLI arguments before Hydra."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mechanics", nargs="*", default=None,
                        help="Mechanics to enable (e.g., inverse_state remote_control)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task file to load")
    parser.add_argument("--use-task-mechanics", action="store_true",
                        help="Use mechanics from task file bindings")
    parser.add_argument("--llm-agents", nargs="*", default=None,
                        help="Agents to make LLM-controlled (e.g., agent_1)")

    # Parse known args, let Hydra handle the rest
    args, remaining = parser.parse_known_args()

    # Reconstruct sys.argv for Hydra
    sys.argv = [sys.argv[0]] + remaining

    return args


@hydra.main(config_path="../../habitat_llm/conf", version_base=None)
def main(config: DictConfig):
    """Main entry point."""
    # Get extra args (parsed before Hydra)
    extra_args = getattr(main, "_extra_args", None)

    print("=" * 70)
    print("EMTOM Human Test Mode")
    print("=" * 70)

    # Fix and setup config
    fix_config(config)
    seed = 47668090
    config = setup_config(config, seed)

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

    # Create environment
    print("Creating environment...")
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Create agents
    print("Creating agents...")
    agents: Dict[int, Agent] = {}

    # Find agent configs
    agent_configs = []
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'agents'):
        agent_configs = list(config.evaluation.agents.values())
    elif hasattr(config, 'agents') and config.agents:
        agent_configs = list(config.agents.values())

    for agent_conf in agent_configs:
        if hasattr(agent_conf, 'config'):
            uid = agent_conf.get('uid', len(agents))
            try:
                agent = Agent(
                    uid=uid,
                    agent_conf=agent_conf.config,
                    env_interface=env_interface,
                )

                # Add EMTOM tools
                emtom_tools = get_emtom_tools(agent_uid=uid)
                for tool_name, tool in emtom_tools.items():
                    tool.set_environment(env_interface)
                    agent.tools[tool_name] = tool

                agents[uid] = agent
                print(f"  Created agent_{uid} with tools: {list(agent.tools.keys())}")
            except Exception as e:
                print(f"  Failed to create agent_{uid}: {e}")

    if not agents:
        print("ERROR: No agents created!")
        return

    # Setup mechanics
    print("\nSetting up mechanics...")
    mechanics: List[Mechanic] = []
    task_info = None

    # Load task if specified
    if extra_args and extra_args.task:
        task_info = load_task(extra_args.task)
        if task_info:
            print(f"Loaded task: {task_info.get('title', 'N/A')}")

            # Use task mechanics if requested
            if extra_args.use_task_mechanics and task_info.get("mechanic_bindings"):
                bindings = task_info["mechanic_bindings"]
                mechanics = MechanicRegistry.instantiate_from_bindings(bindings)
                print(f"  Loaded {len(mechanics)} mechanics from task bindings")

    # Otherwise use CLI-specified mechanics
    if not mechanics and extra_args and extra_args.mechanics:
        for mech_name in extra_args.mechanics:
            try:
                mechanic = MechanicRegistry.instantiate(mech_name)
                mechanics.append(mechanic)
                print(f"  Added {mech_name}")
            except KeyError:
                print(f"  Warning: Unknown mechanic '{mech_name}'")
                print(f"  Available: {MechanicRegistry.list_all()}")

    # Default mechanics if none specified
    if not mechanics:
        print("  No mechanics specified - using defaults")
        mechanics = [
            MechanicRegistry.instantiate("inverse_state"),
            MechanicRegistry.instantiate("remote_control"),
        ]
        print(f"  Added: {[m.name for m in mechanics]}")

    # Bind mechanics to scene
    world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)
    world_state = HabitatMechanicWorldState(world_adapter)

    for mechanic in mechanics:
        mechanic.reset()
        if hasattr(mechanic, 'bind_to_scene'):
            mechanic.bind_to_scene(world_state)

    # Get output directory
    output_dir = config.paths.results_dir if hasattr(config, 'paths') else "data/emtom/human_test"

    # Create test config
    test_config = HumanTestConfig(
        max_rounds=100,
        save_video=True,
        output_dir=output_dir,
    )

    # Get LLM agents
    llm_agents = extra_args.llm_agents if extra_args else None

    # Create and run test runner
    runner = HumanTestRunner(
        env_interface=env_interface,
        agents=agents,
        mechanics=mechanics,
        config=test_config,
        task_info=task_info,
        llm_agents=llm_agents,
    )

    runner.run_interactive()


if __name__ == "__main__":
    # Parse extra args before Hydra
    extra_args = parse_extra_args()
    main._extra_args = extra_args

    main()
