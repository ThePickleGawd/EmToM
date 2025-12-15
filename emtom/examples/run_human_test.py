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
from typing import Any, Dict, List, Optional, Tuple

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

from emtom import GameStateManager, list_mechanics, MECHANIC_INFO, ActionExecutionResult
from emtom.tools import get_emtom_tools
from emtom.actions.custom_actions import EMTOM_ACTIONS, EMTOMActionExecutor
from emtom.exploration.habitat_explorer import HabitatWorldAdapter


def interactive_select(
    title: str,
    items: List[Dict[str, Any]],
    default_selected: Optional[List[str]] = None,
) -> List[str]:
    """
    Interactive checkbox-style selection menu.

    Args:
        title: Title to display
        items: List of dicts with 'name' and 'description' keys
        default_selected: Names to select by default

    Returns:
        List of selected item names
    """
    selected = set(default_selected or [])

    while True:
        # Clear screen and show menu
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        print("Toggle: type number | Select all: 'a' | Clear: 'c' | Confirm: Enter")
        print("-" * 60)

        for i, item in enumerate(items, 1):
            name = item["name"]
            desc = item.get("description", "")[:40]
            check = "x" if name in selected else " "
            print(f"  [{check}] {i}. {name:<20} {desc}")

        print("-" * 60)
        print(f"Selected: {len(selected)}/{len(items)}")

        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Using current selection]")
            break

        if choice == "":
            # Confirm selection
            break
        elif choice == "a":
            # Select all
            selected = {item["name"] for item in items}
        elif choice == "c":
            # Clear all
            selected = set()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                name = items[idx]["name"]
                if name in selected:
                    selected.remove(name)
                else:
                    selected.add(name)
            else:
                print(f"Invalid number. Enter 1-{len(items)}")
        else:
            # Try to match by name prefix
            matches = [item["name"] for item in items if item["name"].lower().startswith(choice)]
            if len(matches) == 1:
                name = matches[0]
                if name in selected:
                    selected.remove(name)
                else:
                    selected.add(name)
            elif len(matches) > 1:
                print(f"Ambiguous: {matches}")
            else:
                print("Unknown input. Use number, 'a', 'c', or Enter")

    return list(selected)


def run_interactive_setup() -> Tuple[List[str], List[str]]:
    """
    Run interactive setup to select mechanics and actions.

    Returns:
        (selected_mechanics, selected_actions)
    """
    print("\n" + "=" * 60)
    print("EMTOM Interactive Setup")
    print("=" * 60)

    # Get available mechanics
    mechanic_items = []
    for name in list_mechanics():
        info = MECHANIC_INFO.get(name, {})
        mechanic_items.append({
            "name": name,
            "description": info.get("description", "No description"),
        })

    # Get available actions
    action_items = []
    for name, action in EMTOM_ACTIONS.items():
        action_items.append({
            "name": name,
            "description": getattr(action, "description", "Custom action"),
        })

    # Default selections
    default_mechanics = ["inverse_state", "remote_control"]
    default_actions = list(EMTOM_ACTIONS.keys())  # All actions enabled by default

    # Select mechanics
    selected_mechanics = interactive_select(
        "Select MECHANICS to enable:",
        mechanic_items,
        default_selected=default_mechanics,
    )

    # Select actions
    selected_actions = interactive_select(
        "Select CUSTOM ACTIONS to enable:",
        action_items,
        default_selected=default_actions,
    )

    print("\n" + "-" * 60)
    print("Configuration complete!")
    print(f"  Mechanics: {selected_mechanics or '(none)'}")
    print(f"  Actions: {selected_actions or '(none)'}")
    print("-" * 60)

    return selected_mechanics, selected_actions


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
        game_manager: GameStateManager,
        config: HumanTestConfig,
        task_info: Optional[Dict[str, Any]] = None,
        llm_agents: Optional[List[str]] = None,
    ):
        self.env = env_interface
        self.agents = agents
        self.game_manager = game_manager
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

        # World adapter for exploration
        self.world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)

        # Custom EMTOM action executor (for actions not handled by game manager)
        self.custom_action_executor = EMTOMActionExecutor(env_interface)

        # LLM client for delegation (lazy loaded)
        self._llm_client = None
        self._curiosity_model = None

        # Action history
        self.action_history: List[Dict[str, Any]] = []

        # Mechanic bindings (set during run_interactive)
        self._bindings: Dict[str, Any] = {}

        # Video recording
        self._dvu = None
        self._setup_video_recording()

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

        # Get entities from Habitat and bind mechanics to real objects
        print("\nSyncing scene and binding mechanics...")
        entities = self.world_adapter.get_interactable_entities()
        print(f"  Found {len(entities)} entities in scene")

        # Set entities on game state for auto-binding
        state = self.game_manager.get_state()
        state.entities = entities
        self.game_manager.set_state(state)

        # Auto-bind mechanics to real objects
        state, bindings = self.game_manager.auto_bind_mechanics()
        self._bindings = bindings  # Store for later display

        # Show what was bound
        if bindings:
            print("\n" + "-" * 50)
            print("MECHANIC BINDINGS (test these!):")
            print("-" * 50)
            for mech, info in bindings.items():
                if mech == "hidden_items":
                    for container, item in info.items():
                        print(f"  Shake[{container}] -> reveals {item}")
                elif mech == "inverse_state":
                    print(f"  inverse_state: Open/Close[{info['target']}] does the OPPOSITE")
                elif mech == "remote_control":
                    print(f"  remote_control: Open[{info['trigger']}] -> affects {info['target']}")
                elif mech == "counting_state":
                    print(f"  counting_state: Open[{info['target']}] {info['required_count']}x to activate")
                elif mech == "state_mirroring":
                    print(f"  state_mirroring: {info['pair'][0]} <-> {info['pair'][1]} stay in sync")
                elif mech == "conditional_unlock":
                    print(f"  conditional_unlock: Open[{info['prerequisite']}] first, then {info['target']}")
                elif mech == "delayed_effect":
                    print(f"  delayed_effect: Open[{info['target']}] takes {info['delay_steps']} steps")
            print("-" * 50)

        # Show task info if available
        if self.task_info:
            print(f"\nTask: {self.task_info.get('title', 'N/A')}")
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
                    # Human input loop - control commands don't consume turn
                    result = None
                    while result is None:
                        try:
                            command = input(f"\n{agent_id}> ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n[Interrupted]")
                            running = False
                            break

                        if not command:
                            continue

                        # Handle control commands (don't consume turn)
                        cmd = command.lower()
                        if cmd == "quit":
                            running = False
                            break
                        elif cmd == "help":
                            self._display_help()
                        elif cmd == "status":
                            self._display_full_status()
                        elif cmd == "mechanics":
                            self._display_mechanics(verbose=True)
                        elif cmd == "bindings":
                            self._display_bindings()
                        elif cmd == "history":
                            self._display_history()
                        # Action commands (consume turn)
                        elif cmd == "llm":
                            result = self._run_llm_turn(agent_id, uid)
                        else:
                            result = self._execute_command(agent_id, uid, command)

                    if not running:
                        break

                # Display result
                if result:
                    self._display_result(agent_id, result)

                # Record frame
                if result:
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

        # Get bound targets from bindings
        bound_targets = set()
        for mech, info in self._bindings.items():
            if mech == "hidden_items":
                bound_targets.update(info.keys())
            elif isinstance(info, dict):
                bound_targets.add(info.get("target", ""))
                bound_targets.add(info.get("trigger", ""))
                bound_targets.add(info.get("prerequisite", ""))
                if "pair" in info:
                    bound_targets.update(info["pair"])

        # Get nearby entities
        entities = self.world_adapter.get_interactable_entities()

        # Filter out floor entities, prioritize articulated and bound items
        furniture = [e for e in entities if e["type"] == "furniture" and not e["name"].startswith("floor_")]
        objects = [e for e in entities if e["type"] == "object"]

        # Sort: bound items first, then articulated, then others
        def sort_key(e):
            is_bound = e["name"] in bound_targets
            is_articulated = e.get("is_articulated", False)
            return (not is_bound, not is_articulated, e["name"])

        furniture.sort(key=sort_key)
        objects.sort(key=sort_key)

        # Show furniture (more items, mark bound ones with *)
        if furniture:
            furniture_strs = []
            for f in furniture[:12]:
                name = f["name"]
                marker = "*" if name in bound_targets else ""
                furniture_strs.append(f"{name}{marker}")
            print(f"Furniture: {', '.join(furniture_strs)}")
            if len(furniture) > 12:
                print(f"  ... and {len(furniture) - 12} more (type 'status' for full list)")

        # Show objects (mark bound ones with *)
        if objects:
            object_strs = [f"{o['name']}{'*' if o['name'] in bound_targets else ''}" for o in objects[:8]]
            print(f"Objects: {', '.join(object_strs)}")

        # Show rooms
        rooms = self.world_adapter.get_room_ids()
        if rooms:
            print(f"Rooms: {', '.join(rooms)}")

        # Show inventory
        state = self.game_manager.get_state()
        inventory = state.agent_inventory.get(agent_id, [])
        if inventory:
            print(f"Inventory: {', '.join(inventory)}")

    def _display_mechanics(self, verbose: bool = False):
        """Display active mechanics."""
        print("\nMechanics Active:")
        debug_info = self.game_manager.get_debug_info()
        active = debug_info.get("active_mechanics", [])

        if not active:
            print("  (none)")
            return

        for mech_name in active:
            mech_info = MECHANIC_INFO.get(mech_name, {})
            if verbose:
                print(f"  {mech_name}:")
                print(f"    Description: {mech_info.get('description', 'Unknown')}")

                # Show mechanic-specific state
                if mech_name == "inverse_state":
                    targets = debug_info.get("inverse_objects", [])
                    print(f"    Inverse targets: {targets}")
                elif mech_name == "remote_control":
                    mappings = debug_info.get("remote_mappings", {})
                    print(f"    Remote mappings: {mappings}")
                elif mech_name == "state_mirroring":
                    pairs = debug_info.get("mirror_pairs", [])
                    print(f"    Mirror pairs: {pairs}")
                elif mech_name == "counting_state":
                    counts = debug_info.get("interaction_counts", {})
                    print(f"    Interaction counts: {counts}")
                elif mech_name == "sequence_lock":
                    progress = debug_info.get("sequence_progress", {})
                    print(f"    Sequence progress: {progress}")
                elif mech_name == "conditional_unlock":
                    unlocked = debug_info.get("unlocked_targets", [])
                    print(f"    Unlocked: {unlocked}")
            else:
                desc = mech_info.get("description", "")[:50]
                print(f"  {mech_name}: {desc}...")

    def _display_bindings(self):
        """Display what objects mechanics are bound to (what to test)."""
        print("\n" + "=" * 50)
        print("MECHANIC BINDINGS - Test These Commands!")
        print("=" * 50)

        if not self._bindings:
            print("  (no bindings - mechanics not bound to objects)")
            return

        for mech, info in self._bindings.items():
            if mech == "hidden_items":
                for container, item in info.items():
                    print(f"\n  SHAKE TEST:")
                    print(f"    Command: Shake[{container}]")
                    print(f"    Expected: '{item}' falls out")
            elif mech == "inverse_state":
                print(f"\n  INVERSE STATE TEST:")
                print(f"    Command: Open[{info['target']}]")
                print(f"    Expected: It CLOSES instead (opposite)")
            elif mech == "remote_control":
                print(f"\n  REMOTE CONTROL TEST:")
                print(f"    Command: Open[{info['trigger']}]")
                print(f"    Expected: '{info['target']}' opens instead")
            elif mech == "counting_state":
                print(f"\n  COUNTING STATE TEST:")
                print(f"    Command: Open[{info['target']}] (repeat {info['required_count']}x)")
                print(f"    Expected: Only works after {info['required_count']} tries")
            elif mech == "state_mirroring":
                print(f"\n  STATE MIRRORING TEST:")
                print(f"    Command: Open[{info['pair'][0]}]")
                print(f"    Expected: '{info['pair'][1]}' also opens")
            elif mech == "conditional_unlock":
                print(f"\n  CONDITIONAL UNLOCK TEST:")
                print(f"    Step 1: Open[{info['prerequisite']}]")
                print(f"    Step 2: Open[{info['target']}]")
                print(f"    Expected: Target only works after prerequisite")
            elif mech == "delayed_effect":
                print(f"\n  DELAYED EFFECT TEST:")
                print(f"    Command: Open[{info['target']}]")
                print(f"    Expected: Takes {info['delay_steps']} steps to take effect")

        print("\n" + "=" * 50)

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
        print("  Shake[object]       - Shake object to reveal hidden items")
        print("  WriteMessage[furniture] - Leave a message")

        print("\nControl Commands:")
        print("  bindings            - Show what to test (mechanic bindings)")
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

    def _check_mechanics(self, action_name: str, target: str) -> Optional[Dict[str, Any]]:
        """
        Check if any mechanic should transform this action.

        Returns dict with:
        - mechanic: name of the mechanic
        - actual_action: action to actually execute
        - actual_target: target to actually affect
        - observation: description of what happened
        """
        state = self.game_manager.get_state()

        # Inverse state: open becomes close, close becomes open
        if target in state.inverse_objects:
            inverse_map = {"Open": "Close", "Close": "Open"}
            if action_name in inverse_map:
                opposite = inverse_map[action_name]
                return {
                    "mechanic": "inverse_state",
                    "actual_action": opposite,
                    "actual_target": target,
                    "observation": f"You try to {action_name.lower()} {target}, but it {opposite.lower()}s instead!",
                }

        # Remote control: acting on trigger affects a different target
        if target in state.remote_mappings:
            remote_target, remote_state = state.remote_mappings[target]
            return {
                "mechanic": "remote_control",
                "actual_action": action_name,
                "actual_target": remote_target,
                "observation": f"You {action_name.lower()} {target}, but {remote_target} responds instead!",
            }

        # Counting state: need N interactions before it works
        if target in state.interaction_counts:
            current = state.interaction_counts[target]
            required = state.object_properties.get(target, {}).get("required_count", 3)
            new_count = current + 1
            state.interaction_counts[target] = new_count
            self.game_manager.set_state(state)

            if new_count < required:
                return {
                    "mechanic": "counting_state",
                    "actual_action": action_name,
                    "actual_target": target,
                    "observation": f"You {action_name.lower()} {target}. It doesn't respond yet ({new_count}/{required}).",
                }
            else:
                return {
                    "mechanic": "counting_state",
                    "actual_action": action_name,
                    "actual_target": target,
                    "observation": f"You {action_name.lower()} {target}. After {required} attempts, it finally responds!",
                }

        # State mirroring: one object mirrors another
        for obj1, obj2, state_prop in state.mirror_pairs:
            if target == obj1:
                return {
                    "mechanic": "state_mirroring",
                    "actual_action": action_name,
                    "actual_target": target,
                    "observation": f"You {action_name.lower()} {target}. {obj2} also {action_name.lower()}s in sync!",
                }
            elif target == obj2:
                return {
                    "mechanic": "state_mirroring",
                    "actual_action": action_name,
                    "actual_target": target,
                    "observation": f"You {action_name.lower()} {target}. {obj1} also {action_name.lower()}s in sync!",
                }

        # Conditional unlock: check if prerequisite was done
        prereq = state.object_properties.get(target, {}).get("prerequisite")
        if prereq and target not in state.unlocked_targets:
            return {
                "mechanic": "conditional_unlock",
                "actual_action": action_name,
                "actual_target": target,
                "observation": f"You try to {action_name.lower()} {target}, but it seems locked. Maybe something else needs to happen first.",
            }

        # Check if this action unlocks something
        unlocks = state.object_properties.get(target, {}).get("unlocks")
        if unlocks:
            state.unlocked_targets.add(unlocks)
            self.game_manager.set_state(state)
            return {
                "mechanic": "conditional_unlock",
                "actual_action": action_name,
                "actual_target": target,
                "observation": f"You {action_name.lower()} {target}. You hear a click - something else is now accessible.",
            }

        return None

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

        # Check for mechanics on this action/target
        mechanic_effect = self._check_mechanics(action_name, target)

        # If mechanic transforms the action, apply it
        if mechanic_effect:
            actual_action = mechanic_effect.get("actual_action", action_name)
            actual_target = mechanic_effect.get("actual_target", target)
            print(f"  [Mechanic: {mechanic_effect['mechanic']}]")
        else:
            actual_action = action_name
            actual_target = target

        # Execute via agent tools
        agent = self.agents.get(uid)
        if not agent:
            return {"observation": f"No agent with uid {uid}"}

        if actual_action not in agent.tools:
            available = list(agent.tools.keys())
            return {"observation": f"Unknown action: {actual_action}. Available: {available}"}

        # Get observations
        obs = self.env.get_observations()

        # Execute the tool (with transformed action/target if mechanic applied)
        try:
            low_level_action, response = agent.process_high_level_action(
                actual_action, actual_target, obs
            )

            # If perception tool (returns None), just return response
            if low_level_action is None:
                obs_text = response
            else:
                # Execute motor skill until done
                skill_steps = 0
                max_steps = 500

                while skill_steps < max_steps:
                    raw_obs, reward, done, info = self.env.step({uid: low_level_action})
                    parsed_obs = self.env.parse_observations(raw_obs)
                    self._record_frame(parsed_obs, {uid: (actual_action, actual_target)})
                    skill_steps += 1

                    # Get next action
                    low_level_action, response = agent.process_high_level_action(
                        actual_action, actual_target, raw_obs
                    )
                    if low_level_action is None:
                        break

                obs_text = response or f"Executed {actual_action}[{actual_target}]"

            # Build observation with mechanic description
            if mechanic_effect:
                mechanic_obs = mechanic_effect.get("observation", "")
                if mechanic_obs:
                    obs_text = mechanic_obs
                elif actual_action != action_name or actual_target != target:
                    # Describe what actually happened
                    obs_text = f"You tried to {action_name} {target}, but {actual_action}d {actual_target} instead. {obs_text}"

            result = {
                "action": action_name,
                "target": target,
                "observation": obs_text,
                "success": True,
            }

            if mechanic_effect:
                result["mechanic"] = mechanic_effect["mechanic"]

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
        """Execute a custom EMTOM action through the GameStateManager."""
        # Sync state from Habitat first
        self.game_manager.sync_from_habitat()

        # Execute action through game manager (applies mechanics)
        state, exec_result = self.game_manager.apply_action(
            action_name, agent_id, target
        )

        # Tick time-based effects
        state, triggered = self.game_manager.tick()
        if triggered:
            print(f"[DELAYED EFFECTS] {', '.join(triggered)}")

        result = {
            "action": action_name,
            "target": target,
            "observation": exec_result.observation,
            "success": exec_result.success,
        }

        if exec_result.surprise_trigger:
            result["surprise"] = exec_result.surprise_trigger

        if exec_result.spawned_items:
            result["spawned"] = exec_result.spawned_items

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
    parser.add_argument("--task", type=str, default=None,
                        help="Task file to load (skips interactive setup)")

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

    # Setup GameStateManager
    print("\nSetting up GameStateManager...")
    game_manager = GameStateManager(env_interface)
    task_info = None

    # Build task data for game manager initialization
    task_data = {"mechanics": [], "enabled_actions": None}

    # Load task if specified (skips interactive)
    if extra_args and extra_args.task:
        task_info = load_task(extra_args.task)
        if task_info:
            print(f"Loaded task: {task_info.get('title', 'N/A')}")

            # Use mechanics from task
            if task_info.get("mechanic_bindings"):
                task_data["mechanics"] = task_info["mechanic_bindings"]
                print(f"  Mechanics: {len(task_data['mechanics'])} from task")

            # Copy hidden items from task
            if task_info.get("hidden_items"):
                task_data["hidden_items"] = task_info["hidden_items"]

            # Copy goals from task
            if task_info.get("goals"):
                task_data["goals"] = task_info["goals"]

    # Interactive mode is DEFAULT
    else:
        selected_mechanics, selected_actions = run_interactive_setup()
        for mech_name in selected_mechanics:
            task_data["mechanics"].append({"mechanic_type": mech_name})
        task_data["enabled_actions"] = selected_actions

    # Initialize game state from task data
    game_manager.initialize_from_task(task_data)
    print(f"  Active mechanics: {game_manager.get_debug_info()['active_mechanics']}")

    # Get output directory
    output_dir = config.paths.results_dir if hasattr(config, 'paths') else "data/emtom/human_test"

    # Create test config
    test_config = HumanTestConfig(
        max_rounds=100,
        save_video=True,
        output_dir=output_dir,
    )

    # Create and run test runner
    runner = HumanTestRunner(
        env_interface=env_interface,
        agents=agents,
        game_manager=game_manager,
        config=test_config,
        task_info=task_info,
    )

    runner.run_interactive()


if __name__ == "__main__":
    # Parse extra args before Hydra
    extra_args = parse_extra_args()
    main._extra_args = extra_args

    main()
