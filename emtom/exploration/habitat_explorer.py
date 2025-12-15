"""
Habitat-integrated exploration for EMTOM benchmark.

Uses the actual Habitat simulator backend instead of TextWorldState,
ensuring the exploration action space matches the benchmark environment.

This module uses partnr's built-in tools:
- Perception: FindObjectTool, FindRoomTool, FindReceptacleTool
- Motor Skills: OracleNavSkill, OracleOpenSkill, OracleCloseSkill, OraclePickSkill
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from emtom.exploration.curiosity import ActionChoice, CuriosityModel
from emtom.exploration.surprise_detector import SurpriseDetector
from emtom.exploration.trajectory_logger import SurpriseRecord, TrajectoryLogger
from emtom.actions.custom_actions import EMTOM_ACTIONS

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.agent import Agent
    from emtom import GameStateManager


@dataclass
class HabitatExplorationConfig:
    """Configuration for Habitat-backed exploration."""

    max_steps: int = 100
    agent_ids: List[str] = field(default_factory=lambda: ["agent_0"])
    log_path: str = "data/emtom/trajectories"
    snapshot_frequency: int = 0
    stop_on_terminal: bool = True

    # Video recording
    save_video: bool = True
    video_fps: int = 30
    play_video: bool = False
    save_fpv: bool = True


@dataclass
class HabitatStepResult:
    """Result of a single exploration step in Habitat."""

    step: int
    agent_actions: Dict[str, ActionChoice]
    action_results: Dict[str, Any]
    surprises: List[SurpriseRecord]
    observations: Dict[str, Any]
    is_terminal: bool = False


class HabitatWorldAdapter:
    """
    Adapts Habitat's WorldGraph to the interface expected by EMTOM.
    """

    def __init__(self, env_interface: "EnvironmentInterface", agent_uid: int = 0):
        self.env = env_interface
        self.agent_uid = agent_uid

    @property
    def world_graph(self):
        """Get the world graph for the agent."""
        return self.env.world_graph[self.agent_uid]

    @property
    def full_world_graph(self):
        """Get the fully observable world graph."""
        return self.env.full_world_graph

    def get_all_objects(self) -> List[Any]:
        """Get all objects from the world graph."""
        return self.full_world_graph.get_all_objects()

    def get_all_furniture(self) -> List[Any]:
        """Get all furniture (receptacles) from the world graph."""
        return self.full_world_graph.get_all_furnitures()

    def get_all_rooms(self) -> List[Any]:
        """Get all rooms from the world graph."""
        return self.full_world_graph.get_all_rooms()

    def get_interactable_entities(self) -> List[Dict[str, Any]]:
        """Get all entities that can be interacted with."""
        entities = []

        # Get furniture (can be opened/closed)
        for furniture in self.get_all_furniture():
            entity_info = {
                "id": getattr(furniture, "sim_handle", furniture.name),
                "name": furniture.name,
                "type": "furniture",
                "states": self._get_entity_states(furniture),
                "is_articulated": furniture.is_articulated() if hasattr(furniture, "is_articulated") else False,
                "properties": getattr(furniture, "properties", {}),
            }
            entities.append(entity_info)

        # Get objects (can be picked up)
        for obj in self.get_all_objects():
            entity_info = {
                "id": getattr(obj, "sim_handle", obj.name),
                "name": obj.name,
                "type": "object",
                "states": self._get_entity_states(obj),
                "is_articulated": False,
                "properties": getattr(obj, "properties", {}),
            }
            entities.append(entity_info)

        return entities

    def _get_entity_states(self, entity: Any) -> Dict[str, Any]:
        """Extract state properties from an entity."""
        states = {}
        props = getattr(entity, "properties", {})

        state_keys = [
            "is_open", "is_closed", "is_on", "is_off",
            "is_powered_on", "is_powered_off", "is_filled", "is_clean",
        ]

        for key in state_keys:
            if key in props:
                states[key] = props[key]

        if "states" in props:
            states.update(props["states"])

        return states

    def get_room_ids(self) -> List[str]:
        """Get list of room names."""
        return [room.name for room in self.get_all_rooms()]

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get the room the agent is currently in."""
        from habitat_llm.world_model.entity import Room

        try:
            agent_name = agent_id if "agent" in agent_id else f"agent_{agent_id}"
            agent_node = self.full_world_graph.get_node_from_name(agent_name)
            if agent_node:
                neighbors = self.full_world_graph.get_neighbors_of_type(agent_node, Room)
                if neighbors:
                    return neighbors[0].name
        except Exception:
            pass
        return None

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find an entity by name."""
        for entity in self.get_interactable_entities():
            if entity["name"] == name or entity["id"] == name:
                return entity
        return None


class HabitatExplorer:
    """
    Exploration loop using Habitat simulator backend with GameStateManager.
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        game_manager: "GameStateManager",
        curiosity_model: CuriosityModel,
        surprise_detector: SurpriseDetector,
        agent: Optional["Agent"] = None,
        config: Optional[HabitatExplorationConfig] = None,
    ):
        """
        Initialize the Habitat explorer.

        Args:
            env_interface: Habitat EnvironmentInterface
            game_manager: GameStateManager for mechanics
            curiosity_model: LLM-based model for action selection
            surprise_detector: LLM-based model for surprise detection
            agent: Partnr Agent with tools
            config: Exploration configuration
        """
        self.env = env_interface
        self.game_manager = game_manager
        self.curiosity = curiosity_model
        self.surprise_detector = surprise_detector
        self.agent = agent
        self.config = config or HabitatExplorationConfig()

        # World adapter
        self.world_adapter = HabitatWorldAdapter(env_interface, agent_uid=0)

        # Trajectory logging
        self.logger = TrajectoryLogger(
            output_dir=self.config.log_path,
            snapshot_frequency=self.config.snapshot_frequency,
        )

        # Video recording
        self._dvu = None
        self._fpv_recorder = None
        self._setup_video_recording()

        # State
        self.step_count = 0
        self.surprise_moments: List[SurpriseRecord] = []
        self._is_running = False

        # Track current skill execution
        self._current_skill_steps = 0
        self._max_skill_steps = 500
        self._episode_done = False

        # Cache tool descriptions from agent
        self._tool_descriptions: Optional[str] = None
        if agent and hasattr(agent, 'tool_descriptions'):
            self._tool_descriptions = agent.tool_descriptions
            self.curiosity.set_tool_descriptions(self._tool_descriptions)

    def _setup_video_recording(self) -> None:
        """Initialize video recording utilities."""
        if not self.config.save_video:
            return

        try:
            from habitat_llm.examples.example_utils import (
                DebugVideoUtil,
                FirstPersonVideoRecorder,
            )

            os.makedirs(self.config.log_path, exist_ok=True)

            self._dvu = DebugVideoUtil(
                self.env,
                self.config.log_path,
                unique_postfix=True,
            )

            if self.config.save_fpv:
                try:
                    self._fpv_recorder = FirstPersonVideoRecorder(
                        self.env,
                        output_dir=self.config.log_path,
                        fps=self.config.video_fps,
                    )
                except Exception as e:
                    print(f"[HabitatExplorer] FPV recorder init failed: {e}")

        except ImportError as e:
            print(f"[HabitatExplorer] Video utils not available: {e}")

    def run(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full exploration loop in Habitat."""
        # Get active mechanics from game manager
        state = self.game_manager.get_state()
        mechanic_names = state.active_mechanics

        self.logger.start_episode(
            agent_ids=self.config.agent_ids,
            mechanics_active=mechanic_names,
            metadata=metadata,
        )

        self._is_running = True
        self.step_count = 0

        # Clear video buffers
        if self._dvu:
            self._dvu.frames.clear()
        if self._fpv_recorder:
            self._fpv_recorder._frames = {}

        # Log mechanic bindings
        self._log_mechanic_bindings()

        # Log scene info
        self._log_scene_info()

        # Record initial frame
        obs = self.env.get_observations()
        self._record_frame(obs, {})

        # Main exploration loop
        while self._is_running and self.step_count < self.config.max_steps and not self._episode_done:
            step_result = self._run_step()

            if step_result.is_terminal and self.config.stop_on_terminal:
                break

            if self._episode_done:
                print("\n[HabitatExplorer] Episode ended early - stopping exploration")
                break

            self.step_count += 1

        # Save videos
        video_paths = self._save_videos()

        # Finalize episode
        episode_data = self.logger.finalize_episode()

        if video_paths:
            episode_data["video_paths"] = video_paths

        return episode_data

    def _log_mechanic_bindings(self) -> None:
        """Log active mechanic bindings."""
        debug_info = self.game_manager.get_debug_info()
        active = debug_info.get("active_mechanics", [])
        self.logger.log_message(f"Active mechanics: {active}")

        # Log specific bindings
        if debug_info.get("inverse_objects"):
            self.logger.log_message(f"Inverse state targets: {list(debug_info['inverse_objects'])}")
        if debug_info.get("remote_mappings"):
            self.logger.log_message(f"Remote control mappings: {debug_info['remote_mappings']}")

    def _log_scene_info(self) -> None:
        """Log information about the current scene."""
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]
        rooms = self.world_adapter.get_room_ids()

        furniture_names = [f["name"] for f in furniture]
        object_names = [o["name"] for o in objects]
        articulated_names = [f["name"] for f in furniture if f.get("is_articulated")]

        self.logger.log_message(f"Scene has {len(rooms)} rooms: {rooms}")
        self.logger.log_message(f"Scene has {len(furniture)} furniture items")
        self.logger.log_message(f"Scene has {len(objects)} objects")

        if articulated_names:
            self.logger.log_message(f"Articulated furniture: {articulated_names[:10]}...")

        self.logger.set_scene_inventory(
            rooms=rooms,
            furniture=furniture_names,
            objects=object_names,
            articulated_furniture=articulated_names,
        )

    def _run_step(self) -> HabitatStepResult:
        """Execute a single exploration step."""
        agent_actions: Dict[str, ActionChoice] = {}
        action_results: Dict[str, Any] = {}
        step_surprises: List[SurpriseRecord] = []

        print(f"\n{'='*60}")
        print(f"Step {self.step_count + 1}/{self.config.max_steps}")
        print(f"{'='*60}")

        obs = self.env.get_observations()

        for agent_id in self.config.agent_ids:
            agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

            # Build world description
            world_text = self._build_world_description(agent_id)
            available_actions = self._get_available_actions(agent_id)
            recent_history = self.logger.get_recent_actions(agent_id, n=5)

            # Select action via curiosity model
            action_choice = self.curiosity.select_action(
                agent_id=agent_id,
                world_description=world_text,
                available_actions=available_actions,
                exploration_history=recent_history,
                tool_descriptions=self._tool_descriptions,
            )
            agent_actions[agent_id] = action_choice

            print(f"Thought: {action_choice.reasoning}")
            print(f"Agent_{agent_uid}_Action: {action_choice.action}[{action_choice.target or ''}]")
            print("Assigned!")

            if action_choice.action == "Done":
                # Don't allow Done during exploration - re-prompt with a hint
                print(f"Agent_{agent_uid}_Observation: Keep exploring! There's more to discover.")
                self.curiosity.add_observation(agent_id, "Keep exploring! There's more to discover.")
                action_results[agent_id] = {
                    "success": True,
                    "observation": "Keep exploring! There's more to discover.",
                }
                continue

            # Execute action
            result = self._execute_action(agent_id, action_choice)
            action_results[agent_id] = result

            obs_text = result.get("observation", "Successful execution!")
            print(f"Agent_{agent_uid}_Observation: {obs_text}")

            self.curiosity.add_observation(agent_id, obs_text)

            # Check for surprise from LLM reasoning OR from mechanic triggers
            surprise_text = action_choice.surprise or result.get("surprise")
            if surprise_text:
                print(f"\n*** SURPRISE DETECTED ***")
                print(f"    {surprise_text}")

                surprise_record = SurpriseRecord(
                    step=self.step_count,
                    agent_id=agent_id,
                    action=action_choice.action,
                    target=action_choice.target or "",
                    observation=obs_text,
                    surprise_level=4 if result.get("surprise") else 3,  # Higher level for mechanic surprises
                    explanation=surprise_text,
                )
                step_surprises.append(surprise_record)
                self.surprise_moments.append(surprise_record)

        # Record frame
        obs = self.env.get_observations()
        actions_for_video = {}
        for i, (aid, ac) in enumerate(agent_actions.items()):
            actions_for_video[i] = (ac.action, ac.target or "")
        self._record_frame(obs, actions_for_video)

        # Get current inventory for logging
        state = self.game_manager.get_state()
        world_objects = getattr(state, 'world_objects', {})
        inventory_log = {}
        for aid in agent_actions.keys():
            agent_inv = state.agent_inventory.get(aid, [])
            # Resolve item names for readability
            inv_names = []
            for item_id in agent_inv:
                if item_id in world_objects:
                    inv_names.append(world_objects[item_id].get("name", item_id))
                else:
                    inv_names.append(item_id)
            inventory_log[aid] = inv_names if inv_names else []

        # Log step (surprises are tracked in the surprises array, not in agent_actions)
        self.logger.log_step(
            step=self.step_count,
            agent_actions={
                aid: {
                    "action": ac.action,
                    "target": ac.target,
                    "reasoning": ac.reasoning,
                }
                for aid, ac in agent_actions.items()
            },
            effects=[],
            observations={
                aid: r.get("observation", "")
                for aid, r in action_results.items()
            },
            surprises=step_surprises,
            inventory=inventory_log,
        )

        return HabitatStepResult(
            step=self.step_count,
            agent_actions=agent_actions,
            action_results=action_results,
            surprises=step_surprises,
            observations=obs,
        )

    def _build_world_description(self, agent_id: str) -> str:
        """Build a text description of the world from Habitat state."""
        lines = []

        location = self.world_adapter.get_agent_location(agent_id)
        if location:
            lines.append(f"You are in {location}.")

        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]

        # Include spawned world objects (like the key on a table)
        state = self.game_manager.get_state()
        world_objects = getattr(state, 'world_objects', {})
        virtual_objects = []
        for obj_id, obj_info in world_objects.items():
            # Skip if already picked up
            if obj_id in state.agent_inventory.get(agent_id, []):
                continue
            obj_name = obj_info.get("name", obj_id)
            obj_location = obj_info.get("location", "somewhere")
            virtual_objects.append({"name": obj_name, "location": obj_location})
            objects.append({"name": obj_name, "location": obj_location})

        if furniture:
            furniture_names = [f["name"] for f in furniture[:10]]
            lines.append(f"Furniture: {', '.join(furniture_names)}")

        if objects:
            # Show objects with their locations for spawned items
            object_descriptions = []
            for o in objects[:10]:
                name = o["name"] if isinstance(o, dict) else o
                loc = o.get("location") if isinstance(o, dict) else None
                if loc:
                    object_descriptions.append(f"{name} (on {loc})")
                else:
                    object_descriptions.append(name)
            lines.append(f"Objects: {', '.join(object_descriptions)}")

        # Highlight special items on furniture that can be picked up directly
        if virtual_objects:
            for vo in virtual_objects:
                lines.append(f"\033[93m*** You notice a {vo['name']} sitting on {vo['location']}. You can Pick[{vo['name']}] to grab it. ***\033[0m")

        rooms = self.world_adapter.get_room_ids()
        if rooms:
            lines.append(f"Rooms you can go to: {', '.join(rooms)}")

        # Show agent inventory
        inventory = state.agent_inventory.get(agent_id, [])
        if inventory:
            # Resolve names for inventory items
            inv_names = []
            for item_id in inventory:
                if item_id in world_objects:
                    inv_names.append(world_objects[item_id].get("name", item_id))
                else:
                    inv_names.append(item_id)
            lines.append(f"\033[92mYour inventory: {', '.join(inv_names)}\033[0m")

        return "\n".join(lines)

    def _get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get available actions based on Habitat environment."""
        actions = []

        rooms = self.world_adapter.get_room_ids()
        entities = self.world_adapter.get_interactable_entities()

        # Include spawned world objects (like the key on a table)
        state = self.game_manager.get_state()
        world_objects = getattr(state, 'world_objects', {})
        for obj_id, obj_info in world_objects.items():
            # Skip if already picked up
            if obj_id in state.agent_inventory.get(agent_id, []):
                continue
            entities.append({
                "id": obj_id,
                "name": obj_info.get("name", obj_id),
                "type": obj_info.get("type", "object"),
                "location": obj_info.get("location"),
                "is_articulated": False,
                "properties": {},
            })

        furniture = [e for e in entities if e["type"] == "furniture"]
        articulated = [f for f in furniture if f.get("is_articulated")]
        objects = [e for e in entities if e["type"] == "object"]

        # Navigation
        nav_targets = rooms[:5] + [f["name"] for f in furniture[:5]]
        if nav_targets:
            actions.append({
                "name": "Navigate",
                "description": "Navigate to a room or furniture",
                "targets": nav_targets,
            })

        # Explore
        if rooms:
            actions.append({
                "name": "Explore",
                "description": "Search a room by visiting receptacles",
                "targets": rooms[:5],
            })

        # Open/Close
        open_targets = [f["name"] for f in articulated[:10]]
        if open_targets:
            actions.append({
                "name": "Open",
                "description": "Open articulated furniture",
                "targets": open_targets,
            })
            actions.append({
                "name": "Close",
                "description": "Close articulated furniture",
                "targets": open_targets,
            })

        # Pick
        pick_targets = [obj["name"] for obj in objects[:10]]
        if pick_targets:
            actions.append({
                "name": "Pick",
                "description": "Pick up an object",
                "targets": pick_targets,
            })

        # Custom EMTOM actions
        for action_name in EMTOM_ACTIONS.keys():
            targets = [e["name"] for e in entities[:10]]
            actions.append({
                "name": action_name,
                "description": f"Custom EMTOM action: {action_name}",
                "targets": targets,
            })

        return actions

    def _execute_action(self, agent_id: str, action_choice: ActionChoice) -> Dict[str, Any]:
        """Execute an action in the Habitat environment.

        ALL actions are routed through GameStateManager first, which applies mechanics.
        Mechanics are world settings that ALWAYS apply:
        - If mechanic blocks the action (counting_state not reached, conditional_unlock locked),
          we return immediately without executing in Habitat
        - If mechanic transforms the action (inverse_state, remote_control),
          we execute the transformed action in Habitat
        """
        action_name = action_choice.action
        target = action_choice.target or ""

        # Route through GameStateManager to apply mechanics
        # apply_mechanics is called inside apply_action, so we get the handler result from there
        from emtom.mechanics.handlers import apply_mechanics

        # First check mechanics to get transformation info
        mech_result = apply_mechanics(action_name, agent_id, target, self.game_manager.get_state())

        # Now apply the action (this will also call apply_mechanics and update state)
        state, result = self.game_manager.apply_action(action_name, agent_id, target)

        # If mechanic blocked the action, return without executing in Habitat
        if mech_result.blocked:
            return {
                "success": False,
                "observation": mech_result.observation,
                "surprise": mech_result.surprise_trigger,
                "blocked": True,
            }

        # Get the actual action to execute (may be transformed by mechanic)
        actual_action = mech_result.actual_action or action_name
        actual_target = mech_result.actual_target or target
        mechanic_observation = mech_result.observation if mech_result.applies else None
        surprise_trigger = mech_result.surprise_trigger if mech_result.applies else result.surprise_trigger

        # Custom EMTOM action - already handled by GameStateManager
        if action_name in EMTOM_ACTIONS:
            return {
                "success": result.success,
                "observation": mechanic_observation or result.observation,
                "surprise": surprise_trigger,
            }

        # Handle Pick action for virtual objects (e.g., key spawned on table)
        if actual_action == "Pick":
            state = self.game_manager.get_state()
            # Check if target matches a virtual object by name or id
            for obj_id, obj_info in state.world_objects.items():
                obj_name = obj_info.get("name", obj_id)
                if actual_target == obj_name or actual_target == obj_id:
                    # Check if already picked up
                    if obj_id in state.agent_inventory.get(agent_id, []):
                        return {
                            "success": False,
                            "observation": f"You already have the {obj_name}.",
                        }
                    # Add to inventory
                    import copy
                    new_state = copy.copy(state)
                    new_state.agent_inventory = copy.copy(state.agent_inventory)
                    if agent_id not in new_state.agent_inventory:
                        new_state.agent_inventory[agent_id] = []
                    else:
                        new_state.agent_inventory[agent_id] = list(new_state.agent_inventory[agent_id])
                    new_state.agent_inventory[agent_id].append(obj_id)
                    self.game_manager.set_state(new_state)

                    location = obj_info.get("location", "the table")
                    return {
                        "success": True,
                        "observation": f"\033[92m✓ You pick up the {obj_name} from {location}. It's now in your inventory.\033[0m",
                    }

        # Partnr tools - execute in Habitat
        if self.agent is None:
            return {
                "success": False,
                "observation": f"No agent configured. Cannot execute {action_name}[{target}].",
            }

        if actual_action and actual_action not in self.agent.tools:
            return {
                "success": False,
                "observation": f"Tool '{actual_action}' not available.",
            }

        obs = self.env.get_observations()

        low_level_action, response = self.agent.process_high_level_action(
            actual_action, actual_target, obs
        )

        if low_level_action is None:
            obs_text = response
        else:
            # Execute motor skill
            skill_steps = 0
            tool = self.agent.tools[actual_action]

            while skill_steps < self._max_skill_steps:
                try:
                    raw_obs, reward, done, info = self.env.step({0: low_level_action})
                except AssertionError as e:
                    # Episode ended - handle gracefully
                    if "Episode over" in str(e):
                        obs_text = f"Episode ended during {actual_action}[{actual_target}]"
                        self._episode_done = True
                        break
                    raise
                parsed_obs = self.env.parse_observations(raw_obs)
                self._record_frame(parsed_obs, {0: (actual_action, actual_target)})
                skill_steps += 1

                # Check if episode ended
                if done:
                    self._episode_done = True
                    break

                if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                    is_done = tool.skill._is_skill_done(
                        raw_obs, None, None, torch.ones(1, 1), 0
                    )
                    if is_done:
                        break

                low_level_action, response = self.agent.process_high_level_action(
                    actual_action, actual_target, raw_obs
                )

                if low_level_action is None:
                    break

            obs_text = response or f"Executed {actual_action}[{actual_target}]"

        # Use mechanic observation if one was generated
        if mechanic_observation:
            obs_text = mechanic_observation

        return {
            "success": True,
            "observation": obs_text,
            "surprise": surprise_trigger,
        }

    def _record_frame(self, obs: Dict[str, Any], actions: Dict[int, Any]) -> None:
        """Record a video frame."""
        if self._dvu:
            try:
                self._dvu._store_for_video(obs, actions, popup_images={})
            except Exception:
                pass

        if self._fpv_recorder:
            try:
                self._fpv_recorder.record_frame(obs)
            except Exception:
                pass

    def _save_videos(self) -> Dict[str, str]:
        """Save recorded videos and return paths."""
        video_paths = {}

        if self._dvu and self._dvu.frames:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._dvu._make_video(play=False, postfix=f"exploration_{timestamp}")
                # Video is saved to {output_dir}/videos/video-exploration_{timestamp}-{ms}.mp4
                video_paths["third_person"] = f"{self.config.log_path}/videos/"
            except Exception as e:
                print(f"[HabitatExplorer] Failed to save third-person video: {e}")

        if self._fpv_recorder and self._fpv_recorder._frames:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fpv_paths = self._fpv_recorder.save(postfix=timestamp)
                video_paths.update(fpv_paths)
            except Exception as e:
                print(f"[HabitatExplorer] Failed to save FPV video: {e}")

        return video_paths
