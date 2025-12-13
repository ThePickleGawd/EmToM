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
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import torch

from emtom.core.mechanic import (
    ActionResult,
    Effect,
    Mechanic,
    SceneAwareMechanic,
    create_default_effect,
)
from emtom.exploration.curiosity import ActionChoice, CuriosityModel
from emtom.exploration.surprise_detector import SurpriseAssessment, SurpriseDetector
from emtom.exploration.trajectory_logger import SurpriseRecord, TrajectoryLogger
from emtom.actions.custom_actions import EMTOMActionExecutor, EMTOM_ACTIONS

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.world_model import WorldGraph, Furniture, Object
    from habitat_llm.agent import Agent


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

    # Navigation
    max_nav_distance: float = 5.0  # Max distance for oracle nav


@dataclass
class HabitatStepResult:
    """Result of a single exploration step in Habitat."""

    step: int
    agent_actions: Dict[str, ActionChoice]
    action_results: Dict[str, ActionResult]
    surprises: List[SurpriseRecord]
    observations: Dict[str, Any]
    is_terminal: bool = False


class HabitatWorldAdapter:
    """
    Adapts Habitat's WorldGraph to the interface expected by mechanics.

    This allows scene-aware mechanics to bind to objects discovered
    from the actual Habitat scene rather than manually created entities.
    """

    def __init__(self, env_interface: "EnvironmentInterface", agent_uid: int = 0):
        self.env = env_interface
        self.agent_uid = agent_uid

    @property
    def world_graph(self) -> "WorldGraph":
        """Get the world graph for the agent."""
        return self.env.world_graph[self.agent_uid]

    @property
    def full_world_graph(self) -> "WorldGraph":
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
        """
        Get all entities that can be interacted with.

        Returns list of dicts with:
        - id: sim handle or name
        - name: display name
        - type: entity type (object, furniture, etc)
        - states: dict of current states
        - is_articulated: whether it can be opened/closed
        """
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

        # Get objects (can be picked up, may have states)
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

        # Check for standard binary states
        state_keys = [
            "is_open", "is_closed",
            "is_on", "is_off",
            "is_powered_on", "is_powered_off",
            "is_filled", "is_clean",
        ]

        for key in state_keys:
            if key in props:
                states[key] = props[key]

        # Also check nested states dict
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
    Exploration loop using Habitat simulator backend.

    This replaces TextWorldState with EnvironmentInterface, ensuring
    the exploration uses the same action space and environment as
    the benchmark evaluation.

    Uses partnr's built-in tools for actions:
    - navigate: Go to rooms/furniture/objects
    - open/close: Interact with articulated furniture
    - pick/place: Manipulate objects
    - find_*: Perception tools for exploration
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        mechanics: List[Mechanic],
        curiosity_model: CuriosityModel,
        surprise_detector: SurpriseDetector,
        agent: Optional["Agent"] = None,
        config: Optional[HabitatExplorationConfig] = None,
    ):
        """
        Initialize the Habitat explorer.

        Args:
            env_interface: Habitat EnvironmentInterface
            mechanics: List of mechanics to apply
            curiosity_model: LLM-based model for action selection
            surprise_detector: LLM-based model for surprise detection
            agent: Partnr Agent with tools (if None, tools accessed directly from env)
            config: Exploration configuration
        """
        self.env = env_interface
        self.mechanics = mechanics
        self.curiosity = curiosity_model
        self.surprise_detector = surprise_detector
        self.agent = agent
        self.config = config or HabitatExplorationConfig()

        # Custom EMTOM action executor
        self.custom_action_executor = EMTOMActionExecutor(env_interface, mechanics)

        # World adapter for mechanics
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
        self._active_mechanics: List[Mechanic] = []

        # Track current skill execution
        self._current_skill_steps = 0
        self._max_skill_steps = 500  # Max steps per skill execution

        # Cache tool descriptions from agent (for curiosity model)
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
        """
        Run the full exploration loop in Habitat.

        Args:
            metadata: Additional metadata for logs

        Returns:
            Complete episode data with statistics and video paths
        """
        # Start episode logging
        mechanic_names = [m.name for m in self.mechanics]
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

        # Bind mechanics to scene
        self._bind_mechanics_to_scene()

        # Log scene info
        self._log_scene_info()

        # Record initial frame
        obs = self.env.get_observations()
        self._record_frame(obs, {})

        # Main exploration loop
        while self._is_running and self.step_count < self.config.max_steps:
            step_result = self._run_step()

            if step_result.is_terminal and self.config.stop_on_terminal:
                break

            self.step_count += 1

        # Save videos
        video_paths = self._save_videos()

        # Finalize episode
        episode_data = self.logger.finalize_episode()

        # Add video paths to episode data
        if video_paths:
            episode_data["video_paths"] = video_paths

        return episode_data

    def _bind_mechanics_to_scene(self) -> None:
        """Bind scene-aware mechanics to actual scene objects."""
        self._active_mechanics = []

        for mechanic in self.mechanics:
            mechanic.reset()

            if isinstance(mechanic, SceneAwareMechanic):
                # Create a mock world state from the world adapter
                # that provides the interface mechanics expect
                mock_world = HabitatMechanicWorldState(self.world_adapter)

                if mechanic.bind_to_scene(mock_world):
                    self._active_mechanics.append(mechanic)
                    debug_state = mechanic.get_hidden_state_for_debug()
                    self.logger.log_message(
                        f"Mechanic '{mechanic.name}' bound to: {debug_state.get('bound_targets', [])}"
                    )
                else:
                    self.logger.log_message(
                        f"Mechanic '{mechanic.name}' - no suitable objects found"
                    )
            else:
                self._active_mechanics.append(mechanic)

        active_names = [m.name for m in self._active_mechanics]
        self.logger.log_message(f"Active mechanics: {active_names}")

    def _log_scene_info(self) -> None:
        """Log information about the current scene and save complete inventory."""
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]
        rooms = self.world_adapter.get_room_ids()

        # Extract names
        furniture_names = [f["name"] for f in furniture]
        object_names = [o["name"] for o in objects]
        articulated_names = [f["name"] for f in furniture if f.get("is_articulated")]

        self.logger.log_message(f"Scene has {len(rooms)} rooms: {rooms}")
        self.logger.log_message(f"Scene has {len(furniture)} furniture items")
        self.logger.log_message(f"Scene has {len(objects)} objects")

        # Log articulated furniture
        if articulated_names:
            self.logger.log_message(f"Articulated furniture: {articulated_names[:10]}...")

        # Save complete scene inventory to trajectory
        # This is critical for task generation to only use real objects
        self.logger.set_scene_inventory(
            rooms=rooms,
            furniture=furniture_names,
            objects=object_names,
            articulated_furniture=articulated_names,
        )

    def _run_step(self) -> HabitatStepResult:
        """Execute a single exploration step."""
        agent_actions: Dict[str, ActionChoice] = {}
        action_results: Dict[str, ActionResult] = {}
        step_surprises: List[SurpriseRecord] = []

        # Print step header
        print(f"\n{'='*60}")
        print(f"Step {self.step_count + 1}/{self.config.max_steps}")
        print(f"{'='*60}")

        # Get current observations
        obs = self.env.get_observations()

        for agent_id in self.config.agent_ids:
            # Extract agent UID for ReACT format (e.g., "agent_0" -> "0")
            agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

            # Build world description from Habitat state
            world_text = self._build_world_description(agent_id)
            available_actions = self._get_available_actions(agent_id)
            recent_history = self.logger.get_recent_actions(agent_id, n=5)

            # Select action via curiosity model (with tool descriptions from agent)
            action_choice = self.curiosity.select_action(
                agent_id=agent_id,
                world_description=world_text,
                available_actions=available_actions,
                exploration_history=recent_history,
                tool_descriptions=self._tool_descriptions,
            )
            agent_actions[agent_id] = action_choice

            # Print in ReACT format (matching benchmark)
            print(f"Thought: {action_choice.reasoning}")
            print(f"Agent_{agent_uid}_Action: {action_choice.action}[{action_choice.target or ''}]")
            print("Assigned!")

            # Check if agent wants to end exploration
            if action_choice.action == "Done":
                print(f"Agent_{agent_uid}_Observation: Exploration complete.")
                self._is_running = False
                action_results[agent_id] = ActionResult(
                    success=True,
                    observations={agent_id: "Exploration complete."},
                )
                continue

            # Execute action in Habitat
            result = self._execute_action(agent_id, action_choice)
            action_results[agent_id] = result

            # Get observation text
            obs_text = result.observations.get(agent_id, "Successful execution!")

            # Print observation in ReACT format
            print(f"Agent_{agent_uid}_Observation: {obs_text}")

            # Add observation to curiosity model's conversation
            self.curiosity.add_observation(agent_id, obs_text)

            # Check if the LLM detected a surprise in its reasoning
            if action_choice.surprise:
                print(f"\n*** SURPRISE DETECTED ***")
                print(f"    {action_choice.surprise}")
                print()

                # Record the surprise
                surprise_record = SurpriseRecord(
                    step=self.step_count,
                    agent_id=agent_id,
                    action=action_choice.action,
                    target=action_choice.target or "",
                    observation=obs_text,
                    surprise_level=3,  # Default level for LLM-detected surprises
                    explanation=action_choice.surprise,
                )
                step_surprises.append(surprise_record)
                self.surprise_moments.append(surprise_record)

        # Record frame with actions
        obs = self.env.get_observations()
        actions_for_video = {}
        for i, (aid, ac) in enumerate(agent_actions.items()):
            actions_for_video[i] = (ac.action, ac.target or "")
        self._record_frame(obs, actions_for_video)

        # Log step
        self.logger.log_step(
            step=self.step_count,
            agent_actions={
                aid: {
                    "action": ac.action,
                    "target": ac.target,
                    "reasoning": ac.reasoning,
                    "surprise": ac.surprise,
                }
                for aid, ac in agent_actions.items()
            },
            effects=[e.to_dict() for r in action_results.values() for e in r.effects],
            observations={
                aid: r.observations.get(aid, "")
                for aid, r in action_results.items()
            },
            surprises=step_surprises,
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

        # Current location
        location = self.world_adapter.get_agent_location(agent_id)
        if location:
            lines.append(f"You are in {location}.")

        # Nearby entities
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        objects = [e for e in entities if e["type"] == "object"]

        if furniture:
            furniture_names = [f["name"] for f in furniture[:10]]
            lines.append(f"Furniture: {', '.join(furniture_names)}")

        if objects:
            object_names = [o["name"] for o in objects[:10]]
            lines.append(f"Objects: {', '.join(object_names)}")

        # Available rooms
        rooms = self.world_adapter.get_room_ids()
        if rooms:
            lines.append(f"Rooms you can go to: {', '.join(rooms)}")

        return "\n".join(lines)

    def _get_available_actions(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get available actions based on Habitat environment, partnr tools, and custom EMTOM actions.

        Partnr tools (motor skills):
        - Navigate: OracleNavSkill - go to rooms, furniture, objects
        - Open: OracleOpenSkill - open articulated furniture
        - Close: OracleCloseSkill - close articulated furniture
        - Pick: OraclePickSkill - pick up objects
        - Explore: OracleExploreSkill - explore a room

        Custom EMTOM actions:
        - FlipLights: Toggle lights in a room
        - PressButton: Press buttons/switches
        - PullLever: Pull levers
        - TurnDial: Turn dials/knobs
        - Inspect: Examine objects closely
        - RingBell: Ring bells (signals other agents)
        - CheckStatus: Check device status
        """
        actions = []

        # Get scene info
        rooms = self.world_adapter.get_room_ids()
        current = self.world_adapter.get_agent_location(agent_id)
        entities = self.world_adapter.get_interactable_entities()
        furniture = [e for e in entities if e["type"] == "furniture"]
        articulated = [f for f in furniture if f.get("is_articulated")]
        objects = [e for e in entities if e["type"] == "object"]

        # ===== Partnr Motor Skills =====

        # Navigate - can go to rooms or furniture
        nav_targets = []
        nav_targets.extend(rooms[:5])  # Rooms
        nav_targets.extend([f["name"] for f in furniture[:5]])  # Furniture
        if nav_targets:
            actions.append({
                "name": "Navigate",
                "description": "Navigate to a room or furniture",
                "targets": nav_targets,
            })

        # Explore - explore rooms
        if rooms:
            actions.append({
                "name": "Explore",
                "description": "Search a room by visiting receptacles in it",
                "targets": rooms[:5],
            })

        # Open - articulated furniture
        open_targets = [f["name"] for f in articulated[:10]]
        if open_targets:
            actions.append({
                "name": "Open",
                "description": "Open articulated furniture (cabinets, drawers, etc)",
                "targets": open_targets,
            })

        # Close - articulated furniture
        if open_targets:
            actions.append({
                "name": "Close",
                "description": "Close articulated furniture",
                "targets": open_targets,
            })

        # Pick - objects
        pick_targets = [obj["name"] for obj in objects[:10]]
        if pick_targets:
            actions.append({
                "name": "Pick",
                "description": "Pick up an object",
                "targets": pick_targets,
            })

        # ===== Custom EMTOM Actions =====
        # Build world state dict for custom action target discovery
        world_state = self._build_world_state_for_actions(agent_id, rooms, entities)

        # Get custom actions with their available targets
        custom_actions = self.custom_action_executor.get_available_actions(world_state)
        actions.extend(custom_actions)

        return actions

    def _build_world_state_for_actions(
        self,
        agent_id: str,
        rooms: List[str],
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build world state dict for custom EMTOM actions."""
        return {
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
            "room_lights": {},  # TODO: Track light states if needed
            "lever_states": {},  # TODO: Track lever states if needed
            "dial_values": {},  # TODO: Track dial values if needed
            "other_agents": [
                aid for aid in self.config.agent_ids if aid != agent_id
            ],
        }

    def _execute_action(
        self, agent_id: str, action_choice: ActionChoice
    ) -> ActionResult:
        """
        Execute an action in the Habitat environment.

        Handles two types of actions:
        1. Partnr tools (Navigate, Open, Close, Pick, Explore) - executed via agent
        2. Custom EMTOM actions (FlipLights, PressButton, etc.) - executed via custom executor

        Returns:
            ActionResult with observations and effects
        """
        action_name = action_choice.action
        target = action_choice.target or ""

        # Check if this is a custom EMTOM action
        if action_name in EMTOM_ACTIONS:
            return self._execute_custom_action(agent_id, action_name, target)

        # Otherwise, execute via partnr tools
        if self.agent is None:
            return self._execute_action_direct(agent_id, action_name, target)

        # Check if the agent has this tool
        if action_name not in self.agent.tools:
            return ActionResult(
                success=False,
                observations={agent_id: f"Tool '{action_name}' not available."},
            )

        # Get current observations
        obs = self.env.get_observations()

        # Execute the tool to get low-level action
        low_level_action, response = self.agent.process_high_level_action(
            action_name, target, obs
        )

        # If it's a perception tool (returns None), just return the response
        if low_level_action is None:
            return ActionResult(
                success=True,
                observations={agent_id: response},
            )

        # Execute motor skill until done
        skill_steps = 0
        max_steps = self._max_skill_steps
        tool = self.agent.tools[action_name]

        while skill_steps < max_steps:
            # Step the environment with the low-level action
            raw_obs, reward, done, info = self.env.step({0: low_level_action})

            # Parse observations for video recording (converts to format with third_rgb, etc.)
            parsed_obs = self.env.parse_observations(raw_obs)

            # Record frame for video using parsed observations
            self._record_frame(parsed_obs, {0: (action_name, target)})

            skill_steps += 1

            # Check if skill is done
            if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                # Check skill termination
                is_done = tool.skill._is_skill_done(
                    raw_obs, None, None, torch.ones(1, 1), 0
                )
                if is_done:
                    break

            # Get next action using raw observations (what the skill expects)
            low_level_action, response = self.agent.process_high_level_action(
                action_name, target, raw_obs
            )

            if low_level_action is None:
                # Skill completed or failed
                break

        # Build result
        final_response = response or f"Executed {action_name}[{target}]"

        # Check if mechanic modifies the result
        result = self._check_mechanics(agent_id, action_name, target, final_response)
        if result:
            return result

        return ActionResult(
            success=True,
            observations={agent_id: final_response},
            effects=[Effect(
                target=target,
                property_changed="last_action",
                old_value=None,
                new_value=action_name,
                visible_to={agent_id},
            )],
        )

    def _execute_action_direct(
        self, agent_id: str, action_name: str, target: str
    ) -> ActionResult:
        """
        Execute action directly without Agent (fallback mode).
        Uses env_interface methods directly when no Agent is available.
        """

        # For now, just log the action - actual implementation would
        # need to instantiate tools directly
        return ActionResult(
            success=False,
            observations={agent_id: f"No agent configured. Cannot execute {action_name}[{target}]."},
        )

    def _execute_custom_action(
        self, agent_id: str, action_name: str, target: str
    ) -> ActionResult:
        """
        Execute a custom EMTOM action.

        Custom actions are simulated interactions (Hide, Inspect, WriteMessage, etc.)
        that can be affected by mechanics to produce surprising outcomes.
        """

        # Record frame with the custom action for video
        obs = self.env.get_observations()
        parsed_obs = self.env.parse_observations(obs)
        # Use agent index 0 for video display
        agent_idx = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
        self._record_frame(parsed_obs, {agent_idx: (action_name, target or "")})

        # Build world state for the action
        entities = self.world_adapter.get_interactable_entities()
        rooms = self.world_adapter.get_room_ids()
        world_state = self._build_world_state_for_actions(agent_id, rooms, entities)

        # Execute via custom action executor (applies mechanics automatically)
        from emtom.actions.custom_actions import ActionResult as CustomActionResult
        custom_result = self.custom_action_executor.execute(
            action_name, agent_id, target, world_state
        )

        # Convert custom ActionResult to mechanic ActionResult
        observation = custom_result.observation
        effects = []

        if custom_result.effect:
            effects.append(Effect(
                target=target or "world",
                property_changed=action_name.lower(),
                old_value=None,
                new_value=custom_result.effect,
                visible_to={agent_id},
            ))

        # Build observations dict
        observations = {agent_id: observation}
        if custom_result.other_observations:
            observations.update(custom_result.other_observations)

        # Build surprise triggers
        surprise_triggers = {}
        if custom_result.surprise_trigger:
            surprise_triggers[agent_id] = custom_result.surprise_trigger

        return ActionResult(
            success=custom_result.success,
            observations=observations,
            effects=effects,
            surprise_triggers=surprise_triggers,
        )

    def _check_mechanics(
        self, agent_id: str, action_name: str, target: str, base_response: str
    ) -> Optional[ActionResult]:
        """Check if any mechanic should transform this action's result."""
        mock_world = HabitatMechanicWorldState(self.world_adapter)

        for mechanic in self._active_mechanics:
            # Map partnr tool names to mechanic action names
            mechanic_action = action_name.lower()
            if mechanic_action == "navigate":
                continue  # Navigation doesn't have mechanic effects

            if mechanic.applies_to(mechanic_action, target, mock_world):
                intended = create_default_effect(mechanic_action, target, mock_world)
                result = mechanic.transform_effect(
                    mechanic_action, agent_id, target, intended, mock_world
                )
                return result

        return None

    def _record_frame(self, obs: Dict[str, Any], actions: Dict[int, Any]) -> None:
        """Record a video frame."""
        if self._dvu:
            try:
                self._dvu._store_for_video(obs, actions, popup_images={})
            except Exception:
                pass

        if self._fpv_recorder:
            try:
                self._fpv_recorder.record_step(obs)
            except Exception:
                pass

    def _save_videos(self) -> Dict[str, str]:
        """Save recorded videos."""
        video_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        postfix = f"exploration_{timestamp}"

        if self._dvu and self._dvu.frames:
            try:
                self._dvu._make_video(play=self.config.play_video, postfix=postfix)
                video_dir = os.path.join(self.config.log_path, "videos")
                video_paths["third_person"] = os.path.join(video_dir, f"video-{postfix}.mp4")
            except Exception as e:
                print(f"[HabitatExplorer] Failed to save video: {e}")

        if self._fpv_recorder:
            try:
                fpv_paths = self._fpv_recorder.save(postfix=postfix)
                for name, path in fpv_paths.items():
                    video_paths[f"fpv_{name}"] = path
            except Exception:
                pass

        return video_paths

    def stop(self):
        """Stop the exploration."""
        self._is_running = False


class HabitatMechanicWorldState:
    """
    Adapter that provides the world state interface expected by mechanics,
    backed by Habitat's world graph.
    """

    def __init__(self, adapter: HabitatWorldAdapter):
        self.adapter = adapter
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
        self._entities: Dict[str, "HabitatEntity"] = {}
        self._build_entity_cache()

    def _build_entity_cache(self):
        """Build cache of entities for mechanic queries."""
        seen_ids = set()
        for entity_info in self.adapter.get_interactable_entities():
            entity_id = entity_info.get("id", entity_info.get("name", "unknown"))
            # Avoid duplicates (same entity by id and name)
            if entity_id not in seen_ids:
                self._entity_cache[entity_id] = entity_info
                self._entities[entity_id] = HabitatEntity(entity_info)
                seen_ids.add(entity_id)
            # Also index by name if different
            name = entity_info.get("name", "")
            if name and name != entity_id and name not in seen_ids:
                self._entity_cache[name] = entity_info
                # Don't duplicate in _entities, just in cache for lookup

    @property
    def entities(self) -> Dict[str, "HabitatEntity"]:
        """Return entities dict for ObjectSelector compatibility."""
        return self._entities

    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Get entity by ID/name - returns a mock Entity object."""
        entity_info = self._entity_cache.get(entity_id)
        if entity_info:
            return HabitatEntity(entity_info)
        return None

    def get_entities_in_location(self, location: str) -> List[Any]:
        """Get all entities - location filtering not implemented for Habitat."""
        return list(self._entities.values())

    def get_room_ids(self) -> List[str]:
        """Get room IDs."""
        return self.adapter.get_room_ids()

    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get agent's current room."""
        return self.adapter.get_agent_location(agent_id)

    def apply_effect(self, effect: Effect) -> None:
        """Apply effect - logged but not actually applied in Habitat."""
        # In exploration, effects are logged but don't modify Habitat state
        # The actual mechanic behavior is simulated for trajectory generation
        pass

    def snapshot(self) -> Dict[str, Any]:
        """Get snapshot of world state."""
        return {
            "entities": list(self._entities.keys()),
            "rooms": self.get_room_ids(),
        }


class HabitatEntity:
    """Mock Entity class that wraps Habitat world graph entity info."""

    def __init__(self, info: Dict[str, Any]):
        self.id = info.get("id", info.get("name", "unknown"))
        self.name = info.get("name", "unknown")
        self.entity_type = info.get("type", "object")
        self.is_articulated_flag = info.get("is_articulated", False)

        # Merge properties and states so ObjectSelector can find binary states
        self.properties = dict(info.get("properties", {}))
        states = info.get("states", {})
        self.properties.update(states)

        # If articulated furniture, add is_open state if not present
        if self.is_articulated_flag and "is_open" not in self.properties:
            self.properties["is_open"] = False

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        self.properties[key] = value

    def is_articulated(self) -> bool:
        """Check if entity is articulated."""
        return self.is_articulated_flag


# Note: HabitatExplorer should be instantiated from within a @hydra.main function
# to ensure proper Hydra context. See emtom/examples/run_habitat_exploration.py
# for the correct usage pattern.
