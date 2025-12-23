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
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

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
        agents: Optional[Dict[int, "Agent"]] = None,
        config: Optional[HabitatExplorationConfig] = None,
    ):
        """
        Initialize the Habitat explorer.

        Args:
            env_interface: Habitat EnvironmentInterface
            game_manager: GameStateManager for mechanics
            curiosity_model: LLM-based model for action selection
            surprise_detector: LLM-based model for surprise detection
            agents: Dict mapping agent UID to Partnr Agent with tools
            config: Exploration configuration
        """
        self.env = env_interface
        self.game_manager = game_manager
        self.curiosity = curiosity_model
        self.surprise_detector = surprise_detector
        self.agents = agents or {}
        self.config = config or HabitatExplorationConfig()

        # World adapters per agent
        self.world_adapters: Dict[int, HabitatWorldAdapter] = {}
        for uid in self.agents.keys():
            self.world_adapters[uid] = HabitatWorldAdapter(env_interface, agent_uid=uid)
        # Default adapter for general use
        self.world_adapter = self.world_adapters.get(0) or HabitatWorldAdapter(env_interface, agent_uid=0)

        # Trajectory logging
        self.logger = TrajectoryLogger(
            output_dir=self.config.log_path,
            snapshot_frequency=self.config.snapshot_frequency,
        )

        # Video recording
        self._dvu = None
        self._fpv_recorder = None
        self._per_agent_recorder = None  # Per-agent third-person recorder
        self._setup_video_recording()

        # State
        self.step_count = 0  # Global step counter (for logging)
        self.agent_step_counts: Dict[str, int] = {}  # Per-agent step counts
        self.surprise_moments: List[SurpriseRecord] = []
        self._is_running = False

        # Track current skill execution
        self._current_skill_steps = 0
        self._max_skill_steps = 1000
        self._episode_done = False

        # Cache tool descriptions from first agent
        self._tool_descriptions: Optional[str] = None
        first_agent = next(iter(self.agents.values()), None) if self.agents else None
        if first_agent and hasattr(first_agent, 'tool_descriptions'):
            self._tool_descriptions = first_agent.tool_descriptions
            self.curiosity.set_tool_descriptions(self._tool_descriptions)

        # Pass story context for atmosphere/setting (but exploration is still free-form)
        story_context = self.game_manager.get_story_context()
        if story_context:
            self.curiosity.set_story_context(story_context)

    def _setup_video_recording(self) -> None:
        """Initialize video recording utilities."""
        if not self.config.save_video:
            return

        try:
            from habitat_llm.examples.example_utils import (
                DebugVideoUtil,
                FirstPersonVideoRecorder,
                PerAgentThirdPersonRecorder,
            )

            os.makedirs(self.config.log_path, exist_ok=True)

            # Legacy combined view recorder (kept for backwards compatibility)
            self._dvu = DebugVideoUtil(
                self.env,
                self.config.log_path,
                unique_postfix=True,
            )

            # Override num_agents to match our exploration config
            # This ensures all agents' views are included in the video
            self._dvu.num_agents = len(self.config.agent_ids)

            # NEW: Per-agent third-person recorder for separate videos + stitching
            self._per_agent_recorder = PerAgentThirdPersonRecorder(
                output_dir=self.config.log_path,
                agent_ids=self.config.agent_ids,
                fps=self.config.video_fps,
            )
            print(f"[HabitatExplorer] Per-agent video recording initialized for {len(self.config.agent_ids)} agents")

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
        """Run the full exploration loop in Habitat.

        Uses round-robin execution: each agent takes one step in turn until
        ALL agents have completed max_steps simulation steps each.

        Example with 3 agents and max_steps=50:
        - agent_0 step 1, agent_1 step 1, agent_2 step 1
        - agent_0 step 2, agent_1 step 2, agent_2 step 2
        - ... (continues until each agent has taken 50 steps)
        - Total actions: 3 agents × 50 steps = 150 actions
        """
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

        # Initialize per-agent step counters
        self.agent_step_counts = {agent_id: 0 for agent_id in self.config.agent_ids}

        # Clear video buffers
        if self._dvu:
            self._dvu.frames.clear()
        if self._fpv_recorder:
            self._fpv_recorder._frames = {}
        if self._per_agent_recorder:
            self._per_agent_recorder.clear()

        # Log mechanic bindings
        self._log_mechanic_bindings()

        # Log scene info
        self._log_scene_info()

        # Record initial frame
        obs = self.env.get_observations()
        # Debug: Show available observation keys
        rgb_keys = [k for k in obs.keys() if 'rgb' in k.lower()] if obs else []
        print(f"[HabitatExplorer] Initial observation RGB keys: {rgb_keys}")
        print(f"[HabitatExplorer] DebugVideoUtil num_agents: {self._dvu.num_agents if self._dvu else 'N/A'}")

        # Record initial frame for all agents
        for init_agent_id in self.config.agent_ids:
            initial_recorded = self._record_frame(obs, {}, current_agent_id=init_agent_id)
        if not initial_recorded:
            print(f"[HabitatExplorer] WARNING: Initial frame recording failed!")
        else:
            print(f"[HabitatExplorer] Initial frame recorded successfully for all agents")

        n_agents = len(self.config.agent_ids)
        total_steps = n_agents * self.config.max_steps

        print(f"\n[HabitatExplorer] Round-robin exploration:")
        print(f"  - {n_agents} agents:")
        for agent_id in self.config.agent_ids:
            uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
            color = self._get_agent_color(uid)
            print(f"      {color}{agent_id}\033[0m")
        print(f"  - {self.config.max_steps} steps per agent")
        print(f"  - {total_steps} total simulation steps")

        # Main exploration loop - round-robin until each agent has max_steps
        while self._is_running:
            # Check if all agents have completed their steps
            all_done = all(
                count >= self.config.max_steps
                for count in self.agent_step_counts.values()
            )
            if all_done:
                break

            # Round-robin: each agent takes one step
            for agent_id in self.config.agent_ids:
                if not self._is_running:
                    break

                # Skip if this agent has already completed their steps
                if self.agent_step_counts[agent_id] >= self.config.max_steps:
                    continue

                # Execute single step for this agent
                self._run_single_agent_step(agent_id)

                # Increment counters
                self.agent_step_counts[agent_id] += 1
                self.step_count += 1

                # Frame count logging (every step for first 5, then every 10)
                if self._dvu:
                    frame_count = len(self._dvu.frames) if self._dvu.frames else 0
                    if self.step_count <= 5 or self.step_count % 10 == 0:
                        print(f"[HabitatExplorer] Progress: {self.step_count}/{total_steps} steps, {frame_count} frames recorded")

        # Save videos
        video_paths = self._save_videos()

        # Save prompts from CuriosityModel (similar to DecentralizedEvaluationRunner)
        self._save_exploration_prompts()

        # Finalize episode
        episode_data = self.logger.finalize_episode()

        if video_paths:
            episode_data["video_paths"] = video_paths

        return episode_data

    def _log_mechanic_bindings(self) -> None:
        """Log active mechanic bindings and save to trajectory."""
        debug_info = self.game_manager.get_debug_info()
        active = debug_info.get("active_mechanics", [])
        self.logger.log_message(f"Active mechanics: {active}")

        # Build structured bindings dict for trajectory
        bindings = {}

        # Log specific bindings
        if debug_info.get("inverse_objects"):
            inverse_list = list(debug_info["inverse_objects"])
            self.logger.log_message(f"Inverse state targets: {inverse_list}")
            bindings["inverse_state"] = {"targets": inverse_list}

        if debug_info.get("remote_mappings"):
            remote = debug_info["remote_mappings"]
            self.logger.log_message(f"Remote control mappings: {remote}")
            # Convert {trigger: (target, state)} to list of bindings
            bindings["remote_control"] = [
                {"trigger": trigger, "target": target_info[0], "target_state": target_info[1]}
                for trigger, target_info in remote.items()
            ]

        if debug_info.get("interaction_counts"):
            counts = debug_info["interaction_counts"]
            self.logger.log_message(f"Counting state targets: {counts}")
            bindings["counting_state"] = {"targets": counts}

        if debug_info.get("hidden_items"):
            hidden = debug_info["hidden_items"]
            self.logger.log_message(f"Hidden items: {hidden}")
            bindings["hidden_items"] = hidden

        if debug_info.get("item_definitions"):
            bindings["item_definitions"] = debug_info["item_definitions"]

        if debug_info.get("agent_inventory"):
            bindings["agent_inventory"] = debug_info["agent_inventory"]

        # Save bindings to trajectory (critical for task generation!)
        self.logger.set_mechanic_bindings(bindings)

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

    @staticmethod
    def _get_agent_color(agent_uid: int) -> str:
        """Get ANSI color code for an agent.

        Args:
            agent_uid: Agent UID (0, 1, 2, ...)

        Returns:
            ANSI escape code for the agent's color
        """
        colors = [
            "\033[94m",  # Blue - Agent 0
            "\033[92m",  # Green - Agent 1
            "\033[93m",  # Yellow - Agent 2
            "\033[95m",  # Magenta - Agent 3
            "\033[96m",  # Cyan - Agent 4
            "\033[91m",  # Red - Agent 5
        ]
        return colors[agent_uid % len(colors)]

    def _run_single_agent_step(self, agent_id: str) -> Dict[str, Any]:
        """Execute a single simulation step for one agent.

        This is called in round-robin fashion so each agent takes independent steps.

        Args:
            agent_id: The agent to execute a step for (e.g., "agent_0")

        Returns:
            Dict with action_choice, result, and any surprises
        """
        agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id
        agent_uid_int = int(agent_uid)
        agent_step = self.agent_step_counts.get(agent_id, 0) + 1

        # Get color for this agent
        agent_color = self._get_agent_color(agent_uid_int)
        reset = "\033[0m"

        print(f"\n{agent_color}{'='*60}{reset}")
        print(f"{agent_color}Agent {agent_uid}{reset} - Step {agent_step}/{self.config.max_steps} (Global: {self.step_count + 1})")
        print(f"{agent_color}{'='*60}{reset}")

        # Build world description for this agent
        world_text = self._build_world_description(agent_id)
        recent_history = self.logger.get_recent_actions(agent_id, n=5)

        # Get available actions for Verbalized Sampling
        available_actions = self._get_available_actions(agent_id)

        # Select action via curiosity model (uses agent's own LLM client)
        # If VS is enabled, this will prompt the LLM for a probability distribution
        # and sample from it using the agent's RNG
        action_choice = self.curiosity.select_action(
            agent_id=agent_id,
            world_description=world_text,
            exploration_history=recent_history,
            tool_descriptions=self._tool_descriptions,
            available_actions=available_actions,
        )

        print(f"Thought: {action_choice.reasoning}")
        print(f"{agent_color}Agent_{agent_uid}_Action:{reset} {action_choice.action}[{action_choice.target or ''}]")
        print("Assigned!")

        step_surprises: List[SurpriseRecord] = []

        if action_choice.action == "Done":
            # Don't allow Done during exploration - re-prompt with a hint
            print(f"{agent_color}Agent_{agent_uid}_Observation:{reset} Keep exploring! There's more to discover.")
            self.curiosity.add_observation(agent_id, "Keep exploring! There's more to discover.")
            result = {
                "success": True,
                "observation": "Keep exploring! There's more to discover.",
            }
        else:
            # Execute action
            result = self._execute_action(agent_id, action_choice)

            obs_text = result.get("observation", "Successful execution!")
            print(f"{agent_color}Agent_{agent_uid}_Observation:{reset} {obs_text}")

            self.curiosity.add_observation(agent_id, obs_text)

            # Check for surprise
            action_blocked = result.get("blocked", False)
            mechanic_trigger = result.get("surprise")
            llm_surprise = action_choice.surprise

            if action_blocked or mechanic_trigger or llm_surprise:
                expected_outcome = f"The {action_choice.target} would open normally" if action_choice.action.lower() == "open" else f"Normal {action_choice.action.lower()} behavior"

                surprise_assessment = self.surprise_detector.assess_surprise(
                    agent_id=agent_id,
                    action=action_choice.action,
                    target=action_choice.target,
                    expected=expected_outcome,
                    actual=obs_text,
                    trigger=mechanic_trigger,
                )

                if surprise_assessment.is_surprised:
                    print(f"\n*** SURPRISE DETECTED ***")
                    print(f"    Level: {surprise_assessment.level}/5")
                    print(f"    Explanation: {surprise_assessment.explanation}")
                    if surprise_assessment.hypothesis:
                        print(f"    Hypothesis: {surprise_assessment.hypothesis}")

                    surprise_record = SurpriseRecord(
                        step=self.step_count,
                        agent_id=agent_id,
                        action=action_choice.action,
                        target=action_choice.target or "",
                        observation=obs_text,
                        surprise_level=surprise_assessment.level,
                        explanation=surprise_assessment.explanation,
                    )
                    step_surprises.append(surprise_record)
                    self.surprise_moments.append(surprise_record)

        # Record frame - CRITICAL: This must succeed for every step to appear in video
        obs = self.env.get_observations()
        agent_uid_int = int(agent_uid)

        # Debug: Log observation keys on first few steps
        if self.step_count < 3:
            rgb_keys = [k for k in obs.keys() if 'rgb' in k.lower()] if obs else []
            print(f"[HabitatExplorer] Step {self.step_count + 1} observation RGB keys: {rgb_keys}")

        frame_recorded = self._record_frame(
            obs,
            {agent_uid_int: (action_choice.action, action_choice.target or "")},
            current_agent_id=agent_id,
        )

        # If frame recording failed, try with agent 0's action format as fallback
        if not frame_recorded:
            print(f"[HabitatExplorer] Frame recording failed for step {self.step_count + 1}, trying fallback...")
            try:
                # Try recording with empty actions (just capture the scene)
                obs = self.env.get_observations()
                frame_recorded = self._record_frame(
                    obs,
                    {0: (action_choice.action, action_choice.target or "")},
                    current_agent_id=agent_id,
                )
                if not frame_recorded:
                    print(f"[HabitatExplorer] ERROR: Frame recording failed with fallback")
                    if obs:
                        print(f"  Available keys: {[k for k in obs.keys() if 'rgb' in k.lower()]}")
            except Exception as e:
                print(f"[HabitatExplorer] ERROR: Frame fallback failed: {e}")
                import traceback
                traceback.print_exc()

        # Get current inventory for logging
        state = self.game_manager.get_state()
        world_objects = getattr(state, 'world_objects', {})
        agent_inv = state.agent_inventory.get(agent_id, [])
        inv_names = []
        for item_id in agent_inv:
            if item_id in world_objects:
                inv_names.append(world_objects[item_id].get("name", item_id))
            else:
                inv_names.append(item_id)

        # Log step for this single agent
        self.logger.log_step(
            step=self.step_count,
            agent_actions={
                agent_id: {
                    "action": action_choice.action,
                    "target": action_choice.target,
                    "reasoning": action_choice.reasoning,
                }
            },
            effects=[],
            observations={
                agent_id: result.get("observation", "")
            },
            surprises=step_surprises,
            inventory={agent_id: inv_names if inv_names else []},
        )

        return {
            "action_choice": action_choice,
            "result": result,
            "surprises": step_surprises,
        }

    def _run_step(self) -> HabitatStepResult:
        """Execute a single exploration step for ALL agents (legacy method).

        Note: The new round-robin execution uses _run_single_agent_step instead.
        This method is kept for backward compatibility.
        """
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
            recent_history = self.logger.get_recent_actions(agent_id, n=5)

            # Get available actions for Verbalized Sampling
            available_actions = self._get_available_actions(agent_id)

            # Select action via curiosity model
            action_choice = self.curiosity.select_action(
                agent_id=agent_id,
                world_description=world_text,
                exploration_history=recent_history,
                tool_descriptions=self._tool_descriptions,
                available_actions=available_actions,
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

            # Check for surprise - if action was blocked or had unexpected outcome,
            # use LLM to assess what the surprise actually is
            action_blocked = result.get("blocked", False)
            mechanic_trigger = result.get("surprise")
            llm_surprise = action_choice.surprise

            if action_blocked or mechanic_trigger or llm_surprise:
                # Use LLM to assess what the actual surprise is
                expected_outcome = f"The {action_choice.target} would open normally" if action_choice.action.lower() == "open" else f"Normal {action_choice.action.lower()} behavior"

                surprise_assessment = self.surprise_detector.assess_surprise(
                    agent_id=agent_id,
                    action=action_choice.action,
                    target=action_choice.target,
                    expected=expected_outcome,
                    actual=obs_text,
                    trigger=mechanic_trigger,  # Pass system hint if any
                )

                if surprise_assessment.is_surprised:
                    print(f"\n*** SURPRISE DETECTED ***")
                    print(f"    Level: {surprise_assessment.level}/5")
                    print(f"    Explanation: {surprise_assessment.explanation}")
                    if surprise_assessment.hypothesis:
                        print(f"    Hypothesis: {surprise_assessment.hypothesis}")

                    surprise_record = SurpriseRecord(
                        step=self.step_count,
                        agent_id=agent_id,
                        action=action_choice.action,
                        target=action_choice.target or "",
                        observation=obs_text,
                        surprise_level=surprise_assessment.level,
                        explanation=surprise_assessment.explanation,
                    )
                    step_surprises.append(surprise_record)
                    self.surprise_moments.append(surprise_record)

        # Record frame for each agent (for per-agent video recording)
        obs = self.env.get_observations()
        actions_for_video = {}
        for i, (aid, ac) in enumerate(agent_actions.items()):
            actions_for_video[i] = (ac.action, ac.target or "")
        # Record for each agent separately for per-agent videos
        for aid in agent_actions.keys():
            self._record_frame(obs, actions_for_video, current_agent_id=aid)

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
        """Build a text description of the world from Habitat state.

        Uses per-agent world adapters to ensure each agent gets their own
        perspective of the world (partial observability).

        IMPORTANT: Lists are shuffled using agent-specific random seeds to ensure
        each agent sees a DIFFERENT ordering of items. This prevents all agents
        from picking the same "first" item and encourages independent exploration.
        """
        lines = []

        # Get agent-specific world adapter for partial observability
        agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
        adapter = self.world_adapters.get(agent_uid, self.world_adapter)

        # Create agent-specific random generator to ensure different shuffles per agent
        # The seed combines agent_uid with step_count so ordering changes over time
        agent_random = random.Random(agent_uid * 1000 + self.step_count + hash(agent_id) % 1000)

        location = adapter.get_agent_location(agent_id)
        if location:
            lines.append(f"You are in {location}.")

        entities = adapter.get_interactable_entities()
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

        # Shuffle furniture using AGENT-SPECIFIC random to ensure different orderings
        if furniture:
            furniture_shuffled = furniture.copy()
            agent_random.shuffle(furniture_shuffled)
            furniture_names = [f["name"] for f in furniture_shuffled[:10]]
            lines.append(f"Furniture: {', '.join(furniture_names)}")

        # Shuffle objects using AGENT-SPECIFIC random to ensure different orderings
        if objects:
            objects_shuffled = objects.copy()
            agent_random.shuffle(objects_shuffled)
            # Show objects with their locations for spawned items
            object_descriptions = []
            for o in objects_shuffled[:10]:
                name = o["name"] if isinstance(o, dict) else o
                loc = o.get("location") if isinstance(o, dict) else None
                if loc:
                    object_descriptions.append(f"{name} (on {loc})")
                else:
                    object_descriptions.append(name)
            lines.append(f"Objects: {', '.join(object_descriptions)}")

        # Note: virtual objects (like the key) are already included in the objects list above
        # Don't highlight them specially - let the agent discover them naturally through exploration

        # Shuffle rooms using AGENT-SPECIFIC random to ensure different orderings
        rooms = adapter.get_room_ids()
        if rooms:
            rooms_shuffled = rooms.copy()
            agent_random.shuffle(rooms_shuffled)
            lines.append(f"Rooms you can go to: {', '.join(rooms_shuffled)}")

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
        """Get available actions based on Habitat environment.

        IMPORTANT: Targets are shuffled using agent-specific random to ensure
        each agent sees DIFFERENT action orderings. This prevents all agents
        from converging on the same targets.
        """
        actions = []

        # Get agent-specific world adapter
        agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
        adapter = self.world_adapters.get(agent_uid, self.world_adapter)

        # Create agent-specific random generator for consistent but different shuffles
        agent_random = random.Random(agent_uid * 1000 + self.step_count + hash(agent_id) % 1000)

        rooms = adapter.get_room_ids()
        entities = adapter.get_interactable_entities()

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

        # Shuffle all lists using AGENT-SPECIFIC random for different orderings per agent
        rooms_shuffled = rooms.copy()
        agent_random.shuffle(rooms_shuffled)
        furniture_shuffled = furniture.copy()
        agent_random.shuffle(furniture_shuffled)
        articulated_shuffled = articulated.copy()
        agent_random.shuffle(articulated_shuffled)
        objects_shuffled = objects.copy()
        agent_random.shuffle(objects_shuffled)
        entities_shuffled = entities.copy()
        agent_random.shuffle(entities_shuffled)

        # Navigation
        nav_targets = rooms_shuffled[:5] + [f["name"] for f in furniture_shuffled[:5]]
        agent_random.shuffle(nav_targets)
        if nav_targets:
            actions.append({
                "name": "Navigate",
                "description": "Navigate to a room or furniture",
                "targets": nav_targets,
            })

        # Explore
        if rooms_shuffled:
            actions.append({
                "name": "Explore",
                "description": "Search a room by visiting receptacles",
                "targets": rooms_shuffled[:5],
            })

        # Open/Close
        open_targets = [f["name"] for f in articulated_shuffled[:10]]
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
        pick_targets = [obj["name"] for obj in objects_shuffled[:10]]
        if pick_targets:
            actions.append({
                "name": "Pick",
                "description": "Pick up an object",
                "targets": pick_targets,
            })

        # Custom EMTOM actions
        for action_name in EMTOM_ACTIONS.keys():
            targets = [e["name"] for e in entities_shuffled[:10]]
            actions.append({
                "name": action_name,
                "description": f"Custom EMTOM action: {action_name}",
                "targets": targets,
            })

        return actions

    def _execute_action(self, agent_id: str, action_choice: ActionChoice) -> Dict[str, Any]:
        """Execute an action in the Habitat environment.

        Order of operations:
        1. Check mechanics for blocking/transformation (but don't apply state changes yet)
        2. If blocked by mechanic, return immediately
        3. Execute in Habitat (physical action)
        4. If Habitat fails (too far, occluded), return failure WITHOUT applying mechanics
        5. If Habitat succeeds, apply mechanics to game state
        """
        action_name = action_choice.action
        target = action_choice.target or ""

        # Get agent UID from agent_id (e.g., "agent_0" -> 0)
        agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else int(agent_id)
        agent = self.agents.get(agent_uid)

        # Check mechanics for transformation/blocking info (doesn't modify state)
        from emtom.mechanics.handlers import apply_mechanics
        mech_result = apply_mechanics(action_name, agent_id, target, self.game_manager.get_state())

        # If mechanic blocked the action (e.g., locked door), first navigate to the target
        # so the video shows the agent approaching before being blocked
        if mech_result.blocked:
            # Extract base target name (remove code suffix like "[#127]")
            import re
            base_target = re.sub(r'\s*\[#\d+\]$', '', target)

            # Navigate to the target first (if we have an agent with navigation)
            if agent is not None and "Navigate" in agent.tools:
                try:
                    obs = self.env.get_observations()
                    low_level_action, response = agent.process_high_level_action(
                        "Navigate", base_target, obs
                    )

                    if low_level_action is not None:
                        tool = agent.tools["Navigate"]
                        skill_steps = 0

                        while skill_steps < self._max_skill_steps:
                            try:
                                raw_obs, reward, done, info = self.env.step({agent_uid: low_level_action})
                            except AssertionError as e:
                                if "Episode over" in str(e):
                                    self._episode_done = True
                                    break
                                raise
                            parsed_obs = self.env.parse_observations(raw_obs)
                            self._record_frame(parsed_obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
                            skill_steps += 1

                            if done:
                                self._episode_done = True
                                break

                            if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                                is_done = tool.skill._is_skill_done(
                                    raw_obs, None, None, torch.ones(1, 1), 0
                                )
                                if is_done:
                                    break

                            low_level_action, response = agent.process_high_level_action(
                                "Navigate", base_target, raw_obs
                            )
                            if low_level_action is None:
                                break

                        # Record a few extra frames at the blocked door for visual clarity
                        for _ in range(5):
                            obs = self.env.get_observations()
                            self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
                except Exception as e:
                    # If navigation fails, still record a frame
                    print(f"[HabitatExplorer] Navigation to blocked target failed: {e}")
                    obs = self.env.get_observations()
                    self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
            else:
                # No navigation available, still record a frame
                obs = self.env.get_observations()
                self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)

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
        surprise_trigger = mech_result.surprise_trigger if mech_result.applies else None

        # Custom EMTOM action - handled by GameStateManager
        if action_name in EMTOM_ACTIONS:
            # Apply custom action to game state
            state, custom_result = self.game_manager.apply_action(action_name, agent_id, target)
            # Record frame for custom action
            obs = self.env.get_observations()
            self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
            return {
                "success": custom_result.success,
                "observation": mechanic_observation or custom_result.observation,
                "surprise": surprise_trigger or custom_result.surprise_trigger,
            }

        # Handle Pick action for virtual objects (e.g., key spawned on table)
        if actual_action == "Pick":
            state = self.game_manager.get_state()
            import re as re_module

            # Helper to check if target matches a virtual object
            def matches_virtual_object(target_name, obj_id, obj_info):
                obj_name = obj_info.get("name", obj_id)
                # Exact match
                if target_name == obj_name or target_name == obj_id:
                    return True
                # Case-insensitive match
                if target_name.lower() == obj_name.lower() or target_name.lower() == obj_id.lower():
                    return True
                # For keys: match if target contains the key's code
                # e.g., target="key [#127]" matches obj_name="key [#127]"
                # or target="key" matches any key
                if "key" in obj_name.lower() and "key" in target_name.lower():
                    # Extract code from object name (e.g., "key [#127]" -> "127")
                    obj_code_match = re_module.search(r'\[#(\d+)\]', obj_name)
                    target_code_match = re_module.search(r'\[#(\d+)\]', target_name)
                    if obj_code_match and target_code_match:
                        # Both have codes - must match
                        return obj_code_match.group(1) == target_code_match.group(1)
                    elif not target_code_match:
                        # Target is just "key" without code - match any key
                        return True
                return False

            # Check if target matches a virtual object by name or id
            for obj_id, obj_info in state.world_objects.items():
                obj_name = obj_info.get("name", obj_id)
                if matches_virtual_object(actual_target, obj_id, obj_info):
                    # Check if already picked up
                    if obj_id in state.agent_inventory.get(agent_id, []):
                        # Record frame even for already-picked-up items
                        obs = self.env.get_observations()
                        self._record_frame(obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)
                        return {
                            "success": False,
                            "observation": f"You already have the {obj_name}.",
                        }

                    # Navigate to the key's location first (for video)
                    location = obj_info.get("location", "the table")
                    if agent is not None and "Navigate" in agent.tools and location:
                        try:
                            obs = self.env.get_observations()
                            low_level_action, response = agent.process_high_level_action(
                                "Navigate", location, obs
                            )

                            if low_level_action is not None:
                                tool = agent.tools["Navigate"]
                                skill_steps = 0

                                while skill_steps < self._max_skill_steps:
                                    try:
                                        raw_obs, reward, done, info = self.env.step({agent_uid: low_level_action})
                                    except AssertionError as e:
                                        if "Episode over" in str(e):
                                            self._episode_done = True
                                            break
                                        raise
                                    parsed_obs = self.env.parse_observations(raw_obs)
                                    self._record_frame(parsed_obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)
                                    skill_steps += 1

                                    if done:
                                        self._episode_done = True
                                        break

                                    if hasattr(tool, 'skill') and hasattr(tool.skill, '_is_skill_done'):
                                        is_done = tool.skill._is_skill_done(
                                            raw_obs, None, None, torch.ones(1, 1), 0
                                        )
                                        if is_done:
                                            break

                                    low_level_action, response = agent.process_high_level_action(
                                        "Navigate", location, raw_obs
                                    )
                                    if low_level_action is None:
                                        break

                                # Record a few extra frames at the pickup location
                                for _ in range(3):
                                    obs = self.env.get_observations()
                                    self._record_frame(obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)
                        except Exception as e:
                            print(f"[HabitatExplorer] Navigation to key location failed: {e}")
                            # Still record a frame
                            obs = self.env.get_observations()
                            self._record_frame(obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)

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

                    # Record frame after successful pickup
                    obs = self.env.get_observations()
                    self._record_frame(obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)

                    return {
                        "success": True,
                        "observation": f"\033[92m✓ You pick up the {obj_name} from {location}. It's now in your inventory.\033[0m",
                    }

            # If we're looking for a key but didn't find it, provide helpful error
            if "key" in actual_target.lower():
                # Record frame for failed key pickup
                obs = self.env.get_observations()
                self._record_frame(obs, {agent_uid: ("Pick", actual_target)}, current_agent_id=agent_id)

                # List available keys
                available_keys = [
                    obj_info.get("name", oid)
                    for oid, obj_info in state.world_objects.items()
                    if "key" in obj_info.get("name", oid).lower()
                    and oid not in state.agent_inventory.get(agent_id, [])
                ]
                if available_keys:
                    return {
                        "success": False,
                        "observation": f"Could not find '{actual_target}'. Available keys: {', '.join(available_keys)}",
                    }
                else:
                    return {
                        "success": False,
                        "observation": f"There is no key here. Try exploring to find one.",
                    }

        # Partnr tools - execute in Habitat
        if agent is None:
            # Still record a frame even if no agent
            obs = self.env.get_observations()
            self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
            return {
                "success": False,
                "observation": f"No agent configured for {agent_id}. Cannot execute {action_name}[{target}].",
            }

        if actual_action and actual_action not in agent.tools:
            # Still record a frame even if tool not available
            obs = self.env.get_observations()
            self._record_frame(obs, {agent_uid: (action_name, target)}, current_agent_id=agent_id)
            return {
                "success": False,
                "observation": f"Tool '{actual_action}' not available for {agent_id}.",
            }

        obs = self.env.get_observations()

        low_level_action, response = agent.process_high_level_action(
            actual_action, actual_target, obs
        )

        if low_level_action is None:
            obs_text = response
            # Record frame even when no low-level action
            self._record_frame(obs, {agent_uid: (actual_action, actual_target)}, current_agent_id=agent_id)
        else:
            # Execute motor skill
            skill_steps = 0
            tool = agent.tools[actual_action]

            while skill_steps < self._max_skill_steps:
                try:
                    raw_obs, reward, done, info = self.env.step({agent_uid: low_level_action})
                except AssertionError as e:
                    # Episode ended - handle gracefully
                    if "Episode over" in str(e):
                        obs_text = f"Episode ended during {actual_action}[{actual_target}]"
                        self._episode_done = True
                        break
                    raise
                parsed_obs = self.env.parse_observations(raw_obs)
                self._record_frame(parsed_obs, {agent_uid: (actual_action, actual_target)}, current_agent_id=agent_id)
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

                low_level_action, response = agent.process_high_level_action(
                    actual_action, actual_target, raw_obs
                )

                if low_level_action is None:
                    break

            obs_text = response or f"Executed {actual_action}[{actual_target}]"

        # Check if Habitat action failed (e.g., "too far", "occluded")
        habitat_failed = any(
            fail_phrase in obs_text.lower()
            for fail_phrase in ["too far", "occluded", "failed to", "unexpected failure", "cannot"]
        )

        if habitat_failed:
            # Habitat action failed - don't apply mechanics, return failure
            return {
                "success": False,
                "observation": obs_text,
            }

        # Habitat action succeeded - now apply mechanic state changes
        # This updates game state (e.g., inverse_state flips, counting_state updates)
        if mech_result.applies:
            _, applied_result = self.game_manager.apply_action(action_name, agent_id, target)
            # Use the mechanic's observation if it provides one
            if mechanic_observation:
                obs_text = f"{obs_text} {mechanic_observation}"
            # Update surprise trigger from applied result if not already set
            if not surprise_trigger and applied_result.surprise_trigger:
                surprise_trigger = applied_result.surprise_trigger

        return {
            "success": True,
            "observation": obs_text,
            "surprise": surprise_trigger,
        }

    def _record_frame(
        self,
        obs: Dict[str, Any],
        actions: Dict[int, Any],
        current_agent_id: Optional[str] = None,
    ) -> bool:
        """Record a video frame with step counter and inventory overlays.

        Args:
            obs: Observations from the environment
            actions: Dict mapping agent UID to (action_name, target) tuple
            current_agent_id: The agent currently taking action (for per-agent recording)

        Returns:
            True if frame was recorded successfully, False otherwise.
        """
        import cv2

        recorded = False

        # Get step info for overlays
        n_agents = len(self.config.agent_ids)
        total_steps = n_agents * self.config.max_steps
        state = self.game_manager.get_state()
        world_objects = getattr(state, 'world_objects', {})

        # NEW: Record to per-agent third-person recorder
        if self._per_agent_recorder and current_agent_id:
            try:
                # Get action for this agent
                agent_uid = int(current_agent_id.split("_")[-1]) if "_" in current_agent_id else 0
                action = actions.get(agent_uid)

                # Get inventory for this agent
                agent_inv = state.agent_inventory.get(current_agent_id, [])
                inv_names = []
                for item_id in agent_inv:
                    if item_id in world_objects:
                        inv_names.append(world_objects[item_id].get("name", item_id))
                    else:
                        inv_names.append(item_id)

                step_info = {
                    "step": self.step_count + 1,
                    "total_steps": total_steps,
                    "inventory": inv_names,
                }

                per_agent_recorded = self._per_agent_recorder.record_frame(
                    agent_id=current_agent_id,
                    observations=obs,
                    action=action,
                    step_info=step_info,
                )
                if per_agent_recorded:
                    recorded = True

            except Exception as e:
                if not hasattr(self, '_per_agent_error_count'):
                    self._per_agent_error_count = 0
                self._per_agent_error_count += 1
                if self._per_agent_error_count <= 5:
                    print(f"[HabitatExplorer] Per-agent recording error #{self._per_agent_error_count}: {e}")

        # Legacy: Also record to combined view (DebugVideoUtil) for backwards compatibility
        if self._dvu:
            try:
                # Track frame count before and after
                frames_before = len(self._dvu.frames) if self._dvu.frames else 0

                self._dvu._store_for_video(obs, actions, popup_images={})

                frames_after = len(self._dvu.frames) if self._dvu.frames else 0

                # Verify frame was actually added
                if frames_after > frames_before:
                    recorded = True

                    # Add step counter and inventory overlays to the last frame
                    frame = self._dvu.frames[-1]
                    h, w = frame.shape[:2]

                    # Step counter (top right) - show global step / total steps
                    step_text = f"Step: {self.step_count + 1}/{total_steps}"
                    text_size = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    x_pos = w - text_size[0] - 20
                    cv2.putText(
                        frame,
                        step_text,
                        (x_pos, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Inventory (bottom right)
                    # Collect inventory items for all agents
                    inv_items = []
                    for agent_id in self.config.agent_ids:
                        agent_inv = state.agent_inventory.get(agent_id, [])
                        for item_id in agent_inv:
                            if item_id in world_objects:
                                inv_items.append(world_objects[item_id].get("name", item_id))
                            else:
                                inv_items.append(item_id)

                    if inv_items:
                        inv_text = f"Inventory: {', '.join(inv_items)}"
                    else:
                        inv_text = "Inventory: (empty)"

                    text_size = cv2.getTextSize(inv_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    x_pos = w - text_size[0] - 20
                    y_pos = h - 20
                    cv2.putText(
                        frame,
                        inv_text,
                        (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # Frame wasn't added - track this
                    if not hasattr(self, '_frame_skip_count'):
                        self._frame_skip_count = 0
                    self._frame_skip_count += 1
                    if self._frame_skip_count <= 5:
                        print(f"[HabitatExplorer] Warning: Frame not added (skip #{self._frame_skip_count})")

            except Exception as e:
                # Track error count instead of just logging first error
                if not hasattr(self, '_frame_error_count'):
                    self._frame_error_count = 0
                self._frame_error_count += 1
                # Log first 5 errors and then every 100th error
                if self._frame_error_count <= 5 or self._frame_error_count % 100 == 0:
                    print(f"[HabitatExplorer] Warning: Frame recording error #{self._frame_error_count}: {e}")
                    # Show observation keys to help debug
                    if obs:
                        rgb_keys = [k for k in obs.keys() if 'rgb' in k.lower()]
                        print(f"  Available RGB observation keys: {rgb_keys}")
                    if self._frame_error_count == 1:
                        import traceback
                        traceback.print_exc()

        if self._fpv_recorder:
            try:
                self._fpv_recorder.record_step(obs)
            except Exception as e:
                if not hasattr(self, '_fpv_error_count'):
                    self._fpv_error_count = 0
                self._fpv_error_count += 1
                if self._fpv_error_count <= 3:
                    print(f"[HabitatExplorer] Warning: FPV recording error #{self._fpv_error_count}: {e}")

        return recorded

    def _save_videos(self) -> Dict[str, str]:
        """Save recorded videos and return paths."""
        video_paths = {}

        # Print video statistics
        n_agents = len(self.config.agent_ids)
        total_steps = n_agents * self.config.max_steps
        fps = getattr(self.config, 'video_fps', 30)

        print(f"\n[HabitatExplorer] Video recording summary:")
        print(f"  - Total high-level steps completed: {self.step_count}/{total_steps}")
        print(f"  - Per-agent steps: {dict(self.agent_step_counts)}")

        # Report any recording errors
        if hasattr(self, '_frame_error_count') and self._frame_error_count > 0:
            print(f"  - WARNING: {self._frame_error_count} frame recording errors occurred!")
        if hasattr(self, '_frame_skip_count') and self._frame_skip_count > 0:
            print(f"  - WARNING: {self._frame_skip_count} frames were skipped!")
        if hasattr(self, '_per_agent_error_count') and self._per_agent_error_count > 0:
            print(f"  - WARNING: {self._per_agent_error_count} per-agent recording errors occurred!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # NEW: Save per-agent third-person videos and stitch them together
        if self._per_agent_recorder:
            frame_counts = self._per_agent_recorder.get_frame_counts()
            total_per_agent_frames = sum(frame_counts.values())
            print(f"  - Per-agent third-person frames: {frame_counts}")

            if total_per_agent_frames > 0:
                try:
                    # Save individual videos for each agent
                    individual_paths = self._per_agent_recorder.save_individual_videos(postfix=timestamp)
                    for agent_id, path in individual_paths.items():
                        video_paths[f"third_person_{agent_id}"] = path

                    # Stitch videos together in round-robin order (agent 0 -> agent 1 -> agent 2 -> agent 0...)
                    stitched_path = self._per_agent_recorder.stitch_videos_round_robin(postfix=timestamp)
                    if stitched_path:
                        video_paths["third_person_stitched"] = stitched_path
                        print(f"  - Stitched video (round-robin): {stitched_path}")

                        # Calculate total duration
                        total_stitched_frames = sum(frame_counts.values())
                        duration = total_stitched_frames / fps
                        print(f"  - Stitched video duration: {duration:.1f}s @ {fps} FPS")

                except Exception as e:
                    print(f"[HabitatExplorer] Failed to save per-agent videos: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  - WARNING: No frames recorded for per-agent videos!")

        # Legacy: Also save combined view video (DebugVideoUtil) for backwards compatibility
        if self._dvu:
            frame_count = len(self._dvu.frames) if self._dvu.frames else 0
            print(f"  - Legacy combined view frames: {frame_count}")

            if frame_count > 0:
                try:
                    # Pass fps to ensure correct video timing
                    self._dvu._make_video(play=False, postfix=f"exploration_{timestamp}", fps=fps)
                    # Video is saved to {output_dir}/videos/video-exploration_{timestamp}-{ms}.mp4
                    video_paths["third_person_combined"] = f"{self.config.log_path}/videos/"
                    duration = frame_count / fps
                    print(f"  - Legacy combined video duration: {duration:.1f}s @ {fps} FPS")
                except Exception as e:
                    print(f"[HabitatExplorer] Failed to save legacy third-person video: {e}")
                    import traceback
                    traceback.print_exc()

        if self._fpv_recorder:
            fpv_frame_count = sum(len(frames) for frames in self._fpv_recorder._frames.values()) if self._fpv_recorder._frames else 0
            print(f"  - FPV frames recorded: {fpv_frame_count}")

            if fpv_frame_count > 0:
                try:
                    fpv_paths = self._fpv_recorder.save(postfix=timestamp)
                    video_paths.update(fpv_paths)
                except Exception as e:
                    print(f"[HabitatExplorer] Failed to save FPV video: {e}")

        return video_paths

    def _save_exploration_prompts(self) -> None:
        """
        Save LLM prompts and traces from the CuriosityModel.

        Similar to DecentralizedEvaluationRunner._log_planner_data(), this saves:
        - prompts/{agent_id}/prompt-exploration-{episode_id}-{agent_id}.txt
        - traces/{agent_id}/trace-exploration-{episode_id}-{agent_id}.txt
        - planner-log/planner-log-exploration-{episode_id}.json
        """
        print("\n[HabitatExplorer] Saving exploration prompts and traces...")

        # Get all agent prompts from CuriosityModel
        all_prompts = self.curiosity.get_all_prompts()

        # Collect prompts and traces for each agent
        prompts = {}
        traces = {}

        for agent_id in self.config.agent_ids:
            # Get this agent's full prompt with conversation history
            if agent_id in all_prompts and all_prompts[agent_id]:
                prompts[agent_id] = all_prompts[agent_id]

                # For traces, extract just the task + actions (without system prompt)
                # by finding where the actual interaction starts
                prompt = all_prompts[agent_id]
                task_marker = "Task:"
                if task_marker in prompt:
                    trace_start = prompt.find(task_marker)
                    traces[agent_id] = prompt[trace_start:]
                else:
                    # Fallback: use entire prompt as trace
                    traces[agent_id] = prompt

        # Save prompts and traces for all agents
        if prompts:
            self.logger.save_prompts(prompts, traces)
            print(f"[HabitatExplorer] Saved prompts for {len(prompts)} agents: {list(prompts.keys())}")

        # Build planner log with step-by-step info
        planner_log = {
            "task": "Exploration",
            "mechanics_active": self.game_manager.get_state().active_mechanics,
            "total_steps": self.step_count,
            "total_surprises": len(self.surprise_moments),
            "steps": [],
        }

        # Add per-step info from logger
        for step_record in self.logger.steps:
            step_info = {
                "step": step_record.step,
                "timestamp": step_record.timestamp,
                "agent_actions": step_record.agent_actions,
                "observations": step_record.observations,
                "surprises": [s.to_dict() for s in step_record.surprises],
                "inventory": step_record.inventory,
            }
            planner_log["steps"].append(step_info)

        # Add summary
        planner_log["surprise_summary"] = [
            s.to_dict() for s in self.surprise_moments
        ]

        # Save planner log
        self.logger.save_planner_log(planner_log)
