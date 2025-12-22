"""
Base runner class for EMTOM benchmark modes.

This provides common setup for environment, GameStateManager, agents, tools,
video recording, and logging across all modes (exploration, benchmark, test).
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from emtom.tracing import EventLog

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.agent import Agent


class EMTOMBaseRunner(ABC):
    """
    Base class for all EMTOM runners.

    Handles common setup:
    - Environment initialization (sensors, actions, measures)
    - GameStateManager with mechanics
    - Agent creation with tool injection
    - Video recording (DebugVideoUtil)
    - Output directory structure
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the base runner.

        Args:
            config: Hydra configuration (should be already fixed/setup)
        """
        self.config = config

        # Will be set during setup
        self.env_interface: Optional["EnvironmentInterface"] = None
        self.game_manager = None
        self.world_adapter = None
        self.agents: Dict[int, "Agent"] = {}
        self.output_dir: str = ""

        # Video recording
        self._dvu = None
        self._fpv_recorder = None

        # State tracking
        self._step_count = 0
        self._action_history: List[Dict[str, Any]] = []
        self._episode_done = False

        # Causal event log (ARE-style tracing)
        self.event_log = EventLog()

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        agent_actions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Full setup sequence. Call before run().

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Optional task data with mechanics/bindings
            output_dir: Output directory for videos/logs
            agent_actions: Optional dict mapping agent_id to list of allowed actions.
                           If None, all actions are available to all agents.
                           Example: {"agent_0": ["Navigate", "Open", "Communicate"], "agent_1": ["Navigate"]}
        """
        self.env_interface = env_interface
        self.output_dir = output_dir or getattr(
            self.config.paths, 'results_dir', 'outputs/emtom'
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Store agent actions for tool setup
        self._agent_actions = agent_actions

        self._setup_game_manager(task_data)
        self._setup_agents()
        self._setup_tools()
        self._setup_video()
        self._setup_logging_dirs()

    def _setup_game_manager(self, task_data: Optional[Dict[str, Any]] = None) -> None:
        """Create GameStateManager, load mechanics, auto-bind if needed."""
        from emtom import GameStateManager, list_mechanics
        from emtom.exploration.habitat_explorer import HabitatWorldAdapter

        self.game_manager = GameStateManager(self.env_interface)
        self.world_adapter = HabitatWorldAdapter(self.env_interface, agent_uid=0)

        # Determine mechanics to use
        if task_data and task_data.get("mechanics"):
            self.game_manager.initialize_from_task(task_data)
            print(f"[EMTOMBaseRunner] Loaded mechanics from task: {len(task_data['mechanics'])} bindings")
        else:
            # Default: enable all mechanics
            all_mechanics = list_mechanics()
            self.game_manager.initialize_from_task({
                "mechanics": [{"mechanic_type": m} for m in all_mechanics]
            })
            print(f"[EMTOMBaseRunner] Enabled all mechanics: {all_mechanics}")

        # Get entities from environment and set on game state
        entities = self.world_adapter.get_interactable_entities()
        state = self.game_manager.get_state()
        state.entities = entities
        self.game_manager.set_state(state)

        # Auto-bind if no explicit bindings provided
        if not (task_data and task_data.get("mechanics")):
            state, bindings = self.game_manager.auto_bind_mechanics()
            if bindings:
                self._print_bindings(bindings)

    def _print_bindings(self, bindings: dict) -> None:
        """Pretty print auto-bound mechanics."""
        print("\n[EMTOMBaseRunner] Auto-bound mechanics:")

        # Mechanics
        for mech in ["inverse_state", "remote_control", "counting_state",
                     "delayed_effect", "state_mirroring", "conditional_unlock"]:
            if mech in bindings:
                info = bindings[mech]
                if mech == "inverse_state":
                    print(f"  • {mech}: {info.get('target')}")
                elif mech == "remote_control":
                    print(f"  • {mech}: {info.get('trigger')} → {info.get('target')}")
                elif mech == "counting_state":
                    print(f"  • {mech}: {info.get('target')} (count={info.get('required_count')})")
                elif mech == "delayed_effect":
                    print(f"  • {mech}: {info.get('target')} (delay={info.get('delay_steps')})")
                elif mech == "state_mirroring":
                    pair = info.get('pair', [])
                    print(f"  • {mech}: {pair[0] if pair else '?'} ↔ {pair[1] if len(pair) > 1 else '?'}")
                elif mech == "conditional_unlock":
                    print(f"  • {mech}: {info.get('prerequisite')} unlocks {info.get('target')}")

        # Scenario
        if "scenario" in bindings:
            s = bindings["scenario"]
            print(f"\n  Scenario: {s.get('id')} ({s.get('theme')})")
            print(f"  Title: {s.get('title')}")

        # Hidden items
        if "hidden_items" in bindings:
            print(f"\n  Hidden items:")
            for container, item in bindings["hidden_items"].items():
                item_name = bindings.get("item_definitions", {}).get(item, item)
                print(f"    • {item_name} in {container}")

        # Locked containers
        if "locked_containers" in bindings:
            print(f"\n  Locked containers:")
            for container, key_type in bindings["locked_containers"].items():
                print(f"    • {container} (requires {key_type})")

        # Suggested locations
        if "suggested_locations" in bindings:
            locs = bindings["suggested_locations"]
            print(f"\n  Suggested locations: {', '.join(locs)}")

        print()

    def _setup_agents(self) -> None:
        """Create Agent instances from config."""
        from habitat_llm.agent import Agent

        # Agent configs are at config.agents.agent_X.config
        # evaluation.agents only has uid references
        if not hasattr(self.config, 'agents'):
            print("[EMTOMBaseRunner] Warning: No agents in config")
            return

        for agent_name, agent_entry in self.config.agents.items():
            if not hasattr(agent_entry, 'config'):
                continue

            # Get uid from agent name (e.g., "agent_0" -> 0) or from config
            uid = getattr(agent_entry, 'uid', None)
            if uid is None:
                uid = int(agent_name.split("_")[-1]) if "_" in agent_name else 0

            agent = Agent(
                uid=uid,
                agent_conf=agent_entry.config,
                env_interface=self.env_interface,
            )
            self.agents[uid] = agent
            print(f"[EMTOMBaseRunner] Created {agent_name} (uid={uid}) with tools: {list(agent.tools.keys())}")

    def _setup_tools(self) -> None:
        """Inject EMTOM tools and Communicate into agents based on allowed actions."""
        from emtom.actions import get_emtom_tools
        from emtom.actions.tool_wrapper import wrap_habitat_tools
        from habitat_llm.tools.perception.communication_tool import CommunicationTool

        for uid, agent in self.agents.items():
            agent_id = f"agent_{uid}"

            # Get allowed actions for this agent (None means all allowed)
            allowed_actions = None
            if self._agent_actions is not None:
                allowed_actions = self._agent_actions.get(agent_id, [])

            def is_allowed(action_name: str) -> bool:
                """Check if action is allowed for this agent."""
                if allowed_actions is None:
                    return True  # No restrictions
                return action_name in allowed_actions

            # Add EMTOM tools (Use, Search, Inspect) if allowed
            emtom_tools = get_emtom_tools(agent_uid=uid)
            for tool_name, tool in emtom_tools.items():
                if is_allowed(tool_name):
                    tool.set_environment(self.env_interface)
                    tool.set_game_manager(self.game_manager)
                    agent.tools[tool_name] = tool

            # Add Communicate tool if allowed
            if is_allowed("Communicate"):
                comm_config = OmegaConf.create({
                    "name": "Communicate",
                    "description": "Send a message to the other agent. Usage: Communicate[your message]. Keep messages on a single line."
                })
                comm_tool = CommunicationTool(comm_config)
                comm_tool.agent_uid = uid
                comm_tool.set_environment(self.env_interface)
                agent.tools["Communicate"] = comm_tool

            # Wrap Habitat tools with EMTOM condition checks (locks, etc.)
            # This also filters based on allowed_actions
            wrap_habitat_tools(agent, self.game_manager, allowed_actions)

            tools_added = list(agent.tools.keys())
            print(f"[EMTOMBaseRunner] Agent_{uid} tools: {tools_added}")

    def inject_tool_from_item(self, uid: int, item_id: str) -> bool:
        """
        Inject a tool action from an inventory item into an agent's toolset.

        Called when an agent obtains a TOOL-type item that grants a new action.

        Args:
            uid: Agent UID
            item_id: Item ID that was obtained

        Returns:
            True if a tool was injected, False otherwise
        """
        from emtom.state.items import ItemType
        from emtom.actions.custom_actions import DynamicItemTool

        if uid not in self.agents:
            return False

        item_def = self.game_manager.get_item_definition(item_id)
        if not item_def or item_def.item_type != ItemType.TOOL:
            return False

        if not item_def.grants_action:
            return False

        agent = self.agents[uid]

        # Skip if agent already has this tool
        if item_def.grants_action in agent.tools:
            return False

        # Create the dynamic tool
        tool = DynamicItemTool(
            name=item_def.grants_action,
            description=item_def.action_description or f"Use {item_def.name}",
            item_id=item_id,
            argument_types=item_def.action_targets or ["OBJECT_INSTANCE", "FURNITURE_INSTANCE"],
            consumable=item_def.consumable,
            agent_uid=uid,
        )
        tool.set_environment(self.env_interface)
        tool.set_game_manager(self.game_manager)

        agent.tools[item_def.grants_action] = tool
        print(f"[EMTOMBaseRunner] Injected {item_def.grants_action} tool from {item_def.name} for agent_{uid}")
        return True

    def check_and_inject_item_tools(self, uid: int) -> None:
        """
        Check agent's inventory and inject any tools from TOOL-type items.

        Called after item grants to ensure tools are available.

        Args:
            uid: Agent UID to check
        """
        agent_id = f"agent_{uid}"
        inventory = self.game_manager.state.agent_inventory.get(agent_id, [])

        for item_id in inventory:
            self.inject_tool_from_item(uid, item_id)

    def _setup_video(self) -> None:
        """Initialize video recording utilities."""
        save_video = True
        if hasattr(self.config, 'evaluation') and hasattr(self.config.evaluation, 'save_video'):
            save_video = self.config.evaluation.save_video

        if not save_video:
            return

        try:
            from habitat_llm.examples.example_utils import DebugVideoUtil

            video_dir = os.path.join(self.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

            self._dvu = DebugVideoUtil(
                self.env_interface,
                video_dir,
                unique_postfix=True,
            )
            print(f"[EMTOMBaseRunner] Video recording enabled: {video_dir}")
        except Exception as e:
            print(f"[EMTOMBaseRunner] Video setup failed: {e}")

    def _setup_logging_dirs(self) -> None:
        """Create output directory structure."""
        for subdir in ["prompts", "traces", "planner-log"]:
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)

    def record_frame(self, observations: Dict[str, Any], actions: Optional[Dict] = None) -> None:
        """Record a video frame."""
        if self._dvu:
            try:
                self._dvu._store_for_video(observations, actions or {}, popup_images={})
            except Exception:
                pass

    def execute_action(
        self,
        uid: int,
        action_name: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Execute an action via GameStateManager and agent tools.

        Routes all actions through GameStateManager first to apply mechanics,
        then executes in Habitat via agent tools.

        Args:
            uid: Agent UID
            action_name: Name of action (Navigate, Open, Pick, etc.)
            target: Target entity name

        Returns:
            Dict with success, observation, and optional surprise info

        Order of operations:
            1. Check mechanics for blocking/transformation (doesn't modify state)
            2. If blocked by mechanic, return immediately
            3. Execute in Habitat (physical action)
            4. If Habitat fails (too far, occluded), return failure WITHOUT applying state
            5. If Habitat succeeds, apply mechanics to game state
        """
        import torch
        from emtom.mechanics.handlers import apply_mechanics

        agent_id = f"agent_{uid}"

        # 1. Check mechanics (doesn't modify state yet)
        mech_result = apply_mechanics(
            action_name, agent_id, target, self.game_manager.get_state()
        )

        # 2. If mechanic blocked the action, return early WITHOUT executing in Habitat
        if mech_result.blocked:
            # Log blocked action
            self.event_log.log_action(
                step=self._step_count,
                agent_id=agent_id,
                action=action_name,
                target=target,
                result=mech_result.observation,
                success=False,
            )
            return {
                "success": False,
                "observation": mech_result.observation,
                "surprise": mech_result.surprise_trigger,
                "blocked": True,
            }

        # Get actual action to execute (may be transformed by mechanic)
        actual_action = mech_result.actual_action or action_name
        actual_target = mech_result.actual_target or target
        mechanic_observation = mech_result.observation if mech_result.applies else None

        # 3. Execute via agent tools in Habitat
        if uid not in self.agents:
            return {
                "success": False,
                "observation": f"No agent with uid {uid}",
            }

        agent = self.agents[uid]

        if actual_action not in agent.tools:
            return {
                "success": False,
                "observation": f"Tool '{actual_action}' not available",
            }

        obs = self.env_interface.get_observations()
        low_level_action, response = agent.process_high_level_action(
            actual_action, actual_target, obs
        )

        if low_level_action is None:
            obs_text = response or f"Executed {actual_action}[{actual_target}]"
        else:
            # Execute motor skill
            tool = agent.tools[actual_action]
            skill_steps = 0
            max_skill_steps = 500

            while skill_steps < max_skill_steps:
                try:
                    raw_obs, reward, done, info = self.env_interface.step({uid: low_level_action})
                except AssertionError as e:
                    if "Episode over" in str(e):
                        self._episode_done = True
                        break
                    raise

                parsed_obs = self.env_interface.parse_observations(raw_obs)
                self.record_frame(parsed_obs, {uid: (actual_action, actual_target)})
                skill_steps += 1

                if done:
                    self._episode_done = True
                    break

                # Check if skill is done
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

        # 4. Check if Habitat action failed (e.g., "too far", "occluded")
        habitat_failed = any(
            fail_phrase in obs_text.lower()
            for fail_phrase in ["too far", "occluded", "failed to", "unexpected failure", "cannot"]
        )

        if habitat_failed:
            # Log failed action
            self.event_log.log_action(
                step=self._step_count,
                agent_id=agent_id,
                action=action_name,
                target=target,
                result=obs_text,
                success=False,
            )
            # Habitat action failed - don't apply mechanics, return failure
            return {
                "success": False,
                "observation": obs_text,
            }

        # 5. Habitat action succeeded - now apply mechanic state changes
        # Log the successful action
        action_event = self.event_log.log_action(
            step=self._step_count,
            agent_id=agent_id,
            action=action_name,
            target=target,
            result=obs_text,
            success=True,
        )

        if mech_result.applies:
            state, result = self.game_manager.apply_action(action_name, agent_id, target)
            # Log mechanic effect
            self.event_log.log_mechanic(
                step=self._step_count,
                mechanic=mech_result.mechanic_type or "unknown",
                trigger=target,
                effect=mechanic_observation or result.observation,
                caused_by=action_event.event_id,
            )
            # Append mechanic observation to Habitat observation
            if mechanic_observation:
                obs_text = f"{obs_text} {mechanic_observation}"
            surprise_trigger = mech_result.surprise_trigger or result.surprise_trigger
        else:
            surprise_trigger = None

        return {
            "success": True,
            "observation": obs_text,
            "surprise": surprise_trigger,
        }

    def execute_parsed_action(self, uid: int, action_str: str) -> Dict[str, Any]:
        """
        Execute an action from a string like "Navigate[kitchen_1]".

        Args:
            uid: Agent UID
            action_str: Action string in format "ActionName[target]"

        Returns:
            Result dict from execute_action
        """
        match = re.match(r'(\w+)\[([^\]]+)\]', action_str)
        if not match:
            return {
                "success": False,
                "observation": f"Invalid action format: {action_str}. Use ActionName[target]",
            }

        action_name, target = match.groups()
        return self.execute_action(uid, action_name, target)

    def save_video(self, suffix: str) -> Optional[str]:
        """Save recorded video with given suffix."""
        if not self._dvu or not self._dvu.frames:
            return None

        try:
            self._dvu._make_video(play=False, postfix=suffix)
            return os.path.join(self.output_dir, "videos")
        except Exception as e:
            print(f"[EMTOMBaseRunner] Failed to save video: {e}")
            return None

    def save_planner_log(self, data: Dict[str, Any]) -> str:
        """Save planner log JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, "planner-log", f"planner-log-{timestamp}.json")

        # Include event trace summary
        data["event_trace"] = self.event_log.get_summary()

        with open(log_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[EMTOMBaseRunner] Saved planner log: {log_file}")
        return log_file

    def save_event_log(self, suffix: str = "") -> str:
        """
        Save the full causal event log.

        Args:
            suffix: Optional suffix for the filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)

        filename = f"event-trace-{timestamp}"
        if suffix:
            filename = f"{filename}-{suffix}"
        filepath = os.path.join(trace_dir, f"{filename}.json")

        self.event_log.save(filepath)
        print(f"[EMTOMBaseRunner] Saved event trace: {filepath}")

        # Also save narrative version for human reading
        narrative_path = os.path.join(trace_dir, f"{filename}-narrative.txt")
        with open(narrative_path, "w") as f:
            f.write(self.event_log.to_narrative())
        print(f"[EMTOMBaseRunner] Saved event narrative: {narrative_path}")

        return filepath

    def get_causal_chain(self, event_id: str) -> list:
        """
        Get the causal chain leading to an event.

        Useful for debugging "why did this happen?"

        Args:
            event_id: Event ID to trace back from

        Returns:
            List of events in chronological order
        """
        return self.event_log.get_causal_chain(event_id)

    def save_prompts(self, prompts: Dict[str, str], traces: Optional[Dict[str, str]] = None) -> None:
        """
        Save per-agent prompts and traces.

        Args:
            prompts: Dict mapping agent_id -> prompt text
            traces: Optional dict mapping agent_id -> trace text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for agent_id, prompt in prompts.items():
            # Extract uid from agent_id (e.g., "agent_0" -> "0")
            uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

            prompt_dir = os.path.join(self.output_dir, "prompts", uid)
            os.makedirs(prompt_dir, exist_ok=True)

            prompt_file = os.path.join(prompt_dir, f"prompt-{timestamp}-{uid}.txt")
            with open(prompt_file, "w") as f:
                f.write(prompt)

        if traces:
            for agent_id, trace in traces.items():
                uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

                trace_dir = os.path.join(self.output_dir, "traces", uid)
                os.makedirs(trace_dir, exist_ok=True)

                trace_file = os.path.join(trace_dir, f"trace-{timestamp}-{uid}.txt")
                with open(trace_file, "w") as f:
                    f.write(trace)

    def cleanup(self) -> None:
        """Close environment and release resources."""
        if self.env_interface:
            try:
                self.env_interface.env.close()
            except Exception:
                pass

    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from environment."""
        return self.env_interface.get_observations()

    def get_world_graph(self) -> Dict[int, Any]:
        """Get world graph for all agents."""
        world_graph = {}
        for uid in self.agents.keys():
            try:
                world_graph[uid] = self.env_interface.world_graph[uid]
            except Exception:
                world_graph[uid] = None
        return world_graph

    def evaluate_task(
        self,
        success_condition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate task completion using PARTNR-style predicates.

        Args:
            success_condition: Task's success_condition dict. If None, returns empty result.

        Returns:
            Dict with percent_complete, success, failure_explanations
        """
        if success_condition is None:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": ["No success_condition defined"],
            }

        try:
            from emtom.evaluation import evaluate_task
            sim = self.env_interface.sim

            # Get world graph for name-to-handle resolution
            # Use first available agent's world graph (all agents share the same entity handles)
            world_graph = None
            if hasattr(self.env_interface, 'world_graph') and self.env_interface.world_graph:
                for uid in self.agents.keys():
                    if uid in self.env_interface.world_graph:
                        world_graph = self.env_interface.world_graph[uid]
                        break

            result = evaluate_task(success_condition, sim, world_graph=world_graph)
            return result.to_dict()
        except Exception as e:
            return {
                "percent_complete": 0.0,
                "success": False,
                "failure_explanations": [f"Evaluation error: {str(e)}"],
            }

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main execution loop. Implemented by subclasses.

        Returns:
            Dict with results (steps, history, etc.)
        """
        pass
