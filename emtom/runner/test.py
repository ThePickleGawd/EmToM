"""
Human-in-the-loop test runner for EMTOM benchmark.

Allows mixing human-controlled and LLM-controlled agents for debugging
and testing tasks interactively.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from omegaconf import DictConfig

from .base import EMTOMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class HumanTestRunner(EMTOMBaseRunner):
    """
    Runner for human-in-the-loop testing.

    Allows humans to control agents via CLI while optionally having
    some agents be LLM-controlled.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.human_agents: Set[str] = set()
        self.planners: Dict[int, Any] = {}  # uid -> LLMPlanner for LLM agents
        self.task_info: Optional[Dict[str, Any]] = None

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task_info: Optional[Dict[str, Any]] = None,
        human_agents: Optional[List[str]] = None,
    ) -> None:
        """
        Setup human test runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics
            output_dir: Output directory
            task_info: Optional task info dict for display
            human_agents: List of agent IDs to be human-controlled (default: all)
        """
        self.task_info = task_info

        # Get agent_actions from task_info if available
        agent_actions = task_info.get("agent_actions") if task_info else None

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions)

        # Determine which agents are human vs LLM
        all_agent_ids = {f"agent_{uid}" for uid in self.agents.keys()}

        if human_agents is None:
            # Default: all agents are human-controlled
            self.human_agents = all_agent_ids
        else:
            self.human_agents = set(human_agents)

        # Setup planners for LLM-controlled agents
        llm_agent_uids = [
            uid for uid in self.agents.keys()
            if f"agent_{uid}" not in self.human_agents
        ]
        self._setup_planners(llm_agent_uids)

        print(f"[HumanTestRunner] Human agents: {self.human_agents}")
        print(f"[HumanTestRunner] LLM agents: {set(f'agent_{uid}' for uid in llm_agent_uids)}")

    def _setup_planners(self, agent_uids: List[int]) -> None:
        """Initialize LLM planner for specified agents."""
        if not agent_uids:
            return

        from hydra.utils import instantiate

        if not hasattr(self.config, 'evaluation') or not hasattr(self.config.evaluation, 'agents'):
            print("[HumanTestRunner] Warning: No agents in config for planner setup")
            return

        agent_confs = list(self.config.evaluation.agents.values())

        for uid in agent_uids:
            if uid >= len(agent_confs):
                continue

            agent_conf = agent_confs[uid]
            if not hasattr(agent_conf, 'planner'):
                print(f"[HumanTestRunner] Warning: No planner config for agent_{uid}")
                continue

            # Use Hydra's instantiate pattern (same as DecentralizedEvaluationRunner)
            planner = instantiate(agent_conf.planner)
            planner = planner(env_interface=self.env_interface)
            planner.agents = [self.agents[uid]]
            self.planners[uid] = planner
            print(f"[HumanTestRunner] Created planner for agent_{uid}")

    def run(
        self,
        instruction: Dict[str, str],
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run interactive test loop.

        Args:
            instruction: Per-agent instruction dict
            max_steps: Maximum steps

        Returns:
            Results dict
        """
        self._print_header()
        self._print_controls()

        observations = self.get_observations()
        self.record_frame(observations)

        done = False

        while self._step_count < max_steps and not done and not self._episode_done:
            self._step_count += 1

            print(f"\n{'='*60}")
            print(f"STEP {self._step_count}")
            print(f"{'='*60}")

            for uid in sorted(self.agents.keys()):
                agent_id = f"agent_{uid}"

                self._print_agent_status(uid)

                if agent_id in self.human_agents:
                    # Human input
                    action = self._get_human_input(agent_id)
                    if action is None:
                        # Quit command
                        done = True
                        break
                    if action == "skip":
                        continue
                else:
                    # LLM planner
                    print(f"\n[{agent_id} is LLM-controlled]")
                    action = self._get_llm_action(uid, instruction, observations)
                    print(f"LLM chose: {action}")

                if action and action != "skip":
                    result = self.execute_parsed_action(uid, action)
                    print(f"\nResult: {result.get('observation', 'Success')}")

                    self._action_history.append({
                        "step": self._step_count,
                        "agent": agent_id,
                        "action": action,
                        "result": result.get("observation", ""),
                        "mode": "human" if agent_id in self.human_agents else "llm",
                    })

            if not done and not self._episode_done:
                observations = self.get_observations()
                self.record_frame(observations)

        # Save outputs
        self._save_outputs()

        return {
            "steps": self._step_count,
            "action_history": self._action_history,
        }

    def _print_header(self) -> None:
        """Print task header with story."""
        if self.task_info:
            print(f"\n{'='*60}")
            print(f"TASK: {self.task_info.get('title', 'Unknown')}")
            print(f"{'='*60}")
            # Print story first (sets the scene)
            if self.task_info.get('story'):
                print(f"\n{self.task_info['story']}\n")
            print(f"Goal: {self.task_info.get('public_goal', 'N/A')}")
            if self.task_info.get('public_context'):
                print(f"Context: {self.task_info['public_context']}")

    def _print_controls(self) -> None:
        """Print control instructions."""
        print(f"\n{'='*60}")
        print("CONTROLS")
        print(f"{'='*60}")
        print("Actions: Navigate[target], Open[target], Close[target], Pick[target], Place[target]")
        print("         Use[target], Inspect[target], Communicate[message]")
        print("Commands: status, mechanics, history, skip, quit, help")
        print(f"Human agents: {sorted(self.human_agents)}")
        llm_agents = sorted(f"agent_{uid}" for uid in self.planners.keys())
        print(f"LLM agents: {llm_agents}")
        print(f"{'='*60}")

    def _print_agent_status(self, uid: int) -> None:
        """Print status for an agent."""
        agent_id = f"agent_{uid}"
        mode = "human" if agent_id in self.human_agents else "LLM"

        # Get agent location
        try:
            room = self.world_adapter.get_agent_location(agent_id)
        except Exception:
            room = "unknown"

        print(f"\n--- {agent_id} ({mode}) in {room} ---")

        # Show nearby entities
        try:
            entities = self.world_adapter.get_interactable_entities()
            furniture = [e["name"] for e in entities if e["type"] == "furniture"][:10]
            objects = [e["name"] for e in entities if e["type"] == "object"]

            if furniture:
                print(f"Furniture: {', '.join(furniture)}")
            if objects:
                print(f"Objects: {', '.join(objects)}")
        except Exception:
            pass

        # Show inventory
        try:
            state = self.game_manager.get_state()
            inventory = state.agent_inventory.get(agent_id, [])
            if inventory:
                world_objects = getattr(state, 'world_objects', {})
                inv_names = [
                    world_objects.get(item, {}).get("name", item)
                    for item in inventory
                ]
                print(f"Inventory: {', '.join(inv_names)}")
        except Exception:
            pass

    def _get_human_input(self, agent_id: str) -> Optional[str]:
        """Get action from human via CLI."""
        while True:
            try:
                prompt = f"{agent_id}> "
                user_input = input(prompt).strip()
            except EOFError:
                return None
            except KeyboardInterrupt:
                return None

            if not user_input:
                continue

            cmd_lower = user_input.lower()

            # Handle special commands
            if cmd_lower in ("quit", "q", "exit"):
                return None

            if cmd_lower == "skip":
                return "skip"

            if cmd_lower == "status":
                self._print_full_status()
                continue

            if cmd_lower == "mechanics":
                self._print_mechanics()
                continue

            if cmd_lower == "history":
                self._print_history()
                continue

            if cmd_lower == "help":
                self._print_help()
                continue

            # Parse action
            action = self._parse_action(user_input)
            if action:
                return action

            print(f"Invalid action: {user_input}")
            print("Format: ActionName[target] (e.g., Navigate[kitchen_1], Open[fridge_58])")

    def _parse_action(self, text: str) -> Optional[str]:
        """Parse user input into action string."""
        # Match Action[target] format
        match = re.match(r'(\w+)\[([^\]]+)\]', text)
        if match:
            return text

        # Also accept "Action target" format
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            action, target = parts
            return f"{action}[{target}]"

        return None

    def _get_llm_action(
        self,
        uid: int,
        instruction: Dict[str, str],
        observations: Dict[str, Any],
    ) -> Optional[str]:
        """Get action from LLM planner."""
        if uid not in self.planners:
            return None

        planner = self.planners[uid]
        agent_id = f"agent_{uid}"
        agent_instruction = instruction.get(agent_id, "")

        try:
            world_graph = self.get_world_graph()
            _, planner_info, _ = planner.get_next_action(
                agent_instruction, observations, world_graph
            )
            return self._extract_action_from_planner_info(planner_info, uid)
        except Exception as e:
            print(f"[HumanTestRunner] LLM planner error: {e}")
            return None

    def _extract_action_from_planner_info(self, planner_info: Dict, uid: int) -> Optional[str]:
        """Extract action string from planner info."""
        try:
            if "high_level_action" in planner_info:
                ha = planner_info["high_level_action"]
                if isinstance(ha, dict) and uid in ha:
                    action_tuple = ha[uid]
                    if action_tuple and len(action_tuple) >= 2:
                        return f"{action_tuple[0]}[{action_tuple[1]}]"

            for key in ["action", "actions", f"agent_{uid}"]:
                if key in planner_info:
                    val = planner_info[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, tuple) and len(val) >= 2:
                        return f"{val[0]}[{val[1]}]"
        except Exception:
            pass
        return None

    def _print_full_status(self) -> None:
        """Print full world status."""
        print(f"\n{'='*50}")
        print("WORLD STATUS")
        print(f"{'='*50}")

        if self.game_manager:
            debug_info = self.game_manager.get_debug_info()
            print(f"Active mechanics: {debug_info.get('active_mechanics', [])}")
            if debug_info.get('inverse_objects'):
                print(f"Inverse objects: {list(debug_info['inverse_objects'])}")
            if debug_info.get('remote_mappings'):
                print(f"Remote mappings: {debug_info['remote_mappings']}")

        try:
            entities = self.world_adapter.get_interactable_entities()
            furniture = [e["name"] for e in entities if e["type"] == "furniture"]
            objects = [e["name"] for e in entities if e["type"] == "object"]
            rooms = self.world_adapter.get_room_ids()

            print(f"\nRooms: {', '.join(rooms)}")
            print(f"Furniture ({len(furniture)}): {', '.join(furniture[:20])}...")
            print(f"Objects ({len(objects)}): {', '.join(objects)}")
        except Exception as e:
            print(f"Error getting entities: {e}")

    def _print_mechanics(self) -> None:
        """Print active mechanics."""
        print(f"\n{'='*50}")
        print("ACTIVE MECHANICS")
        print(f"{'='*50}")

        if self.game_manager:
            debug_info = self.game_manager.get_debug_info()
            for mech in debug_info.get('active_mechanics', []):
                print(f"  - {mech}")

            if debug_info.get('inverse_objects'):
                print(f"\nInverse state targets: {list(debug_info['inverse_objects'])}")
            if debug_info.get('remote_mappings'):
                print("Remote control mappings:")
                for trigger, (target, state) in debug_info['remote_mappings'].items():
                    print(f"  {trigger} -> {target} ({state})")
            if debug_info.get('interaction_counts'):
                print(f"Counting state targets: {debug_info['interaction_counts']}")
        else:
            print("No game manager configured")

    def _print_history(self) -> None:
        """Print action history."""
        print(f"\n{'='*50}")
        print("ACTION HISTORY")
        print(f"{'='*50}")

        for record in self._action_history[-20:]:
            mode = record.get("mode", "?")
            print(f"  [{record['step']}] {record['agent']} ({mode}): {record['action']}")

    def _print_help(self) -> None:
        """Print help."""
        print(f"\n{'='*50}")
        print("HELP")
        print(f"{'='*50}")
        print("Actions:")
        print("  Navigate[target]  - Move to a location")
        print("  Open[target]      - Open furniture")
        print("  Close[target]     - Close furniture")
        print("  Pick[target]      - Pick up object")
        print("  Place[target]     - Place held object")
        print("  Use[target]       - Use/interact with object")
        print("  Inspect[target]   - Examine object")
        print("  Communicate[msg]  - Send message to teammate")
        print("\nCommands:")
        print("  status    - Show full world status")
        print("  mechanics - Show active mechanics")
        print("  history   - Show action history")
        print("  skip      - Skip this agent's turn")
        print("  quit      - Exit and save")

    def _save_outputs(self) -> None:
        """Save video and planner log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*50}")
        print("Finishing...")

        # Save video
        self.save_video(f"human_test_{timestamp}")

        # Build planner log
        log_data = {
            "mode": "human_test",
            "task": self.task_info.get("title", "Human Test") if self.task_info else "Human Test",
            "mechanics_active": self.game_manager.get_state().active_mechanics if self.game_manager else [],
            "total_steps": self._step_count,
            "human_agents": list(self.human_agents),
            "llm_agents": [f"agent_{uid}" for uid in self.planners.keys()],
            "action_history": self._action_history,
        }

        self.save_planner_log(log_data)

        # Save prompts from LLM agents
        prompts = {}
        traces = {}
        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"
            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                prompts[agent_id] = planner.curr_prompt
                traces[agent_id] = planner.curr_prompt

        if prompts:
            self.save_prompts(prompts, traces)

        print("Done!")
