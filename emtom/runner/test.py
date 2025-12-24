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
        self._completed_subtasks: Set[str] = set()  # Track completed subtask IDs

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

                    # Check for subtask completion after each action
                    self._check_subtasks()

                    # Check for overall task completion
                    eval_result = self._check_task_completion()
                    if eval_result and eval_result.get("success"):
                        print(f"\n{'='*60}")
                        print("🎉 TASK COMPLETE! 🎉")
                        print(f"{'='*60}")
                        done = True
                        break

            if not done and not self._episode_done:
                observations = self.get_observations()
                self.record_frame(observations)

        # Save outputs
        self._save_outputs()

        # Final evaluation
        eval_result = self._check_task_completion()

        return {
            "steps": self._step_count,
            "action_history": self._action_history,
            "evaluation": eval_result,
            "success": eval_result.get("success", False) if eval_result else False,
        }

    def _check_task_completion(self) -> Optional[Dict[str, Any]]:
        """
        Check if the task is complete using PARTNR predicates.

        Returns:
            Evaluation result dict or None if no success_condition defined
        """
        if not self.task_info:
            return None

        success_condition = self.task_info.get("success_condition")
        if not success_condition:
            return None

        return self.evaluate_task(success_condition)

    def _check_subtasks(self) -> List[str]:
        """
        Check all subtasks and return list of newly completed subtask IDs.

        Only evaluates subtasks whose dependencies (depends_on) are all completed.
        Logs completion to console when a subtask is newly completed.
        """
        if not self.task_info:
            return []

        subtasks = self.task_info.get("subtasks", [])
        if not subtasks:
            return []

        newly_completed = []

        for subtask in subtasks:
            subtask_id = subtask.get("id", "")
            if not subtask_id or subtask_id in self._completed_subtasks:
                continue

            # Check if all dependencies are completed
            depends_on = subtask.get("depends_on", [])
            if not all(dep in self._completed_subtasks for dep in depends_on):
                # Dependencies not met, skip this subtask
                continue

            success_condition = subtask.get("success_condition")
            if not success_condition:
                continue

            # Check this subtask's condition
            result = self.evaluate_task({"required_states": [success_condition]})
            if result and result.get("success"):
                self._completed_subtasks.add(subtask_id)
                newly_completed.append(subtask_id)

                # Log the completion
                desc = subtask.get("description", subtask_id)
                print(f"\n{'─'*50}")
                print(f"✓ SUBTASK COMPLETE: {subtask_id}")
                print(f"  {desc}")
                print(f"  Progress: {len(self._completed_subtasks)}/{len(subtasks)} subtasks")
                print(f"{'─'*50}")

        return newly_completed

    def _print_subtasks(self) -> None:
        """Print current subtask status."""
        if not self.task_info:
            print("No task info available.")
            return

        subtasks = self.task_info.get("subtasks", [])
        if not subtasks:
            print("No subtasks defined.")
            return

        print(f"\n{'='*60}")
        print(f"SUBTASKS ({len(self._completed_subtasks)}/{len(subtasks)} complete)")
        print(f"{'='*60}")

        for subtask in subtasks:
            subtask_id = subtask.get("id", "unknown")
            desc = subtask.get("description", "")
            assigned = subtask.get("assigned_agent", "")
            depends = subtask.get("depends_on", [])

            if subtask_id in self._completed_subtasks:
                status = "✓"
            else:
                # Check if dependencies are met
                deps_met = all(d in self._completed_subtasks for d in depends)
                status = "○" if deps_met else "◌"

            agent_info = f" [{assigned}]" if assigned else ""
            print(f"  {status} {subtask_id}{agent_info}: {desc}")

        print(f"{'='*60}")

    def _print_header(self) -> None:
        """Print task header - minimal since run_human_test.py already prints details."""
        # Task info is already printed by run_human_test.py, so just print a separator
        pass

    def _print_controls(self) -> None:
        """Print control instructions with per-agent actions."""
        print(f"\n{'='*60}")
        print("CONTROLS")
        print(f"{'='*60}")

        # Show per-agent actions if available
        agent_actions = self.task_info.get("agent_actions") if self.task_info else None
        if agent_actions:
            for agent_id in sorted(agent_actions.keys()):
                actions = agent_actions[agent_id]
                mode = "human" if agent_id in self.human_agents else "LLM"
                print(f"  {agent_id} ({mode}): {', '.join(actions)}")
        else:
            print("Actions: Navigate[target], Open[target], Close[target], Pick[target], Place[target]")
            print("         Use[target], Inspect[target], Communicate[message]")

        print(f"\nCommands: status, subtasks, mechanics, history, skip, quit, help")
        print(f"{'='*60}")

    def _print_agent_status(self, uid: int) -> None:
        """Print status for an agent (location and inventory only)."""
        agent_id = f"agent_{uid}"
        mode = "human" if agent_id in self.human_agents else "LLM"

        # Get agent location
        try:
            room = self.world_adapter.get_agent_location(agent_id)
        except Exception:
            room = "unknown"

        print(f"\n--- {agent_id} ({mode}) in {room} ---")

        # Show inventory only (use FindObjectTool/FindRoomTool for other info)
        try:
            state = self.game_manager.get_state()
            inventory = state.agent_inventory.get(agent_id, set())
            if inventory:
                # Get item names from definitions
                inv_names = []
                for item_id in inventory:
                    item_def = state.item_definitions.get(item_id)
                    if item_def:
                        inv_names.append(item_def.get("name", item_id))
                    else:
                        inv_names.append(item_id)
                print(f"Inventory: {', '.join(inv_names)}")
            else:
                print("Inventory: (empty)")
        except Exception:
            pass

        # Hint about using tools
        print("(Use FindObjectTool, FindRoomTool to explore)")

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

            if cmd_lower == "subtasks":
                self._print_subtasks()
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

    def _get_action_description(self, action_name: str) -> str:
        """Get description for an action from registry."""
        from emtom.actions.registry import STANDARD_ACTIONS, ActionRegistry

        # Check standard actions first
        if action_name in STANDARD_ACTIONS:
            return STANDARD_ACTIONS[action_name]

        # Check custom actions
        if ActionRegistry.is_registered(action_name):
            info = ActionRegistry.get_info(action_name)
            return f"{action_name}: {info['description']}"

        return f"{action_name}: (no description available)"

    def _print_help(self) -> None:
        """Print help with per-agent actions."""
        from emtom.actions.registry import STANDARD_ACTIONS, ActionRegistry

        print(f"\n{'='*50}")
        print("HELP")
        print(f"{'='*50}")

        agent_actions = self.task_info.get("agent_actions") if self.task_info else None

        if agent_actions:
            # Show per-agent actions with descriptions
            for agent_id in sorted(agent_actions.keys()):
                actions = agent_actions[agent_id]
                mode = "human" if agent_id in self.human_agents else "LLM"
                print(f"\n{agent_id} ({mode}) Actions:")
                for action_name in actions:
                    desc = self._get_action_description(action_name)
                    print(f"  {desc}")
        else:
            # Fallback: show all standard actions
            print("Actions:")
            for name, desc in sorted(STANDARD_ACTIONS.items()):
                print(f"  {desc}")
        print("\nCommands:")
        print("  status    - Show full world status")
        print("  subtasks  - Show subtask progress")
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
