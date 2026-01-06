"""
Unified benchmark runner for EMTOM evaluation.

Supports both LLM and human-controlled agents. Each agent can be configured
as human or LLM via the human_agents parameter.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from omegaconf import DictConfig

from .base import EMTOMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    from emtom.task_gen import GeneratedTask


class BenchmarkRunner(EMTOMBaseRunner):
    """
    Unified runner for EMTOM benchmark evaluation.

    Supports:
    - All LLM agents (default)
    - All human agents (human_agents=["agent_0", "agent_1", ...])
    - Mixed mode (some human, some LLM)
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.planners: Dict[int, Any] = {}
        self.task: Optional["GeneratedTask"] = None
        self.human_agents: Set[str] = set()
        self._completed_subtasks: Set[str] = set()

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        task: Optional["GeneratedTask"] = None,
        save_video: Optional[bool] = None,
        human_agents: Optional[List[str]] = None,
    ) -> None:
        """
        Setup benchmark runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Task data with mechanics/bindings
            output_dir: Output directory
            task: Optional GeneratedTask object
            save_video: Whether to save video
            human_agents: List of agent IDs to be human-controlled (e.g., ["agent_0"])
                         If None, all agents are LLM-controlled.
        """
        self.task = task

        # If task is provided, always extract task_data from it (items, mechanics, locked_containers)
        # This ensures items are never missing when a GeneratedTask is available
        if task:
            task_data = self._task_to_mechanics_dict(task)

        agent_actions = task.agent_actions if task else None

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video)

        # Set human agents
        if human_agents:
            self.human_agents = set(human_agents)
        else:
            self.human_agents = set()

        # Setup planners only for LLM agents
        llm_agent_uids = [
            uid for uid in self.agents.keys()
            if f"agent_{uid}" not in self.human_agents
        ]
        self._setup_planners(llm_agent_uids)

        if self.human_agents:
            print(f"[BenchmarkRunner] Human agents: {self.human_agents}")
            print(f"[BenchmarkRunner] LLM agents: {set(f'agent_{uid}' for uid in llm_agent_uids)}")

    def _task_to_mechanics_dict(self, task: "GeneratedTask") -> Dict[str, Any]:
        """Convert GeneratedTask to task data for GameStateManager."""
        result = {}
        if task.mechanic_bindings:
            result["mechanics"] = [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        if task.items:
            result["items"] = task.items
        if task.locked_containers:
            result["locked_containers"] = task.locked_containers
        return result

    def _setup_planners(self, agent_uids: List[int]) -> None:
        """Initialize LLM planners for specified agents."""
        if not agent_uids:
            return

        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from emtom.planner import EmtomPlanner

        if not hasattr(self.config, 'evaluation') or not hasattr(self.config.evaluation, 'agents'):
            print("[BenchmarkRunner] Warning: No agents in config for planner setup")
            return

        agent_confs = list(self.config.evaluation.agents.values())

        for uid in agent_uids:
            if uid not in self.agents:
                continue
            if uid >= len(agent_confs):
                continue

            agent_conf = agent_confs[uid]
            if not hasattr(agent_conf, 'planner'):
                print(f"[BenchmarkRunner] Warning: No planner config for agent_{uid}")
                continue

            planner_conf = OmegaConf.to_container(agent_conf.planner, resolve=True)
            if '_target_' in planner_conf and 'LLMPlanner' in planner_conf['_target_']:
                planner_conf['_target_'] = 'emtom.planner.EmtomPlanner'

            planner = instantiate(OmegaConf.create(planner_conf))
            planner = planner(env_interface=self.env_interface)
            planner.agents = [self.agents[uid]]
            self.planners[uid] = planner
            print(f"[BenchmarkRunner] Created planner for agent_{uid}")

    def run(
        self,
        instruction: Dict[str, str],
        max_steps: int = 20000,
        max_turns: int = 20,
    ) -> Dict[str, Any]:
        """
        Run benchmark task. Each turn executes all agents' actions to completion.

        Args:
            instruction: Per-agent instruction dict (agent_id -> instruction)
            max_steps: Maximum turns (legacy param name, same as max_turns)
            max_turns: Maximum turns (each turn = all agents complete one action each)

        Returns:
            Results dict with steps, turns, done, action_history, evaluation
        """
        task_title = self.task.title if self.task else "Unknown Task"
        n_agents = len(self.agents)
        has_humans = bool(self.human_agents)

        print(f"\n{'='*60}", flush=True)
        print(f"BENCHMARK: {task_title}", flush=True)
        print(f"Agents: {n_agents} | Max turns: {max_turns}", flush=True)
        if has_humans:
            print(f"Human: {list(self.human_agents)} | LLM: {[f'agent_{u}' for u in self.planners.keys()]}", flush=True)
        print(f"{'='*60}\n", flush=True)

        if has_humans:
            self._print_controls()

        observations = self.get_observations()
        self.record_frame(observations)

        done = False
        turn_count = 0
        agents_done: Set[int] = set()

        # Reset step count for this run
        self._step_count = 0

        # Main loop - each iteration is one step where all agents act
        while self._step_count < max_steps and not done and not self._episode_done:
            # Check turn limit before processing
            if max_turns and turn_count >= max_turns:
                print(f"\n[Benchmark] Reached max turns ({max_turns})", flush=True)
                break

            self._step_count += 1
            turn_count += 1

            print(f"\n{'='*60}", flush=True)
            print(f"TURN {turn_count}", flush=True)
            print(f"{'='*60}", flush=True)

            world_graph = self.get_world_graph()

            for uid in sorted(self.agents.keys()):
                if uid in agents_done:
                    continue

                agent_id = f"agent_{uid}"
                agent_instruction = instruction.get(agent_id, instruction.get(str(uid), ""))

                # Print agent status for human agents
                if agent_id in self.human_agents:
                    self._print_agent_status(uid)

                try:
                    if agent_id in self.human_agents:
                        # Human agent - get input from CLI
                        action, agent_done = self._get_human_action(agent_id)
                        if agent_done:
                            done = True
                            break
                        if action == "skip":
                            continue

                        # Execute action
                        if action:
                            result = self.execute_parsed_action(uid, action)
                            observation = result.get("observation", "")
                            print(f"Agent_{uid}_Observation: {observation}", flush=True)

                            self._action_history.append({
                                "sim_step": self._step_count,
                                "turn": turn_count,
                                "agent": agent_id,
                                "action": action,
                                "result": observation,
                                "mode": "human",
                            })

                            self._check_subtasks()

                            # Check if action granted items that provide new tools
                            self.check_and_inject_item_tools(uid)
                    else:
                        # LLM agent - use planner, execute action to completion
                        if uid not in self.planners:
                            continue

                        planner = self.planners[uid]

                        # Execute action to completion by looping get_next_action()
                        # The planner handles state internally:
                        # - First call: replan_required=True, makes LLM call, starts action
                        # - Subsequent calls: replan_required=False, continues action
                        # - When action completes: response returned, replan_required=True
                        action_done = False
                        skill_steps = 0
                        max_skill_steps = 1500
                        high_level_action = None
                        response = ""
                        planner_done = False

                        while not action_done and skill_steps < max_skill_steps:
                            # Get next action step from planner
                            low_level_actions, planner_info, planner_done = planner.get_next_action(
                                agent_instruction, observations, world_graph
                            )

                            # On first iteration (replan), extract the action
                            # EmtomPlanner prints Thought/Action during replan
                            if high_level_action is None:
                                high_level_action = self._extract_high_level_action(planner_info, uid)
                                if not high_level_action:
                                    break  # No action returned

                            # Check if action completed (response returned)
                            responses_dict = planner_info.get("responses", {})
                            if responses_dict.get(uid):
                                response = responses_dict[uid]
                                action_done = True

                            # Step the environment
                            if low_level_actions:
                                try:
                                    obs, reward, done_flag, step_info = self.env_interface.step(low_level_actions)
                                    observations = self.env_interface.parse_observations(obs)
                                    self.record_frame(observations)
                                except Exception as e:
                                    print(f"[Agent {uid}] Step error: {e}", flush=True)
                                    break

                            skill_steps += 1

                            # Also break if planner says done
                            if planner_done:
                                break

                        if high_level_action:
                            # Log result
                            print(f"Agent_{uid}_Observation: {response}", flush=True)
                            print(f"  ({skill_steps} steps)", flush=True)

                            self._action_history.append({
                                "sim_step": self._step_count,
                                "turn": turn_count,
                                "agent": agent_id,
                                "action": high_level_action,
                                "result": response,
                                "mode": "llm",
                                "skill_steps": skill_steps,
                            })

                            self._check_subtasks()

                            # Check if action granted items that provide new tools
                            self.check_and_inject_item_tools(uid)

                        # Update world graph for next agent
                        world_graph = self.get_world_graph()

                        if planner_done:
                            agents_done.add(uid)
                            print(f"[Agent {uid} DONE]", flush=True)

                except AssertionError as e:
                    if "Episode over" in str(e) or "call reset before calling step" in str(e):
                        print(f"\n[Benchmark] Episode ended at step {self._step_count}", flush=True)
                        self._episode_done = True
                        break
                    raise
                except Exception as e:
                    print(f"[Agent {uid} ERROR] {e}", flush=True)
                    continue

                # Check task completion after each action
                eval_result = self._check_task_completion()
                if eval_result and eval_result.get("success"):
                    print(f"\n{'='*60}", flush=True)
                    print("TASK COMPLETE!", flush=True)
                    print(f"{'='*60}", flush=True)
                    done = True
                    break

            # Check if all agents done
            if len(agents_done) == len(self.agents):
                done = True

            if not self._episode_done and not done:
                observations = self.get_observations()
                self.record_frame(observations)

        print(f"\n[Benchmark] Finished: steps={self._step_count}, turns={turn_count}, done={done}", flush=True)

        # Final evaluation
        evaluation = self._check_task_completion() or {}

        # Save outputs
        self._save_outputs(instruction, evaluation, turn_count)

        return {
            "steps": self._step_count,
            "turns": turn_count,
            "done": done,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
            "evaluation": evaluation,
            "success": evaluation.get("success", False),
        }

    # -------------------------------------------------------------------------
    # Human input methods
    # -------------------------------------------------------------------------

    def _get_human_action(self, agent_id: str) -> tuple:
        """
        Get action from human via CLI.

        Returns:
            (action_string, should_quit)
        """
        while True:
            try:
                user_input = input(f"{agent_id}> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None, True

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "q", "exit"):
                return None, True
            if cmd == "skip":
                return "skip", False
            if cmd == "status":
                self._print_full_status()
                continue
            if cmd == "subtasks":
                self._print_subtasks()
                continue
            if cmd == "mechanics":
                self._print_mechanics()
                continue
            if cmd == "history":
                self._print_history()
                continue
            if cmd == "world":
                self._print_world_description(agent_id)
                continue
            if cmd == "prompt":
                self._print_llm_prompt(agent_id)
                continue
            if cmd == "help":
                self._print_help()
                continue

            # Parse action
            action = self._parse_action(user_input)
            if action:
                return action, False

            print(f"Invalid action: {user_input}")
            print("Format: ActionName[target] (e.g., Navigate[kitchen_1])")

    def _parse_action(self, text: str) -> Optional[str]:
        """Parse user input into action string."""
        # Allow empty brackets for actions like Wait[]
        match = re.match(r'(\w+)\[([^\]]*)\]', text)
        if match:
            return text

        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            return f"{parts[0]}[{parts[1]}]"

        return None

    def _print_agent_status(self, uid: int) -> None:
        """Print status for an agent."""
        agent_id = f"agent_{uid}"
        mode = "human" if agent_id in self.human_agents else "LLM"

        try:
            room = self.world_adapter.get_agent_location(agent_id)
        except Exception:
            room = "unknown"

        print(f"\n--- {agent_id} ({mode}) in {room} ---", flush=True)

        # Show inventory
        if self.game_manager:
            inv_text = self.game_manager.get_inventory_text(agent_id)
            print(f"Inventory: {inv_text}", flush=True)

    def _print_controls(self) -> None:
        """Print control instructions."""
        print(f"{'='*60}")
        print("CONTROLS")
        print(f"{'='*60}")

        if self.task and self.task.agent_actions:
            for agent_id in sorted(self.task.agent_actions.keys()):
                actions = self.task.agent_actions[agent_id]
                mode = "human" if agent_id in self.human_agents else "LLM"
                print(f"  {agent_id} ({mode}): {', '.join(actions)}")
        else:
            print("Actions: Navigate, Open, Close, Pick, Place, Search, UseItem, Communicate")

        print(f"\nCommands: status, world, prompt, subtasks, mechanics, history, skip, quit, help")
        print(f"{'='*60}\n")

    def _print_full_status(self) -> None:
        """Print full world status."""
        print(f"\n{'='*50}")
        print("WORLD STATUS")
        print(f"{'='*50}")

        if self.game_manager:
            debug_info = self.game_manager.get_debug_info()
            print(f"Active mechanics: {debug_info.get('active_mechanics', [])}")

        try:
            entities = self.world_adapter.get_interactable_entities()
            furniture = [e["name"] for e in entities if e["type"] == "furniture"]
            objects = [e["name"] for e in entities if e["type"] == "object"]
            rooms = self.world_adapter.get_room_ids()

            print(f"\nRooms: {', '.join(rooms)}")
            print(f"Furniture ({len(furniture)}): {', '.join(furniture[:15])}...")
            print(f"Objects ({len(objects)}): {', '.join(objects)}")
        except Exception as e:
            print(f"Error: {e}")

    def _print_subtasks(self) -> None:
        """Print subtask status."""
        if not self.task or not self.task.subtasks:
            print("No subtasks defined.")
            return

        subtasks = self.task.subtasks
        print(f"\n{'='*50}")
        print(f"SUBTASKS ({len(self._completed_subtasks)}/{len(subtasks)})")
        print(f"{'='*50}")

        for subtask in subtasks:
            subtask_id = subtask.id if hasattr(subtask, 'id') else subtask.get("id", "")
            desc = subtask.description if hasattr(subtask, 'description') else subtask.get("description", "")

            if subtask_id in self._completed_subtasks:
                status = "✓"
            else:
                depends = subtask.depends_on if hasattr(subtask, 'depends_on') else subtask.get("depends_on", [])
                deps_met = all(d in self._completed_subtasks for d in depends)
                status = "○" if deps_met else "◌"

            print(f"  {status} {subtask_id}: {desc}")

    def _print_mechanics(self) -> None:
        """Print active mechanics."""
        print(f"\n{'='*50}")
        print("ACTIVE MECHANICS")
        print(f"{'='*50}")

        if self.game_manager:
            debug_info = self.game_manager.get_debug_info()
            for mech in debug_info.get('active_mechanics', []):
                print(f"  - {mech}")
        else:
            print("No mechanics configured")

    def _print_history(self) -> None:
        """Print action history."""
        print(f"\n{'='*50}")
        print("ACTION HISTORY")
        print(f"{'='*50}")

        for record in self._action_history[-20:]:
            mode = record.get("mode", "?")
            turn = record.get("turn", "?")
            print(f"  [T{turn}] {record['agent']} ({mode}): {record['action']}")

    def _print_help(self) -> None:
        """Print help."""
        from emtom.actions.registry import STANDARD_ACTIONS

        print(f"\n{'='*50}")
        print("HELP")
        print(f"{'='*50}")
        print("\nActions:")
        for name, desc in sorted(STANDARD_ACTIONS.items()):
            print(f"  {desc}")
        print("\nCommands:")
        print("  status    - Show world status (rooms, furniture, objects)")
        print("  world     - Show world description (what LLM sees in prompt)")
        print("  prompt    - Show full LLM prompt (saves to file)")
        print("  subtasks  - Show subtask progress")
        print("  mechanics - Show active mechanics")
        print("  history   - Show action history")
        print("  skip      - Skip this agent's turn")
        print("  quit      - Exit and save")

    def _print_world_description(self, agent_id: str) -> None:
        """Print world description matching what LLM sees in its prompt."""
        from habitat_llm.llm.instruct.utils import get_world_descr

        uid = int(agent_id.split("_")[1])

        print(f"\n{'='*60}")
        print(f"WORLD DESCRIPTION (what LLM sees for {agent_id})")
        print(f"{'='*60}")

        try:
            world_graph = self.get_world_graph()
            if uid in world_graph:
                wg = world_graph[uid]
                world_desc = get_world_descr(
                    wg,
                    agent_uid=uid,
                    add_state_info=True,
                    include_room_name=True,
                    centralized=True,
                )
                print(world_desc)
            else:
                print(f"World graph not available for {agent_id}")
        except Exception as e:
            print(f"Error getting world description: {e}")

        # Also show inventory
        if self.game_manager:
            print(f"\nInventory: {self.game_manager.get_inventory_text(agent_id)}")

        print(f"{'='*60}")

    def _print_llm_prompt(self, agent_id: str) -> None:
        """Print and save the full LLM prompt for debugging."""
        uid = int(agent_id.split("_")[1])

        print(f"\n{'='*60}")
        print(f"LLM PROMPT for {agent_id}")
        print(f"{'='*60}")

        if uid in self.planners:
            planner = self.planners[uid]
            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                prompt = planner.curr_prompt
                print(prompt)

                # Save to file
                if self.output_dir:
                    prompt_file = Path(self.output_dir) / f"prompt_{agent_id}.txt"
                    prompt_file.write_text(prompt)
                    print(f"\n[Saved to {prompt_file}]")
            else:
                print(f"No prompt available yet for {agent_id}")
                print("(Prompt is built after first LLM call)")
        else:
            print(f"No LLM planner for {agent_id} (human-controlled)")
            print("\nTo see what LLM would see, use 'world' command")

        print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # Task evaluation
    # -------------------------------------------------------------------------

    def _check_task_completion(self) -> Optional[Dict[str, Any]]:
        """
        Check if task is complete using smart success evaluation.

        Required subtasks are evaluated based on their dependencies:
        - Required subtasks with required deps → must be latched via DAG
        - Required subtasks with optional-only deps → check directly
        - Required subtasks with no deps → check directly
        """
        if not self.task or not self.task.subtasks:
            return None

        # Get all required subtasks
        required_subtasks = [
            s for s in self.task.subtasks
            if getattr(s, 'required', True)
        ]

        if not required_subtasks:
            return {"success": False, "error": "No required subtasks"}

        # Build lookup for required subtask IDs
        required_ids = {
            s.id if hasattr(s, 'id') else s.get("id", "")
            for s in required_subtasks
        }

        all_satisfied = True
        conditions_checked = []

        for subtask in required_subtasks:
            subtask_id = subtask.id if hasattr(subtask, 'id') else subtask.get("id", "")
            depends_on = subtask.depends_on if hasattr(subtask, 'depends_on') else subtask.get("depends_on", [])
            success_condition = subtask.success_condition if hasattr(subtask, 'success_condition') else subtask.get("success_condition")

            if not success_condition:
                continue

            # Check if this subtask has any REQUIRED dependencies
            required_deps = [dep for dep in depends_on if dep in required_ids]

            if required_deps:
                # Has required dependencies → must be latched via DAG
                if subtask_id not in self._completed_subtasks:
                    all_satisfied = False
            else:
                # No required dependencies → check directly
                result = self.evaluate_task({"required_states": [success_condition]})
                if not result or not result.get("success"):
                    all_satisfied = False

            conditions_checked.append(subtask_id)

        # Calculate completion percentages
        total_subtasks = len(self.task.subtasks)
        required_count = len(required_subtasks)
        completed_required = len([s for s in self._completed_subtasks if s in required_ids])

        return {
            "success": all_satisfied,
            "conditions_checked": conditions_checked,
            "completed_subtasks": list(self._completed_subtasks),
            "total_subtasks": total_subtasks,
            "required_subtasks": required_count,
            "completed_required": completed_required,
            "percent_complete": len(self._completed_subtasks) / total_subtasks if total_subtasks else 0.0,
            "percent_required_complete": completed_required / required_count if required_count else 0.0,
        }

    def _check_subtasks(self) -> List[str]:
        """Check subtasks and return newly completed IDs."""
        if not self.task or not self.task.subtasks:
            return []

        subtasks = self.task.subtasks
        newly_completed = []

        for subtask in subtasks:
            subtask_id = subtask.id if hasattr(subtask, 'id') else subtask.get("id", "")
            if not subtask_id or subtask_id in self._completed_subtasks:
                continue

            depends_on = subtask.depends_on if hasattr(subtask, 'depends_on') else subtask.get("depends_on", [])
            if not all(dep in self._completed_subtasks for dep in depends_on):
                continue

            success_condition = subtask.success_condition if hasattr(subtask, 'success_condition') else subtask.get("success_condition")
            if not success_condition:
                continue

            result = self.evaluate_task({"required_states": [success_condition]})
            if result and result.get("success"):
                self._completed_subtasks.add(subtask_id)
                newly_completed.append(subtask_id)

                desc = subtask.description if hasattr(subtask, 'description') else subtask.get("description", subtask_id)
                print(f"\n{'─'*50}", flush=True)
                print(f"✓ SUBTASK COMPLETE: {subtask_id}", flush=True)
                print(f"  {desc}", flush=True)
                print(f"  Progress: {len(self._completed_subtasks)}/{len(subtasks)}", flush=True)
                print(f"{'─'*50}", flush=True)

        return newly_completed

    def _extract_high_level_action(self, planner_info: Dict, uid: int) -> Optional[str]:
        """Extract high-level action from planner info."""
        ha = planner_info.get("high_level_actions", {})
        if uid in ha:
            action_tuple = ha[uid]
            if action_tuple and action_tuple[0]:
                return f"{action_tuple[0]}[{action_tuple[1]}]"
        return None

    # -------------------------------------------------------------------------
    # Output saving
    # -------------------------------------------------------------------------

    def _save_outputs(
        self,
        instruction: Dict[str, str],
        evaluation: Dict[str, Any],
        turn_count: int,
    ) -> None:
        """Save video, planner log, and prompts."""
        task_id = self.task.task_id if self.task else "unknown"
        sim_steps = self._step_count

        # Save video
        mode = "human" if self.human_agents else "benchmark"
        self.save_video(f"{mode}_{task_id}_{sim_steps}")

        # Build log data
        log_data = {
            "task_id": task_id,
            "task_title": self.task.title if self.task else "Unknown",
            "instruction": instruction,
            "mechanics_active": self.game_manager.get_state().active_mechanics if self.game_manager else [],
            "sim_steps": sim_steps,
            "turns": turn_count,
            "episode_over": self._episode_done,
            "human_agents": list(self.human_agents),
            "llm_agents": [f"agent_{uid}" for uid in self.planners.keys()],
            "action_history": self._action_history,
        }

        if evaluation:
            log_data["evaluation"] = evaluation
            log_data["percent_complete"] = evaluation.get("percent_complete", 0.0)
            log_data["success"] = evaluation.get("success", False)

        if self.task and self.task.mechanic_bindings:
            log_data["mechanic_bindings"] = [b.to_dict() for b in self.task.mechanic_bindings]

        self.save_planner_log(log_data)

        # Save prompts from LLM planners
        self._save_planner_prompts()

    def _save_planner_prompts(self) -> None:
        """Save prompts from LLM planners."""
        prompts = {}
        traces = {}

        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"

            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                prompts[agent_id] = planner.curr_prompt

                prompt = planner.curr_prompt
                task_marker = "Task:"
                if task_marker in prompt:
                    traces[agent_id] = prompt[prompt.find(task_marker):]
                else:
                    traces[agent_id] = prompt

        if prompts:
            self.save_prompts(prompts, traces)

    def get_planner_traces(self) -> Dict[str, str]:
        """Get conversation traces from planners."""
        traces = {}
        for uid, planner in self.planners.items():
            agent_id = f"agent_{uid}"
            if hasattr(planner, 'curr_prompt') and planner.curr_prompt:
                prompt = planner.curr_prompt
                task_marker = "Task:"
                if task_marker in prompt:
                    traces[agent_id] = prompt[prompt.find(task_marker):]
                else:
                    traces[agent_id] = prompt
        return traces


def task_to_instruction(task: "GeneratedTask") -> Dict[str, str]:
    """Convert GeneratedTask to per-agent instructions."""
    instructions = {}

    for agent_id in task.agent_roles.keys():
        parts = []

        # Header with agent identity
        parts.append(f"You are {agent_id.replace('_', ' ').title()}. Given the following task, take a sequence of actions to solve and complete the task at hand.")
        parts.append("")

        # Task description
        if task.task:
            parts.append(f"[Task]: {task.task}")
            parts.append("")

        # Known Information - what this agent knows
        secrets = task.agent_secrets.get(agent_id, [])
        if secrets:
            parts.append("[Known Information]:")
            for s in secrets:
                parts.append(f"- {s}")

        instructions[agent_id] = "\n".join(parts)

    return instructions
