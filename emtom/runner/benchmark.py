"""
Unified benchmark runner for EMTOM evaluation.

Supports both LLM and human-controlled agents. Each agent can be configured
as human or LLM via the human_agents parameter.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

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
        message_targets = task.message_targets if task else None

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video, message_targets=message_targets)

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
        if task.active_mechanics:
            result["active_mechanics"] = task.active_mechanics
        if task.mechanic_bindings:
            result["mechanic_bindings"] = [
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
        self.record_frame(observations, turn=0)

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

            # =====================================================================
            # Phase 1: Handle human agents first (sequential)
            # =====================================================================
            for uid in sorted(self.agents.keys()):
                if uid in agents_done:
                    continue

                agent_id = f"agent_{uid}"
                if agent_id not in self.human_agents:
                    continue

                self._print_agent_status(uid)

                try:
                    action, agent_quit = self._get_human_action(agent_id)
                    if agent_quit:
                        done = True
                        break
                    if action == "skip":
                        continue

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

                        newly_completed = self._check_subtasks()
                        if newly_completed:
                            self._action_history.append({
                                "turn": turn_count,
                                "type": "subtask_completion",
                                "subtasks_completed": newly_completed,
                            })
                        self.check_and_inject_item_tools(uid)

                except AssertionError as e:
                    if "Episode over" in str(e) or "call reset before calling step" in str(e):
                        print(f"\n[Benchmark] Episode ended at step {self._step_count}", flush=True)
                        self._episode_done = True
                        break
                    raise
                except Exception as e:
                    print(f"[Agent {uid} ERROR] {e}", flush=True)
                    continue

            if done or self._episode_done:
                break

            # =====================================================================
            # Phase 2: Plan for all LLM agents (get high-level actions)
            # Buffer messages during planning so they're only visible next turn
            # =====================================================================
            llm_agent_state: Dict[int, Dict[str, Any]] = {}
            max_skill_steps = 1500

            # Buffer messages sent during this turn - they should only be visible next turn
            message_buffer: List[tuple] = []
            original_post_message = self.env_interface.post_agent_message

            def buffered_post_message(sender_uid: int, message: str, target_uids=None) -> None:
                message_buffer.append((sender_uid, message, target_uids))

            self.env_interface.post_agent_message = buffered_post_message

            for uid in sorted(self.agents.keys()):
                if uid in agents_done:
                    continue
                agent_id = f"agent_{uid}"
                if agent_id in self.human_agents:
                    continue
                if uid not in self.planners:
                    continue

                planner = self.planners[uid]
                agent_instruction = instruction.get(agent_id, instruction.get(str(uid), ""))

                try:
                    # First call triggers planning (LLM call)
                    low_level_actions, planner_info, planner_done = planner.get_next_action(
                        agent_instruction, observations, world_graph
                    )
                    high_level_action = self._extract_high_level_action(planner_info, uid)

                    if not high_level_action:
                        if planner_done:
                            agents_done.add(uid)
                            print(f"[Agent {uid} DONE]", flush=True)
                        continue

                    # Pre-check mechanics (block/transform) using current game state.
                    mech_result = None
                    orig_action_name, orig_target = self._parse_action_to_tuple(high_level_action)
                    action_target = orig_target if orig_target not in ("", "None") else None
                    actual_action_name = orig_action_name
                    actual_target = action_target

                    if self.game_manager:
                        from emtom.mechanics.handlers import apply_mechanics
                        mech_result = apply_mechanics(
                            orig_action_name, agent_id, action_target, self.game_manager.get_state()
                        )

                        # If blocked, skip execution and return mechanic observation
                        if mech_result.blocked:
                            llm_agent_state[uid] = {
                                'planner': planner,
                                'instruction': agent_instruction,
                                'high_level_action': high_level_action,
                                'low_level_actions': {},
                                'planner_info': planner_info,
                                'planner_done': planner_done,
                                'action_done': True,
                                'response': mech_result.observation,
                                'skill_steps': 0,
                                'mech_result': mech_result,
                                'orig_action': orig_action_name,
                                'orig_target': action_target,
                            }
                            continue

                        actual_action_name = mech_result.actual_action or orig_action_name
                        actual_target = mech_result.actual_target or action_target

                        # If mechanic transformed the action, recompute low-level action.
                        if actual_action_name != orig_action_name or actual_target != action_target:
                            agent = self.agents.get(uid)
                            obs = self.env_interface.get_observations()
                            low_level_action, response = agent.process_high_level_action(
                                actual_action_name, actual_target or "", obs
                            )
                            if low_level_action is None:
                                llm_agent_state[uid] = {
                                    'planner': planner,
                                    'instruction': agent_instruction,
                                    'high_level_action': f"{actual_action_name}[{actual_target or ''}]",
                                    'low_level_actions': {},
                                    'planner_info': planner_info,
                                    'planner_done': planner_done,
                                    'action_done': True,
                                    'response': response or "",
                                    'skill_steps': 0,
                                    'mech_result': mech_result,
                                    'orig_action': orig_action_name,
                                    'orig_target': action_target,
                                }
                                continue
                            low_level_actions = {uid: low_level_action}
                            high_level_action = f"{actual_action_name}[{actual_target or ''}]"

                    llm_agent_state[uid] = {
                        'planner': planner,
                        'instruction': agent_instruction,
                        'high_level_action': high_level_action,
                        'low_level_actions': low_level_actions,
                        'planner_info': planner_info,
                        'planner_done': planner_done,
                        'action_done': False,
                        'response': "",
                        'skill_steps': 0,
                        'mech_result': mech_result,
                        'orig_action': orig_action_name,
                        'orig_target': action_target,
                    }

                    # Check if action already completed in first call
                    responses_dict = planner_info.get("responses", {})
                    if responses_dict.get(uid):
                        llm_agent_state[uid]['response'] = responses_dict[uid]
                        llm_agent_state[uid]['action_done'] = True
                    elif not low_level_actions or uid not in low_level_actions:
                        # Perception tool (Communicate, Find*, etc.) completed
                        # instantly — no simulation steps needed. The planner may
                        # have cleared the response string (e.g. Communicate
                        # responses are set to "" to prevent replans), so
                        # reconstruct a sensible fallback.
                        response = responses_dict.get(uid, "")
                        if not response and orig_action_name == "Communicate":
                            response = "Message delivered."
                        llm_agent_state[uid]['response'] = response or "Executed."
                        llm_agent_state[uid]['action_done'] = True

                except AssertionError as e:
                    if "Episode over" in str(e) or "call reset before calling step" in str(e):
                        print(f"\n[Benchmark] Episode ended at step {self._step_count}", flush=True)
                        self._episode_done = True
                        break
                    raise
                except Exception as e:
                    print(f"[Agent {uid} ERROR during planning] {e}", flush=True)
                    continue

            # Restore original post_message and flush buffered messages to queues
            # These messages will be consumed at the start of NEXT turn
            self.env_interface.post_agent_message = original_post_message
            for sender_uid, message, target_uids in message_buffer:
                original_post_message(sender_uid, message, target_uids=target_uids)

            if self._episode_done:
                break

            # =====================================================================
            # Phase 3: Execute all LLM agents concurrently
            # =====================================================================
            total_skill_steps = 0
            surroundings: Dict[int, List[str]] = {}
            agents_passed: Dict[int, Dict[str, tuple]] = {}  # uid -> {agent_name -> (room, step)}

            while llm_agent_state and total_skill_steps < max_skill_steps:
                # Check if all agents are done
                all_done = all(state['action_done'] for state in llm_agent_state.values())
                if all_done:
                    break

                # Collect low-level actions from all active agents
                combined_low_level: Dict[int, Any] = {}
                for uid, state in llm_agent_state.items():
                    if state['action_done']:
                        continue
                    if state['low_level_actions'] and uid in state['low_level_actions']:
                        combined_low_level[uid] = state['low_level_actions'][uid]

                # Step environment with ALL agents at once
                if combined_low_level:
                    try:
                        obs, reward, done_flag, step_info = self.env_interface.step(combined_low_level)
                        observations = self.env_interface.parse_observations(obs)

                        # Record frame with ALL agents' current actions
                        action_tuples = {
                            uid: self._parse_action_to_tuple(state['high_level_action'])
                            for uid, state in llm_agent_state.items()
                        }
                        self.record_frame(observations, action_tuples, turn=turn_count)

                    except Exception as e:
                        print(f"[Concurrent step error] {e}", flush=True)
                        break

                total_skill_steps += 1

                # Get next low-level actions for each active agent
                for uid, state in llm_agent_state.items():
                    if state['action_done']:
                        continue

                    state['skill_steps'] += 1

                    # Capture surroundings every 30 frames
                    if state['skill_steps'] % 30 == 0:
                        snapshot = self._get_surroundings_description(uid, state['skill_steps'])
                        surroundings.setdefault(uid, []).append(snapshot)
                        surroundings[uid] = surroundings[uid][-3:]
                        # Track agents encountered in same room (first sighting only)
                        ap = agents_passed.setdefault(uid, {})
                        for agent_name, room in self._get_nearby_agents(uid):
                            if agent_name not in ap:
                                ap[agent_name] = (room, state['skill_steps'])

                    try:
                        low_level_actions, planner_info, planner_done = state['planner'].get_next_action(
                            state['instruction'], observations, world_graph
                        )
                        state['low_level_actions'] = low_level_actions
                        state['planner_info'] = planner_info
                        state['planner_done'] = planner_done

                        # Check if action completed
                        responses_dict = planner_info.get("responses", {})
                        if responses_dict.get(uid):
                            state['response'] = responses_dict[uid]
                            state['action_done'] = True

                        if planner_done:
                            state['action_done'] = True

                    except Exception as e:
                        print(f"[Agent {uid} step error] {e}", flush=True)
                        state['action_done'] = True

            # =====================================================================
            # Phase 4: Log results and check completion for all LLM agents
            # =====================================================================
            for uid, state in llm_agent_state.items():
                agent_id = f"agent_{uid}"
                high_level_action = state['high_level_action']
                response = state['response']
                skill_steps = state['skill_steps']

                # Append surroundings observations collected during motor skill
                response += self._format_surroundings(surroundings.get(uid, []), agents_passed.get(uid))

                # Apply mechanic state changes after successful execution
                mech_result = state.get('mech_result')
                if self.game_manager and mech_result and not getattr(mech_result, "blocked", False):
                    obs_text = response or ""
                    habitat_failed = any(
                        fail_phrase in obs_text.lower()
                        for fail_phrase in ["too far", "occluded", "failed to", "unexpected failure", "cannot"]
                    )
                    if not habitat_failed and mech_result.applies:
                        orig_action = state.get('orig_action')
                        orig_target = state.get('orig_target')
                        _, mechanic_result = self.game_manager.apply_action(
                            orig_action, agent_id, orig_target
                        )
                        self._sync_remote_effects_to_simulator(mechanic_result.effects)
                        if mech_result.observation:
                            response = f"{response} {mech_result.observation}".strip()

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

                # Update belief tracker with action results
                self._update_beliefs_for_action(agent_id, high_level_action, response)

                self.check_and_inject_item_tools(uid)

                if state['planner_done']:
                    agents_done.add(uid)
                    print(f"[Agent {uid} DONE]", flush=True)

            # Check subtasks and task completion after all agents acted
            newly_completed = self._check_subtasks()
            if newly_completed:
                self._action_history.append({
                    "turn": turn_count,
                    "type": "subtask_completion",
                    "subtasks_completed": newly_completed,
                })
            world_graph = self.get_world_graph()

            eval_result = self._check_task_completion()
            if eval_result and eval_result.get("success"):
                print(f"\n{'='*60}", flush=True)
                print("TASK COMPLETE!", flush=True)
                print(f"{'='*60}", flush=True)
                done = True

            # Check if all agents done
            if len(agents_done) == len(self.agents):
                done = True

            if not self._episode_done and not done:
                observations = self.get_observations()
                # Record end-of-turn frame (no actions displayed)
                self.record_frame(observations, turn=turn_count)

        print(f"\n[Benchmark] Finished: steps={self._step_count}, turns={turn_count}, done={done}", flush=True)

        # Final evaluation
        evaluation = self._check_task_completion() or {}

        # Communication metrics
        comm_metrics = None
        if self.task and self._action_history:
            try:
                from emtom.evaluation_comms import evaluate_communication
                comm_metrics = evaluate_communication(
                    self._action_history, self.task, model="gpt-5.2",
                )
                comm_dict = comm_metrics.to_dict()
                print(f"\n[Communication Metrics]", flush=True)
                print(f"  Leakage score: {comm_metrics.overall_leakage_score:.2f}", flush=True)
                print(f"  Efficiency score: {comm_metrics.overall_efficiency_score:.2f}", flush=True)
                print(f"  Overall: {comm_metrics.overall_score:.2f}", flush=True)
                if comm_metrics.efficiency_reasoning:
                    print(f"  Reasoning: {comm_metrics.efficiency_reasoning}", flush=True)
            except Exception as e:
                print(f"[Communication Metrics] Error: {e}", flush=True)
                comm_dict = None
        else:
            comm_dict = None

        # Save outputs
        self._save_outputs(instruction, evaluation, turn_count, comm_metrics=comm_dict)

        result = {
            "steps": self._step_count,
            "turns": turn_count,
            "done": done,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
            "evaluation": evaluation,
            "success": evaluation.get("success", False),
        }
        if comm_dict:
            result["communication_metrics"] = comm_dict
        return result

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
            if cmd in ("subtasks", "goals"):
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
            print("Actions: Navigate, Open, Close, Pick, Place, UseItem, Communicate")

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
        """Print PDDL goal status."""
        if not self.task:
            print("No task loaded.")
            return

        checker = getattr(self, '_pddl_checker', None)
        if checker is None:
            checker = self.task.get_pddl_goal_checker()
            self._pddl_checker = checker

        if checker is None:
            print("No PDDL goals defined.")
            return

        total = len(checker.conjuncts)
        completed = len(checker.completed)
        print(f"\n{'='*50}")
        print(f"PDDL GOALS ({completed}/{total})")
        print(f"{'='*50}")

        for i, conjunct in enumerate(checker.conjuncts):
            done = checker.is_conjunct_completed(i)
            status = "✓" if done else "○"
            owner = checker.get_owner(i) if hasattr(checker, 'get_owner') else ""
            owner_str = f" [{owner}]" if owner else ""
            print(f"  {status} {conjunct}{owner_str}")

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
        """Check if task is complete using PDDL goal evaluation."""
        if not self.task:
            return None
        return self._check_pddl_completion()

    def _check_pddl_completion(self) -> Dict[str, Any]:
        """Check task completion using PDDL goal checker."""
        checker = getattr(self, '_pddl_checker', None)
        if checker is None:
            checker = self.task.get_pddl_goal_checker()
            self._pddl_checker = checker

        if checker is None:
            return {"success": False, "error": "No PDDL goal"}

        category = getattr(self.task, "category", "cooperative")

        if category == "competitive":
            # Check team-owned conjuncts
            teams = checker.get_all_teams()
            team_status = {}
            team_progress = {}
            winner = None

            # Check shared (required) first
            required = checker.get_required_conjuncts()
            shared_ok = all(
                checker.is_conjunct_completed(checker.conjuncts.index(c))
                for c in required
            ) if required else True

            for team_id in teams:
                team_conj = checker.get_team_conjuncts(team_id)
                if not team_conj:
                    team_status[team_id] = shared_ok
                    team_progress[team_id] = 1.0 if shared_ok else 0.0
                    continue
                done = sum(1 for c in team_conj if checker.is_conjunct_completed(checker.conjuncts.index(c)))
                progress = done / len(team_conj)
                team_progress[team_id] = progress
                team_status[team_id] = shared_ok and (progress == 1.0)
                if team_status[team_id] and winner is None:
                    winner = team_id

            total = len(checker.conjuncts)
            completed = len(checker.completed)
            return {
                "success": winner is not None,
                "winner": winner,
                "team_status": team_status,
                "team_progress": team_progress,
                "completed_subtasks": list(checker.completed),
                "total_subtasks": total,
                "percent_complete": completed / total if total else 0.0,
            }

        elif category == "mixed":
            required = checker.get_required_conjuncts()
            required_done = sum(
                1 for c in required
                if checker.is_conjunct_completed(checker.conjuncts.index(c))
            )
            main_goal_success = (required_done == len(required)) if required else True

            agent_subgoal_status = {}
            for i in range(self.task.num_agents):
                agent_id = f"agent_{i}"
                agent_conj = checker.get_agent_conjuncts(agent_id)
                if agent_conj:
                    agent_subgoal_status[agent_id] = all(
                        checker.is_conjunct_completed(checker.conjuncts.index(c))
                        for c in agent_conj
                    )

            total = len(checker.conjuncts)
            completed = len(checker.completed)
            return {
                "success": main_goal_success,
                "main_goal_success": main_goal_success,
                "agent_subgoal_status": agent_subgoal_status,
                "completed_subtasks": list(checker.completed),
                "total_subtasks": total,
                "percent_complete": completed / total if total else 0.0,
            }

        else:
            # Cooperative: all conjuncts must be complete
            total = len(checker.conjuncts)
            completed = len(checker.completed)
            all_complete = completed == total
            return {
                "success": all_complete,
                "completed_subtasks": list(checker.completed),
                "total_subtasks": total,
                "percent_complete": completed / total if total else 0.0,
            }

    def _check_subtasks(self) -> List[str]:
        """Check PDDL goal conjuncts and return newly completed IDs."""
        if not self.task:
            return []
        return self._check_pddl_goals()

    def _get_belief_tracker(self):
        """Get or create the belief state tracker for epistemic goal evaluation."""
        if hasattr(self, '_belief_tracker'):
            return self._belief_tracker

        self._belief_tracker = None
        try:
            from emtom.pddl.belief_tracker import BeliefStateTracker
            from emtom.pddl.epistemic import ObservabilityModel

            # Build object-room mapping from world graph
            # Include BOTH furniture and small objects (for binary predicate tracking)
            object_rooms = {}
            world_graph = self.get_world_graph()
            if world_graph:
                try:
                    from habitat_llm.world_model.world_graph import Furniture, Object

                    room_map = world_graph.get_furniture_to_room_map()
                    furn_name_to_room = {}
                    for furn, room in room_map.items():
                        furn_name = getattr(furn, 'name', str(furn))
                        room_name = getattr(room, 'name', str(room))
                        object_rooms[furn_name] = room_name
                        furn_name_to_room[furn_name] = room_name

                    for room in world_graph.get_all_rooms():
                        room_name = getattr(room, 'name', str(room))
                        object_rooms[room_name] = room_name

                    # Also map small objects to their room via their furniture parent
                    for obj_node in world_graph.get_all_objects():
                        obj_name = getattr(obj_node, 'name', str(obj_node))
                        for neighbor in world_graph.graph[obj_node]:
                            if isinstance(neighbor, Furniture):
                                furn_name = getattr(neighbor, 'name', str(neighbor))
                                if furn_name in furn_name_to_room:
                                    object_rooms[obj_name] = furn_name_to_room[furn_name]
                                break
                except Exception:
                    pass

            observability = None
            if self.task:
                observability = ObservabilityModel.from_task(self.task)

            self._belief_tracker = BeliefStateTracker.from_scene_and_observability(
                object_rooms=object_rooms,
                observability=observability,
                num_agents=getattr(self.task, 'num_agents', 2),
            )
        except Exception as e:
            print(f"[Benchmark] Could not create belief tracker: {e}", flush=True)

        return self._belief_tracker

    def _update_beliefs_for_action(
        self, agent_id: str, action: str, result: str
    ) -> None:
        """Update belief tracker after an action completes."""
        tracker = self._get_belief_tracker()
        if tracker is None:
            return

        # Parse action name and target
        action_name, target = self._parse_action_to_tuple(action)
        if not action_name:
            return

        # Skip failed actions
        if any(phrase in (result or "").lower() for phrase in [
            "too far", "occluded", "failed to", "unexpected failure",
            "cannot", "blocked",
        ]):
            return

        def check_fn(pred, args):
            prop = {"property": pred}
            if args:
                prop["entity"] = args[0]
            if len(args) > 1:
                prop["target"] = args[1]
            res = self.evaluate_task({"required_states": [prop]})
            return res and res.get("success", False)

        if action_name == "Navigate" and target:
            # Resolve furniture/object targets to their room
            # (Navigate targets are often furniture like table_29, not rooms)
            room = tracker.object_rooms.get(target, target)
            tracker.record_room_entry(agent_id, room, check_fn)

        elif action_name == "Communicate" and target:
            # Parse Communicate["message", recipient]
            import re
            comm_match = re.match(r'Communicate\[(["\'])(.*?)\1,\s*(.*?)\]', action)
            if comm_match:
                message = comm_match.group(2)
                recipient = comm_match.group(3).strip()
                if recipient == "all":
                    for i in range(getattr(self.task, 'num_agents', 2)):
                        other = f"agent_{i}"
                        if other != agent_id:
                            tracker.record_communication(
                                agent_id, other, message, check_fn
                            )
                else:
                    tracker.record_communication(
                        agent_id, recipient, message, check_fn
                    )

        elif action_name in ("Open", "Close", "Pick", "Place", "Clean", "Fill",
                             "PowerOn", "PowerOff", "UseItem"):
            # State-changing action — agents in same room observe it
            if target:
                change_room = tracker.object_rooms.get(target)
                # Record for all agents in the same room
                for pred in ("is_open", "is_closed", "is_on_top", "is_inside",
                             "is_clean", "is_dirty", "is_unlocked"):
                    if check_fn(pred, (target,)):
                        tracker.record_state_change(pred, (target,), change_room)

    def _check_pddl_goals(self) -> List[str]:
        """Check PDDL goal conjuncts and return newly completed ones."""
        if not hasattr(self, '_pddl_checker') or self._pddl_checker is None:
            belief_tracker = self._get_belief_tracker()
            self._pddl_checker = self.task.get_pddl_goal_checker()
            if self._pddl_checker is None:
                return []
            # Inject belief tracker into checker
            self._pddl_checker._belief_tracker = belief_tracker

        def check_predicate(pred_name, args):
            prop = {"property": pred_name}
            if args:
                prop["entity"] = args[0]
            if len(args) > 1:
                prop["target"] = args[1]
            result = self.evaluate_task({"required_states": [prop]})
            return result and result.get("success", False)

        result = self._pddl_checker.update(check_predicate)
        newly_completed = result.get("newly_completed", [])

        if newly_completed:
            total = len(self._pddl_checker.conjuncts)
            done = len(self._pddl_checker.completed)
            for goal_str in newly_completed:
                self._completed_subtasks.add(goal_str)
                print(f"\n{'─'*50}", flush=True)
                print(f"✓ GOAL COMPLETE: {goal_str}", flush=True)
                print(f"  Progress: {done}/{total}", flush=True)
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

    def _parse_action_to_tuple(self, action_str: str) -> Tuple[str, str]:
        """Parse action string like 'Navigate[kitchen_1]' to tuple ('Navigate', 'kitchen_1')."""
        match = re.match(r'(\w+)\[([^\]]*)\]', action_str)
        if match:
            return (match.group(1), match.group(2))
        return (action_str, "")

    # -------------------------------------------------------------------------
    # Output saving
    # -------------------------------------------------------------------------

    def _save_outputs(
        self,
        instruction: Dict[str, str],
        evaluation: Dict[str, Any],
        turn_count: int,
        comm_metrics: Optional[Dict[str, Any]] = None,
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

        if comm_metrics:
            log_data["communication_metrics"] = comm_metrics

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

    # Build team membership lookup: agent_id -> list of teammate agent_ids
    all_agents = sorted(task.agent_actions.keys())

    for agent_id in all_agents:
        parts = []

        # Header with agent identity
        parts.append(f"You are {agent_id.replace('_', ' ').title()}. Given the following task, take a sequence of actions to solve and complete the task at hand.")
        parts.append("")

        # Task description
        if task.task:
            parts.append(f"[Task]: {task.task}")
            parts.append("")

        # Known Information - what this agent knows
        secrets = list(task.agent_secrets.get(agent_id, []))

        # Prepend teammate info if not already present in secrets
        teammate_info = _build_teammate_info(agent_id, all_agents, task.teams)
        if teammate_info and not any("team" in s.lower() and "agent_" in s.lower() for s in secrets):
            secrets.insert(0, teammate_info)

        if secrets:
            parts.append("[Known Information]:")
            for s in secrets:
                parts.append(f"- {s}")

        # Active mechanic constraints — warn agents about mechanics that
        # affect how they should plan (e.g., irreversible actions)
        mechanic_warnings = _build_mechanic_warnings(task)
        if mechanic_warnings:
            parts.append("")
            parts.append("[Important Constraints]:")
            for w in mechanic_warnings:
                parts.append(f"- {w}")

        instructions[agent_id] = "\n".join(parts)

    return instructions


def _build_teammate_info(agent_id: str, all_agents: list, teams: dict = None) -> str:
    """Build a string describing which agents are on this agent's team."""
    if teams:
        # Competitive/team-based: find this agent's team and opponents
        my_team = None
        for team_id, members in teams.items():
            if agent_id in members:
                my_team = team_id
                break
        if my_team:
            teammates = [a for a in teams[my_team] if a != agent_id]
            opponents = []
            for team_id, members in teams.items():
                if team_id != my_team:
                    opponents.extend(members)
            parts = [f"You are on {my_team}"]
            if teammates:
                parts[0] += f" with {', '.join(teammates)}"
            if opponents:
                parts.append(f"the opposing {'team is' if len(opponents) == 1 else 'agents are'} {', '.join(opponents)}")
            return ". ".join(parts) + "."
    else:
        # Cooperative/mixed: all other agents are teammates
        others = [a for a in all_agents if a != agent_id]
        if others:
            return f"Your teammates are: {', '.join(others)}."
    return ""


def _build_mechanic_warnings(task: "GeneratedTask") -> List[str]:
    """Build warning strings for active mechanics that agents should know about upfront."""
    warnings = []
    active = getattr(task, "active_mechanics", []) or []

    if "irreversible_action" in active:
        warnings.append(
            "IRREVERSIBLE ACTIONS: Actions on objects are permanent. "
            "Once you perform an action on an object (Open, Close, Pick, Place), "
            "that object becomes permanently locked and no further actions can be "
            "taken on it by anyone for the rest of the task. Think carefully and "
            "coordinate with your partner before acting."
        )

    if "limited_bandwidth" in active:
        warnings.append(
            "LIMITED COMMUNICATION: You have a limited number of messages you can send. "
            "Choose your words carefully and prioritize the most important information."
        )

    return warnings
