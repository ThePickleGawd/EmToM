"""
Unified benchmark runner for EMTOM evaluation.

Supports both LLM and human-controlled agents. Each agent can be configured
as human or LLM via the human_agents parameter.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from omegaconf import DictConfig

from emtom.actions.baseline_tools import ReadAgentTrajectoryTool
from emtom.vision import VisualObservationStore

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
        self._visual_store: Optional[VisualObservationStore] = None
        self._trajectory_store: List[Dict[str, Any]] = []

    def get_run_mode(self) -> str:
        raw_mode = str(getattr(self.config, "benchmark_run_mode", "standard")).strip().lower()
        return raw_mode if raw_mode in {"standard", "baseline", "full_info"} else "standard"

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

        agent_actions = self._resolve_agent_actions(task)
        message_targets = self._resolve_message_targets(task)

        super().setup(env_interface, task_data, output_dir, agent_actions=agent_actions, save_video=save_video, message_targets=message_targets)

        self._trajectory_store = []
        if self.get_run_mode() in {"baseline", "full_info"}:
            self._inject_baseline_tools()

        if self._is_vision_mode():
            vision_cfg = getattr(self.config, "benchmark_vision", None)
            image_format = getattr(vision_cfg, "image_format", "jpeg") if vision_cfg else "jpeg"
            self._visual_store = VisualObservationStore(
                os.path.join(self.output_dir, "visual_observations"),
                image_format=image_format,
            )
        else:
            self._visual_store = None

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

    @staticmethod
    def _normalize_agent_id(raw_agent: Any) -> Optional[str]:
        """Normalize mixed agent ID formats to canonical 'agent_<uid>'."""
        if isinstance(raw_agent, int):
            return f"agent_{raw_agent}"
        if not isinstance(raw_agent, str):
            return None

        agent = raw_agent.strip()
        if not agent:
            return None
        if agent.startswith("agent_"):
            suffix = agent.split("_", 1)[1]
            if suffix.isdigit():
                return f"agent_{int(suffix)}"
            return None
        if agent.isdigit():
            return f"agent_{int(agent)}"
        return None

    @classmethod
    def _normalize_message_targets(
        cls,
        raw_targets: Any,
        num_agents: int,
    ) -> Dict[str, List[str]]:
        """Normalize sender->recipients message target map."""
        if not isinstance(raw_targets, dict):
            return {}

        valid_agents = {f"agent_{i}" for i in range(max(0, int(num_agents or 0)))}
        normalized: Dict[str, List[str]] = {}

        for raw_sender, raw_recipients in raw_targets.items():
            sender = cls._normalize_agent_id(raw_sender)
            if not sender or sender not in valid_agents:
                continue
            if not isinstance(raw_recipients, list):
                continue

            recipients: List[str] = []
            for raw_recipient in raw_recipients:
                recipient = cls._normalize_agent_id(raw_recipient)
                if not recipient or recipient not in valid_agents:
                    continue
                if recipient == sender or recipient in recipients:
                    continue
                recipients.append(recipient)

            normalized[sender] = recipients

        return normalized

    @classmethod
    def _resolve_message_targets(
        cls,
        task: Optional["GeneratedTask"],
    ) -> Optional[Dict[str, List[str]]]:
        """
        Resolve message targets for runtime communication enforcement.

        Priority:
        1) Explicit task.message_targets
        2) restricted_communication mechanic_bindings.allowed_targets
        """
        if task is None:
            return None

        explicit = cls._normalize_message_targets(task.message_targets, task.num_agents)
        if explicit:
            return explicit

        for binding in task.mechanic_bindings:
            if binding.mechanic_type != "restricted_communication":
                continue
            derived = cls._normalize_message_targets(
                binding.allowed_targets, task.num_agents
            )
            if derived:
                return derived

        return None

    def _resolve_agent_actions(
        self,
        task: Optional["GeneratedTask"],
    ) -> Optional[Dict[str, List[str]]]:
        if task is None:
            return None

        agent_actions = {
            agent_id: list(actions or [])
            for agent_id, actions in (task.agent_actions or {}).items()
        }

        if self.get_run_mode() in {"baseline", "full_info"}:
            for i in range(task.num_agents):
                agent_id = f"agent_{i}"
                actions = agent_actions.setdefault(agent_id, [])
                if "ReadAgentTrajectoryTool" not in actions:
                    actions.append("ReadAgentTrajectoryTool")

        return agent_actions

    def _inject_baseline_tools(self) -> None:
        include_observations = self.get_run_mode() == "full_info"
        for uid, agent in self.agents.items():
            tool = ReadAgentTrajectoryTool(
                agent_uid=uid,
                include_observations=include_observations,
            )
            tool.env_interface = self.env_interface
            tool.set_trajectory_store(self._trajectory_store)
            agent.tools["ReadAgentTrajectoryTool"] = tool

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

    @staticmethod
    def _make_action_history_entry(
        *,
        sim_step: int,
        turn: int,
        agent_id: str,
        action: str,
        result: str,
        mode: str,
        skill_steps: Optional[int] = None,
        selected_frames: Optional[List[str]] = None,
        selected_frame_paths: Optional[List[str]] = None,
        selected_frame_handles: Optional[List[Dict[str, Any]]] = None,
        selector: Optional[Dict[str, Any]] = None,
        thought: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a stable action-history record for downstream analysis."""
        entry: Dict[str, Any] = {
            "sim_step": sim_step,
            "turn": turn,
            "agent": agent_id,
            "agent_id": agent_id,
            "action": action,
            "action_taken": action,
            "result": result,
            "observation": result,
            "mode": mode,
        }
        if thought is not None:
            entry["thought"] = thought
        if skill_steps is not None:
            entry["skill_steps"] = skill_steps
        if selected_frames is not None:
            entry["selected_frames"] = list(selected_frames)
        if selected_frame_paths is not None:
            entry["selected_frame_paths"] = list(selected_frame_paths)
        if selected_frame_handles is not None:
            entry["selected_frame_handles"] = [dict(handle) for handle in selected_frame_handles]
        if selector is not None:
            entry["selector"] = selector
        return entry

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
        self._capture_visual_frames(observations, turn=0, skill_step=0, kind="initial")

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
            self._set_planner_visual_context(turn=max(turn_count - 1, 0))

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

                        self._action_history.append(
                            self._make_action_history_entry(
                                sim_step=self._step_count,
                                turn=turn_count,
                                agent_id=agent_id,
                                action=action,
                                result=observation,
                                mode="human",
                            )
                        )
                        self._append_trajectory_entry(
                            turn=turn_count,
                            agent_id=agent_id,
                            action=action,
                            thought=None,
                            observation=observation,
                        )

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

            # Benchmark turns are action-completion rounds; force a fresh
            # high-level plan each turn to avoid stale-action carryover loops.
            for uid, planner in self.planners.items():
                if f"agent_{uid}" not in self.human_agents:
                    planner.replan_required = True

            # Buffer messages sent during this turn - they should only be visible next turn
            message_buffer: List[tuple] = []
            blocked_message_by_sender: Dict[int, str] = {}
            original_post_message = self.env_interface.post_agent_message
            initial_sent_counts: Dict[str, int] = {}
            if self.game_manager:
                gs = self.game_manager.get_state()
                initial_sent_counts = dict(getattr(gs, "messages_sent", {}) or {})
            planned_sent_counts: Dict[str, int] = dict(initial_sent_counts)

            def _allowed_recipients_for_sender(sender_uid: int) -> Optional[List[int]]:
                sender_id = f"agent_{sender_uid}"
                mapping = self._message_targets if isinstance(self._message_targets, dict) else {}
                raw_allowed = mapping.get(sender_id)
                if raw_allowed is None:
                    return None
                allowed_uids: List[int] = []
                for raw in raw_allowed:
                    normalized = self._normalize_agent_id(raw)
                    if not normalized:
                        continue
                    try:
                        uid = int(normalized.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                    if uid not in allowed_uids:
                        allowed_uids.append(uid)
                return allowed_uids

            def _enforce_message_policy(
                sender_uid: int,
                target_uids: Optional[List[int]],
            ) -> Tuple[bool, Optional[List[int]], Optional[str]]:
                """Return (allowed, effective_targets, reason_if_blocked)."""
                sender_id = f"agent_{sender_uid}"
                effective_targets = (
                    list(target_uids) if isinstance(target_uids, list) else None
                )

                # Enforce recipient topology first.
                allowed_recipients = _allowed_recipients_for_sender(sender_uid)
                if allowed_recipients is not None:
                    if effective_targets is None:
                        effective_targets = [
                            uid for uid in allowed_recipients if uid != sender_uid
                        ]
                    else:
                        allowed_set = set(allowed_recipients)
                        effective_targets = [
                            uid for uid in effective_targets if uid in allowed_set
                        ]
                    if not effective_targets:
                        return (
                            False,
                            [],
                            "Message blocked: recipient is not allowed by restricted communication.",
                        )

                # Enforce message budget using turn-local counters.
                if self.game_manager:
                    gs = self.game_manager.get_state()
                    limit = getattr(gs, "message_limits", {}).get(sender_id)
                    if limit is not None:
                        used = planned_sent_counts.get(
                            sender_id, initial_sent_counts.get(sender_id, 0)
                        )
                        if used >= limit:
                            return (
                                False,
                                effective_targets,
                                f"Message blocked: {sender_id} has no messages remaining.",
                            )
                        planned_sent_counts[sender_id] = used + 1

                return True, effective_targets, None

            def buffered_post_message(sender_uid: int, message: str, target_uids=None) -> None:
                allowed, effective_targets, blocked_reason = _enforce_message_policy(
                    sender_uid,
                    target_uids,
                )
                if not allowed:
                    blocked_message_by_sender[sender_uid] = blocked_reason or "Message blocked."
                    return
                message_buffer.append((sender_uid, message, effective_targets))

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
                                'skip_mechanic_apply': False,
                                'selected_frames': planner_info.get("selected_frames", []),
                                'selector': planner_info.get("selector"),
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
                                    'skip_mechanic_apply': False,
                                    'selected_frames': planner_info.get("selected_frames", []),
                                    'selector': planner_info.get("selector"),
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
                        'skip_mechanic_apply': False,
                        'selected_frames': planner_info.get("selected_frames", []),
                        'selector': planner_info.get("selector"),
                    }

                    # Check if action already completed in first call
                    responses_dict = planner_info.get("responses", {})
                    if responses_dict.get(uid):
                        llm_agent_state[uid]['response'] = responses_dict[uid]
                        llm_agent_state[uid]['action_done'] = True
                    elif not low_level_actions or uid not in low_level_actions:
                        # Perception tool (Communicate, Find*, etc.) completed
                        # instantly — no simulation steps needed. The planner may
                        # return no response text, so reconstruct a fallback.
                        response = responses_dict.get(uid, "")
                        llm_agent_state[uid]['response'] = response or "Executed."
                        llm_agent_state[uid]['action_done'] = True

                    if orig_action_name == "Communicate":
                        blocked_reason = blocked_message_by_sender.get(uid)
                        if blocked_reason:
                            llm_agent_state[uid]['response'] = blocked_reason
                            llm_agent_state[uid]['action_done'] = True
                            llm_agent_state[uid]['skip_mechanic_apply'] = True

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
            blocked_mechanic_senders = {
                uid
                for uid, state in llm_agent_state.items()
                if state.get("orig_action") == "Communicate"
                and state.get("mech_result") is not None
                and getattr(state.get("mech_result"), "blocked", False)
            }
            for sender_uid, message, target_uids in message_buffer:
                if sender_uid in blocked_mechanic_senders:
                    # Communicate failed at mechanic layer (e.g., unreliable_communication).
                    # Drop buffered message so it never reaches recipients.
                    continue
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

                        if done_flag:
                            self._episode_done = True
                            for state in llm_agent_state.values():
                                if state['action_done']:
                                    continue
                                if state.get('response') is None:
                                    state['response'] = "Episode ended during concurrent skill execution."
                                state['action_done'] = True

                    except Exception as e:
                        print(f"[Concurrent step error] {e}", flush=True)
                        break

                total_skill_steps += 1
                if self._episode_done:
                    break
                if total_skill_steps % 30 == 0:
                    self._capture_visual_frames(
                        observations,
                        turn=turn_count,
                        skill_step=total_skill_steps,
                        kind="in_action",
                    )

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
                        # Enforce one LLM call per turn: after the turn's initial
                        # planning pass, phase-3 updates must never trigger a replan.
                        state['planner'].replan_required = False
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
                visual_summary = self._build_textual_visual_summary(
                    uid,
                    surroundings.get(uid, []),
                    agents_passed.get(uid),
                )

                # Append surroundings observations collected during motor skill
                response = self._append_textual_visual_summary(
                    response,
                    uid,
                    surroundings.get(uid, []),
                    agents_passed.get(uid),
                )
                self._inject_summary_into_planner_context(
                    state.get("planner"),
                    visual_summary,
                )

                # Apply mechanic state changes after successful execution
                mech_result = state.get('mech_result')
                if (
                    self.game_manager
                    and mech_result
                    and not getattr(mech_result, "blocked", False)
                    and not state.get("skip_mechanic_apply", False)
                ):
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

                self._action_history.append(
                    self._make_action_history_entry(
                        sim_step=self._step_count,
                        turn=turn_count,
                        agent_id=agent_id,
                        action=high_level_action,
                        result=response,
                        mode="llm",
                        skill_steps=skill_steps,
                        selected_frames=[
                            handle.get("frame_id")
                            for handle in state.get("selected_frames", [])
                        ],
                        selected_frame_paths=[
                            handle.get("path")
                            for handle in state.get("selected_frames", [])
                            if handle.get("path")
                        ],
                        selected_frame_handles=state.get("selected_frames", []),
                        selector=state.get("selector"),
                        thought=state.get("planner_info", {}).get("thought", {}).get(uid),
                    )
                )
                self._append_trajectory_entry(
                    turn=turn_count,
                    agent_id=agent_id,
                    action=high_level_action,
                    thought=state.get("planner_info", {}).get("thought", {}).get(uid),
                    observation=response,
                )

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
                self._capture_visual_frames(
                    observations,
                    turn=turn_count,
                    skill_step=total_skill_steps,
                    kind="turn_end",
                )

        print(f"\n[Benchmark] Finished: steps={self._step_count}, turns={turn_count}, done={done}", flush=True)

        # Final evaluation
        evaluation = self._check_task_completion() or {}
        evaluation["functional_success"] = evaluation.get("success", False)
        probe_metrics = self._run_literal_tom_probes(instruction)
        if probe_metrics:
            evaluation["literal_tom_probe_score"] = probe_metrics.get("score")
            evaluation["literal_tom_probe_results"] = probe_metrics.get("results", [])
            evaluation["literal_tom_probe_summary"] = probe_metrics.get("summary", {})

        # Communication metrics
        comm_metrics = None
        if self.get_run_mode() in {"baseline", "full_info"}:
            comm_dict = {
                "status": "not_applicable",
                "reason": "This run mode shares private reasoning explicitly.",
            }
        elif self.task and self._action_history:
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
        self._save_outputs(
            instruction,
            evaluation,
            turn_count,
            done=done,
            comm_metrics=comm_dict,
        )

        result = {
            "run_mode": self.get_run_mode(),
            "steps": self._step_count,
            "turns": turn_count,
            "done": done,
            "episode_over": self._episode_done,
            "action_history": self._action_history,
            "evaluation": evaluation,
            "success": evaluation.get("success", False),
        }
        if self._is_vision_mode():
            result["vision_mode"] = True
            result["selector_metrics"] = self._collect_selector_metrics()
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

        category = getattr(self.task, "category", "cooperative")

        if checker.is_or_goal and category == "competitive":
            # Group goals by branch/team for competitive display
            teams = checker.get_all_teams()
            print(f"\n{'='*50}")
            print(f"PDDL GOALS (competitive)")
            print(f"{'='*50}")

            for branch_idx in range(checker.num_branches):
                indices = checker.get_branch_conjunct_indices(branch_idx)
                branch_done = sum(1 for i in indices if checker.is_conjunct_completed(i))
                branch_total = len(indices)
                # Find team for this branch
                team_label = None
                for team_id in teams:
                    if checker.get_branch_for_team(team_id) == branch_idx:
                        team_label = team_id
                        break
                label = team_label or f"branch_{branch_idx}"
                print(f"  {label} ({branch_done}/{branch_total}):")
                for i in indices:
                    done = checker.is_conjunct_completed(i)
                    status = "✓" if done else "○"
                    print(f"    {status} {checker.conjuncts[i].to_pddl()}")
        else:
            total = len(checker.conjuncts)
            completed = len(checker.completed)
            print(f"\n{'='*50}")
            print(f"PDDL GOALS ({completed}/{total})")
            print(f"{'='*50}")

            for i, conjunct in enumerate(checker.conjuncts):
                done = checker.is_conjunct_completed(i)
                status = "✓" if done else "○"
                owner = checker.get_owner(i)
                owner_str = f" [{owner}]" if owner else ""
                print(f"  {status} {conjunct.to_pddl()}{owner_str}")

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
            return {"success": False, "error": "No PDDL goal",
                    "total_subtasks": 0, "percent_complete": 0.0}

        # Guard against empty conjuncts (vacuous success)
        if not checker.conjuncts:
            return {"success": False, "error": "PDDL goal checker has no conjuncts",
                    "total_subtasks": 0, "percent_complete": 0.0}

        category = getattr(self.task, "category", "cooperative")

        if category == "competitive":
            # Use branch-aware methods for competitive Or-goals
            teams = checker.get_all_teams()
            team_status = {}
            team_progress = {}
            winner = None

            if checker.is_or_goal:
                # Map each team to its Or-branch and use branch progress
                for team_id in teams:
                    branch_idx = checker.get_branch_for_team(team_id)
                    if branch_idx is not None:
                        progress = checker.get_branch_progress(branch_idx)
                        complete = checker.is_branch_complete(branch_idx)
                    else:
                        progress = 0.0
                        complete = False
                    team_progress[team_id] = progress
                    team_status[team_id] = complete
                    if complete and winner is None:
                        winner = team_id

                # Best branch progress as overall percent
                best_progress = max(team_progress.values()) if team_progress else 0.0
            else:
                # Fallback for competitive tasks without Or-structure
                for team_id in teams:
                    team_conj = checker.get_team_conjuncts(team_id)
                    if not team_conj:
                        team_status[team_id] = False
                        team_progress[team_id] = 0.0
                        continue
                    done = sum(1 for c in team_conj if checker.is_conjunct_completed(checker.conjuncts.index(c)))
                    progress = done / len(team_conj)
                    team_progress[team_id] = progress
                    team_status[team_id] = progress == 1.0
                    if team_status[team_id] and winner is None:
                        winner = team_id
                best_progress = max(team_progress.values()) if team_progress else 0.0

            return {
                "success": winner is not None,
                "functional_success": winner is not None,
                "winner": winner,
                "team_status": team_status,
                "team_progress": team_progress,
                "completed_subtasks": list(checker.completed),
                "total_subtasks": len(checker.conjuncts),
                "percent_complete": best_progress,
            }

        elif category == "mixed":
            required = checker.get_required_conjuncts()
            # If all goals have owners (no unowned "required" goals),
            # treat all conjuncts as required to avoid vacuous success
            if not required and checker.conjuncts:
                required = list(checker.conjuncts)
            required_done = sum(
                1 for c in required
                if checker.is_conjunct_completed(checker.conjuncts.index(c))
            )
            main_goal_success = (required_done == len(required)) if required else False

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
            completed_required_ids = [
                checker.conjuncts.index(c)
                for c in required
                if checker.is_conjunct_completed(checker.conjuncts.index(c))
            ]
            required_total = len(required)
            return {
                "success": main_goal_success,
                "functional_success": main_goal_success,
                "main_goal_success": main_goal_success,
                "agent_subgoal_status": agent_subgoal_status,
                # Mixed-task progress should reflect the success-relevant
                # functional goal scope, not optional/private side goals.
                "completed_subtasks": completed_required_ids,
                "total_subtasks": required_total,
                "percent_complete": (
                    required_done / required_total if required_total else 0.0
                ),
                "completed_all_subtasks": list(checker.completed),
                "total_all_subtasks": total,
                "all_goal_percent_complete": completed / total if total else 0.0,
            }

        else:
            # Cooperative: all conjuncts must be complete
            total = len(checker.conjuncts)
            completed = len(checker.completed)
            all_complete = completed == total
            return {
                "success": all_complete,
                "functional_success": all_complete,
                "completed_subtasks": list(checker.completed),
                "total_subtasks": total,
                "percent_complete": completed / total if total else 0.0,
            }

    def _check_subtasks(self) -> List[str]:
        """Check PDDL goal conjuncts and return newly completed IDs."""
        if not self.task:
            return []
        return self._check_pddl_goals()

    def _check_pddl_goals(self) -> List[str]:
        """Check PDDL goal conjuncts and return newly completed ones."""
        if not hasattr(self, '_pddl_checker') or self._pddl_checker is None:
            self._pddl_checker = self.task.get_pddl_goal_checker()
            if self._pddl_checker is None:
                return []

        def check_predicate(pred_name, args):
            prop = {"property": pred_name}
            if args:
                prop["entity"] = args[0]
            if len(args) > 1:
                prop["target"] = args[1]
            result = self.evaluate_task({"required_states": [prop]})
            return result and result.get("success", False)

        checker = self._pddl_checker
        result = checker.update(check_predicate)
        newly_completed = result.get("newly_completed", [])

        if newly_completed:
            category = getattr(self.task, "category", "cooperative")
            for goal_str in newly_completed:
                self._completed_subtasks.add(goal_str)
                print(f"\n{'─'*50}", flush=True)

                if checker.is_or_goal and category == "competitive":
                    # Show team attribution for competitive goals
                    # Find which team owns this goal
                    team_label = ""
                    for i, c in enumerate(checker.conjuncts):
                        if c.to_pddl() == goal_str:
                            owner = checker.get_owner(i)
                            if owner:
                                team_label = f" [{owner}]"
                            break
                    print(f"✓ GOAL COMPLETE: {goal_str}{team_label}", flush=True)
                    # Show per-team progress
                    teams = checker.get_all_teams()
                    parts = []
                    for team_id in teams:
                        branch_idx = checker.get_branch_for_team(team_id)
                        if branch_idx is not None:
                            indices = checker.get_branch_conjunct_indices(branch_idx)
                            done = sum(1 for idx in indices if checker.is_conjunct_completed(idx))
                            parts.append(f"{team_id}: {done}/{len(indices)}")
                    if parts:
                        print(f"  {' | '.join(parts)}", flush=True)
                else:
                    total = len(checker.conjuncts)
                    done = len(checker.completed)
                    print(f"✓ GOAL COMPLETE: {goal_str}", flush=True)
                    print(f"  Progress: {done}/{total}", flush=True)

                print(f"{'─'*50}", flush=True)

        return newly_completed

    def _run_literal_tom_probes(
        self,
        instruction: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Run deterministic end-of-episode literal-ToM probes."""
        if not self.task:
            return None

        from emtom.pddl.runtime_projection import (
            evaluate_literal_tom_probe,
            load_literal_tom_probes,
        )

        probes = load_literal_tom_probes(self.task.get_literal_tom_probes())
        if not probes:
            return {
                "score": None,
                "results": [],
                "summary": {"probe_count": 0, "supported_probe_count": 0, "passed_count": 0},
            }

        def check_fn(pred_name: str, args: Tuple[str, ...]) -> bool:
            prop = {"property": pred_name}
            if args:
                prop["entity"] = args[0]
            if len(args) > 1:
                prop["target"] = args[1]
            result = self.evaluate_task({"required_states": [prop]})
            return bool(result and result.get("success", False))

        results: List[Dict[str, Any]] = []
        supported_count = 0
        passed_count = 0

        probe_groups: Dict[str, List[Any]] = defaultdict(list)
        for probe in probes:
            probe_groups[probe.agent_id].append(probe)

        for agent_id, agent_probes in probe_groups.items():
            batch_raw_response: Any = None
            batch_answers: Dict[str, Dict[str, Any]] = {}
            batch_error: Optional[str] = None

            supported_agent_probes = [probe for probe in agent_probes if probe.supported]
            if supported_agent_probes:
                planner = self.planners.get(int(agent_id.split("_", 1)[1])) if agent_id else None
                if planner is None or not hasattr(planner, "llm"):
                    batch_error = "No LLM planner available for probed agent."
                else:
                    probe_prompt = self._build_literal_tom_probe_prompt(
                        agent_id,
                        supported_agent_probes,
                        instruction,
                    )
                    try:
                        batch_raw_response = planner.llm.generate(probe_prompt)
                        response_data = self._extract_probe_json(batch_raw_response)
                        batch_answers = self._extract_probe_answers(response_data)
                    except Exception as exc:
                        batch_error = f"Probe generation failed: {exc}"

                    # Append probe prompt + response to the agent's prompt file
                    self._append_probe_to_prompt_file(
                        agent_id, probe_prompt, batch_raw_response, batch_error,
                    )

            for probe in agent_probes:
                result_entry: Dict[str, Any] = {
                    "probe_id": probe.probe_id,
                    "agent_id": probe.agent_id,
                    "source_pddl": probe.source_pddl,
                    "question": probe.question,
                    "supported": probe.supported,
                }
                if not probe.supported:
                    result_entry["status"] = "unsupported"
                    result_entry["reason"] = probe.unsupported_reason
                    results.append(result_entry)
                    continue

                supported_count += 1
                if batch_error is not None:
                    result_entry["status"] = "error"
                    result_entry["reason"] = batch_error
                    result_entry["raw_response"] = batch_raw_response
                    results.append(result_entry)
                    continue

                response_data = batch_answers.get(probe.probe_id, {})
                passed, details = evaluate_literal_tom_probe(probe, response_data, check_fn)
                if passed:
                    passed_count += 1
                result_entry.update({
                    "status": "passed" if passed else "failed",
                    "raw_response": batch_raw_response,
                    "details": details,
                })
                results.append(result_entry)

        score = (passed_count / supported_count) if supported_count else None
        return {
            "score": score,
            "results": results,
            "summary": {
                "probe_count": len(probes),
                "supported_probe_count": supported_count,
                "passed_count": passed_count,
            },
        }

    def _build_literal_tom_probe_prompt(
        self,
        agent_id: str,
        probes: List[Any],
        instruction: Dict[str, str],
    ) -> str:
        """Build a deterministic post-episode probe prompt."""
        uid = int(agent_id.split("_", 1)[1])
        planner = self.planners.get(uid)
        if planner is not None and getattr(planner, "curr_prompt", ""):
            context = str(planner.curr_prompt)
        else:
            context = self._build_probe_context_fallback(agent_id, instruction)
        probe_lines = ["Report one structured answer per probe."]
        probe_lines.append(
            'Respond with JSON only: {"answers":[{"probe_id":"k_probe_X","predicate":"<predicate_name>|unknown","holds":true|false|null,"args":["entity_or_target", ...]}, ...]}'
        )
        probe_lines.append("Use predicate \"unknown\" with holds null and empty args if the agent does not know.")
        for probe in probes:
            probe_lines.append(f"- {probe.question}")
        return (
            f"{context}\n\n"
            "The episode is over. Do not propose any more actions.\n"
            "Using only the episode context above, provide the requested structured report.\n"
            f"{chr(10).join(probe_lines)}\n"
        )

    def _append_probe_to_prompt_file(
        self,
        agent_id: str,
        probe_prompt: str,
        raw_response: Any,
        error: Optional[str],
    ) -> None:
        """Append the literal ToM probe prompt and response to the agent's prompt file."""
        uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id
        prompt_dir = os.path.join(self.output_dir, "prompts", uid)
        if not os.path.isdir(prompt_dir):
            return
        prompt_files = sorted(
            f for f in os.listdir(prompt_dir) if f.startswith("prompt-") and f.endswith(".txt")
        )
        if not prompt_files:
            return
        prompt_file = os.path.join(prompt_dir, prompt_files[-1])
        with open(prompt_file, "a") as f:
            f.write(probe_prompt)
            f.write(str(raw_response or error or ""))

    def _build_probe_context_fallback(
        self,
        agent_id: str,
        instruction: Dict[str, str],
    ) -> str:
        """Fallback probe context when the planner prompt is unavailable."""
        agent_instruction = instruction.get(agent_id, "")
        history_lines = []
        for entry in self._action_history:
            if entry.get("agent_id") != agent_id and entry.get("agent") != agent_id:
                continue
            action = entry.get("action", "")
            observation = entry.get("observation") or entry.get("result") or ""
            history_lines.append(f"- Action: {action}")
            if observation:
                history_lines.append(f"  Observation: {observation}")
        history_text = "\n".join(history_lines) if history_lines else "- No recorded actions."
        return (
            f"Task instruction for {agent_id}:\n{agent_instruction}\n\n"
            f"Episode history for {agent_id}:\n{history_text}"
        )

    @staticmethod
    def _extract_probe_json(raw_response: Any) -> Dict[str, Any]:
        """Extract the first JSON object from a probe response."""
        if isinstance(raw_response, dict):
            return raw_response
        text = str(raw_response or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _extract_probe_answers(response_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize a batch probe response into probe_id -> answer mapping."""
        if not isinstance(response_data, dict):
            return {}
        raw_answers = response_data.get("answers")
        if not isinstance(raw_answers, list):
            probe_id = str(response_data.get("probe_id", "")).strip()
            return {probe_id: response_data} if probe_id else {}

        answers: Dict[str, Dict[str, Any]] = {}
        for raw in raw_answers:
            if not isinstance(raw, dict):
                continue
            probe_id = str(raw.get("probe_id", "")).strip()
            if not probe_id:
                continue
            answers[probe_id] = raw
        return answers

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
        done: bool,
        comm_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save video, planner log, and prompts."""
        task_id = self.task.task_id if self.task else "unknown"
        sim_steps = self._step_count
        success = evaluation.get("success", False)

        # Save video
        mode = "human" if self.human_agents else "benchmark"
        self.save_video(f"{mode}_{task_id}_{sim_steps}")

        # Build log data
        log_data = {
            "task_id": task_id,
            "task_title": self.task.title if self.task else "Unknown",
            "run_mode": self.get_run_mode(),
            "instruction": instruction,
            "mechanics_active": self.game_manager.get_state().active_mechanics if self.game_manager else [],
            "steps": sim_steps,
            "sim_steps": sim_steps,  # Legacy alias kept for older analysis scripts.
            "turns": turn_count,
            "done": done,
            "episode_over": self._episode_done,
            "success": success,
            "human_agents": list(self.human_agents),
            "llm_agents": [f"agent_{uid}" for uid in self.planners.keys()],
            "action_history": self._action_history,
        }

        if evaluation:
            log_data["evaluation"] = evaluation
            log_data["percent_complete"] = evaluation.get("percent_complete", 0.0)

        if self.task and self.task.mechanic_bindings:
            log_data["mechanic_bindings"] = [b.to_dict() for b in self.task.mechanic_bindings]

        if comm_metrics:
            log_data["communication_metrics"] = comm_metrics

        if self._is_vision_mode():
            log_data["vision_mode"] = True
            log_data["selector_metrics"] = self._collect_selector_metrics()
            if self._visual_store is not None:
                log_data["visual_frame_index"] = self._visual_store.export_index()

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

    def _capture_visual_frames(
        self,
        observations: Dict[str, Any],
        turn: int,
        skill_step: int,
        kind: str,
    ) -> None:
        if not self._is_vision_mode() or self._visual_store is None:
            return

        agent_ids = [f"agent_{uid}" for uid in sorted(self.agents.keys())]
        self._visual_store.capture(
            observations=observations,
            agent_ids=agent_ids,
            turn=turn,
            skill_step=skill_step,
            sim_step=self.get_sim_steps(),
            kind=kind,
        )

    def _set_planner_visual_context(self, turn: int) -> None:
        if not self._is_vision_mode() or self._visual_store is None:
            return

        for uid, planner in self.planners.items():
            if not hasattr(planner, "set_visual_context"):
                continue
            agent_id = f"agent_{uid}"
            planner.set_visual_context(
                turn=turn,
                available_frames=self._visual_store.get_turn_handles(turn, agent_id),
            )

    def _collect_selector_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for uid, planner in self.planners.items():
            if hasattr(planner, "get_selector_metrics"):
                metrics[f"agent_{uid}"] = planner.get_selector_metrics()
        return metrics

    def _append_trajectory_entry(
        self,
        *,
        turn: int,
        agent_id: str,
        action: str,
        thought: Optional[str],
        observation: Optional[str] = None,
    ) -> None:
        entry = {
            "turn": turn,
            "agent_id": agent_id,
            "action": action,
            "thought": thought,
        }
        if observation is not None:
            entry["observation"] = observation
        self._trajectory_store.append(entry)


def _build_known_information(task: "GeneratedTask", agent_id: str, run_mode: str) -> List[str]:
    if run_mode in {"baseline", "full_info"}:
        lines: List[str] = []
        for other_agent_id in sorted(task.agent_actions.keys()):
            for secret in task.agent_secrets.get(other_agent_id, []):
                lines.append(f"{other_agent_id} knows: {secret}")
        if task.team_secrets:
            for team_id, team_secs in sorted(task.team_secrets.items()):
                for ts in team_secs:
                    lines.append(f"{team_id} knows: {ts}")
        return lines

    secrets = list(task.agent_secrets.get(agent_id, []))

    # Surface team_secrets for competitive tasks — each agent sees their
    # team's shared secrets (e.g., which appliance to target).
    if task.team_secrets and task.teams:
        for team_id, members in task.teams.items():
            if agent_id in members:
                for ts in task.team_secrets.get(team_id, []):
                    if ts not in secrets:
                        secrets.append(ts)
                break

    teammate_info = _build_teammate_info(agent_id, sorted(task.agent_actions.keys()), task.teams)
    if teammate_info and not any("team" in s.lower() and "agent_" in s.lower() for s in secrets):
        secrets.insert(0, teammate_info)

    return secrets


def task_to_instruction(task: "GeneratedTask", run_mode: str = "standard") -> Dict[str, str]:
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

        if run_mode == "baseline":
            parts.append(
                "[Baseline Mode]: All agents' secrets are shared, and you may read "
                "other agents' completed Thought + Action trajectories with "
                "ReadAgentTrajectoryTool."
            )
            parts.append("")
        elif run_mode == "full_info":
            parts.append(
                "[Full Information Mode]: All agents' secrets are shared, and you "
                "may read other agents' completed Observation + Thought + Action "
                "trajectories with ReadAgentTrajectoryTool."
            )
            parts.append("")

        known_information = _build_known_information(task, agent_id, run_mode)
        if known_information:
            parts.append("[Known Information]:")
            for s in known_information:
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
