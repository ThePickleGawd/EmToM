"""
Game-aware evaluation runner wrapper. Reuses the standard decentralized runner loop
but injects game context into the instruction and monitors game termination.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

from habitat.sims.habitat_simulator.sim_utilities import get_obj_from_id

from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.utils import rollout_print
from game.game_tool import GameTool


class GameDecentralizedEvaluationRunner(DecentralizedEvaluationRunner):
    def __init__(
        self,
        evaluation_runner_config_arg,
        env_arg,
        game_orchestrator,
        base_instruction: Optional[str] = None,
    ):
        self.game_orchestrator = game_orchestrator
        self.base_instruction = base_instruction
        super().__init__(evaluation_runner_config_arg, env_arg)
        # Wrap post_agent_message to preprocess messages through the game spec
        self._wrap_post_agent_message()

    def _wrap_post_agent_message(self) -> None:
        """
        Wrap the env_interface.post_agent_message to preprocess messages
        through the game spec before sending. This allows games to redact
        sensitive information (e.g., secret codes) from agent communications.
        """
        if not self.game_orchestrator:
            return
        original_post = self.env_interface.post_agent_message
        orchestrator = self.game_orchestrator

        def wrapped_post(sender_uid: int, message: str, target_uids=None) -> None:
            if orchestrator.state is not None:
                message = orchestrator.game_spec.preprocess_message(
                    str(sender_uid), message, orchestrator.state
                )
            original_post(sender_uid, message, target_uids=target_uids)

        self.env_interface.post_agent_message = wrapped_post

    def _inject_game_tools(self) -> None:
        """
        Attach game tools to each agent's tool registry so planners can call them.
        """
        if not self.game_orchestrator:
            return
        state = self.game_orchestrator.state
        if state is None:
            return
        for agent in self.agents.values():
            descs = self.game_orchestrator.game_spec.get_tools_for_agent(
                str(agent.uid), state, self.game_orchestrator.env
            )
            added = []
            for desc in descs:
                if desc.name in agent.tools:
                    continue
                tool = GameTool(desc, self.game_orchestrator, agent.uid, self.env_interface)
                agent.tools[tool.name] = tool
                added.append(desc.name)

    def _compose_instruction(self, fallback_instruction: str, agent_uid: int) -> str:
        """
        Merge game public context and role info with the base task instruction for a specific agent.
        Only includes that agent's private info.
        """
        if not self.game_orchestrator or not self.game_orchestrator.state:
            return fallback_instruction

        state = self.game_orchestrator.state
        public = self.game_orchestrator.game_spec.render_public_context(state)
        roles_overview = ", ".join(
            [f"{aid}:{role.name}" for aid, role in state.agent_roles.items()]
        )

        # Per-agent private info and allowed tools
        role = state.agent_roles.get(str(agent_uid)) or state.agent_roles.get(agent_uid)
        private = ""
        if role:
            private = self.game_orchestrator.game_spec.render_private_context(
                str(agent_uid), state
            )
        allowed_tools = getattr(self.game_orchestrator.game_spec, "allowed_tools", None)
        tool_line = (
            f"Allowed tools: {sorted(list(allowed_tools))}"
            if allowed_tools
            else "Use the provided game tools."
        )
        per_agent_text = (
            f"Agent {agent_uid} ({role.name if role else ''}): "
            f"{private if private else 'No private info.'} {tool_line}"
        )

        image_note = ""
        if state.secret_state.get("latest_image_path"):
            image_note = f"\nImage available: {state.secret_state.get('latest_image_path')}"

        return (
            f"{public}\nRoles: {roles_overview}\n{per_agent_text}{image_note}\nTask: {fallback_instruction}"
        )

    def _maybe_update_game(self) -> None:
        if not self.game_orchestrator or not self.game_orchestrator.state:
            return
        spec = self.game_orchestrator.game_spec
        if hasattr(spec, "maybe_auto_resolve"):
            spec.maybe_auto_resolve(self.game_orchestrator.state, self.game_orchestrator.env)
        terminal, outcome = spec.check_end_condition(
            self.game_orchestrator.state, self.game_orchestrator.env
        )
        self.game_orchestrator.state.terminal = terminal
        self.game_orchestrator.state.outcome = outcome

    def _sync_held_objects(self, observations: Dict[str, Any]) -> None:
        """
        Record currently held objects per agent into game state (if available) so game tools can enforce pick requirements.
        """
        if not self.game_orchestrator or not self.game_orchestrator.state:
            return
        held_map: Dict[str, Any] = {}
        for agent in self.agents.values():
            key = f"agent_{agent.uid}_is_holding"
            raw = observations.get(key, None)
            grasp_handle = None
            grasp_idx = None
            grasp_state = None
            grasp_name = None
            try:
                gm = self.env_interface.sim.agents_mgr[agent.uid].grasp_mgr
                grasp_state = getattr(gm, "is_grasped", None)
                grasp_idx = getattr(gm, "snap_idx", None)
                if grasp_idx is not None:
                    obj = get_obj_from_id(self.env_interface.sim, int(grasp_idx))
                    if obj is not None and hasattr(obj, "handle"):
                        grasp_handle = obj.handle
                        # Translate to WG name if possible for human-readable comparisons
                        try:
                            wg = getattr(self.env_interface, "full_world_graph", None)
                            if wg is not None and wg.has_node_with_sim_handle(grasp_handle):
                                grasp_name = wg.get_node_from_sim_handle(grasp_handle).name
                        except Exception:
                            grasp_name = None
            except Exception as exc:
                grasp_state = f"grasp_mgr_err:{exc}"
            held_obj = None
            # Normalize observed value
            if raw is not None:
                try:
                    if hasattr(raw, "item"):
                        raw = raw.item()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                except Exception:
                    pass
                if isinstance(raw, str) and raw != "":
                    held_obj = raw
                elif isinstance(raw, (int, float)) or raw is True:
                    if grasp_name is not None:
                        held_obj = grasp_name
                    elif grasp_handle is not None:
                        held_obj = grasp_handle
                else:
                    if grasp_name is not None:
                        held_obj = grasp_name
                    elif grasp_handle is not None:
                        held_obj = grasp_handle
            else:
                if grasp_name is not None:
                    held_obj = grasp_name
                elif grasp_handle is not None:
                    held_obj = grasp_handle

            if held_obj is not None:
                held_map[str(agent.uid)] = held_obj
        if held_map:
            self.game_orchestrator.state.secret_state["held_objects"] = held_map

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = ""
    ) -> Dict[str, Any]:
        """
        Mirrors EvaluationRunner.run_instruction with game-state awareness.
        """
        # Derive base instruction
        base_instruction = (
            instruction
            if instruction is not None
            else self.base_instruction
            if self.base_instruction is not None
            else None
        )
        if base_instruction is None:
            # fallback to episode instruction
            curr_env = self.env_interface.env.env.env._env
            base_instruction = curr_env.current_episode.instruction

        # Start the game if it hasn't been started yet
        if self.game_orchestrator and self.game_orchestrator.state is None:
            agent_ids = [str(agent.uid) for agent in self.agents.values()]
            self.game_orchestrator.start(agent_ids)
            self._inject_game_tools()
            print("Game + PARTNR tools per agent:")
            for agent in self.agents.values():
                tool_names = sorted(list(agent.tools.keys()))
                print(f"Agent {agent.uid}: {tool_names}")

        # Rest of loop follows EvaluationRunner.run_instruction
        t_0 = time.time()
        total_step_count = 1
        self.reset_planners()
        # Per-agent instructions
        for agent in self.agents.values():
            agent.composed_instruction = self._compose_instruction(
                base_instruction, agent.uid
            )
        self.initialize_instruction_metadata(base_instruction, output_name)
        observations = self.env_interface.get_observations()
        # Sync held objects at episode start
        self._sync_held_objects(observations)
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
            "num_steps": 0.0,
        }
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        low_level_actions: Any = []
        should_end = False
        # FPV bookkeeping
        self._fpv_frames = {}
        self._fpv_missing = set()
        self._fpv_started = set()

        # Capture an initial frame so FPV videos are not empty if no actions execute
        if self.evaluation_runner_config.save_video:
            try:
                self.dvu._store_for_video(observations, {}, popup_images={})
                self._store_first_person_frames(observations)
            except Exception:
                pass

        while not should_end:
            if (
                "print" in planner_info
                and len(planner_info["print"])
                and self.evaluation_runner_config.do_print
            ):
                rollout_print(planner_info["print"])

            if len(low_level_actions) > 0:
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                observations = self.env_interface.parse_observations(obs)
                self._sync_held_objects(observations)
                if self.evaluation_runner_config.save_video:
                    fp_popups = self._save_first_person_popups(
                        observations, total_step_count
                    )
                    popup_images = planner_info.get("popup_images", {}) or {}
                    if fp_popups:
                        popup_images = {**popup_images, **fp_popups}
                    self.dvu._store_for_video(
                        observations,
                        planner_info.get("high_level_actions", {}),
                        popup_images=popup_images,
                    )
                    # Also capture first-person frames for per-agent videos
                    self._store_first_person_frames(observations)

            # Update game state based on latest env situation
            self._maybe_update_game()

            # refresh game tools in case availability changed (e.g., entering bomb room)
            self._inject_game_tools()

            # Get next low level actions
            # Use agent-specific instructions where possible
            agent_instructions = {
                agent.uid: getattr(agent, "composed_instruction", base_instruction)
                for agent in self.agents.values()
            }
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                agent_instructions, observations, self.env_interface.world_graph
            )

            # If manual planners returned prebuilt frames (they execute skills internally),
            # stash them so the main runner still produces a cumulative video.
            if (
                self.evaluation_runner_config.save_video
                and planner_info.get("manual_video_frames")
            ):
                try:
                    for frames in planner_info.get("manual_video_frames", {}).values():
                        if isinstance(frames, list):
                            self.dvu.frames.extend(frames)
                except Exception:
                    pass
                # Avoid logging huge frame blobs downstream.
                planner_info.pop("manual_video_frames", None)
            # Manual planner may also pass pre-captured FPV frames; merge them into the recorder.
            if (
                self.evaluation_runner_config.save_video
                and planner_info.get("manual_fpv_frames")
                and getattr(self, "_fpv_recorder", None) is not None
                and not getattr(self, "_fpv_recorder_failed", False)
            ):
                try:
                    for agent_name, frames in planner_info.get("manual_fpv_frames", {}).items():
                        if not isinstance(frames, list):
                            continue
                        self._fpv_recorder._frames.setdefault(agent_name, []).extend(frames)
                except Exception:
                    pass
                planner_info.pop("manual_fpv_frames", None)

            # Prepare popup images for subsequent video overlay
            if (
                self.game_orchestrator
                and self.game_orchestrator.state
                and self.game_orchestrator.state.secret_state.get("latest_image_path")
            ):
                img_path = self.game_orchestrator.state.secret_state.get(
                    "latest_image_path"
                )
                planner_info["popup_images"] = {
                    agent.uid: img_path for agent in self.agents.values()
                }
            else:
                planner_info["popup_images"] = {}

        # Allow game termination to short-circuit the loop
            if self.game_orchestrator and self.game_orchestrator.state:
                if self.game_orchestrator.state.terminal and not should_end:
                    print(
                        f"[BombGame] Game ended with outcome: {self.game_orchestrator.state.outcome}"
                    )
                should_end = should_end or self.game_orchestrator.state.terminal

            curr_env = self.env_interface.env.env.env._env
            if total_step_count > curr_env._max_episode_steps:
                should_end = True

            measure_names = [
                "auto_eval_proposition_tracker",
                "task_constraint_validation",
                "task_percent_complete",
                "task_state_success",
                "task_evaluation_log",
                "task_explanation",
            ]
            measures_to_log = [
                "task_percent_complete",
                "task_state_success",
                "task_explanation",
            ]
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in info:
                        info[measure_name] = measures[measure_name].get_metric()

            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = info["num_steps"]

            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                        is_human_wg=int(agent_id) == self.env_interface.human_agent_uid
                    )
                    for agent_id in range(len(self.agents))
                }

            # Prepare popup images for video overlay downstream
            if (
                self.game_orchestrator
                and self.game_orchestrator.state
                and self.game_orchestrator.state.secret_state.get("latest_image_path")
            ):
                img_path = self.game_orchestrator.state.secret_state.get(
                    "latest_image_path"
                )
                planner_info["popup_images"] = {
                    agent.uid: img_path for agent in self.agents.values()
                }
            else:
                planner_info["popup_images"] = {}

            copy_planner_info = copy.deepcopy(planner_info)
            # Ensure high_level_actions has entries for all agents marked as replanned
            # to avoid assertion errors in update_agent_action_history
            if "replanned" in copy_planner_info and "high_level_actions" in copy_planner_info:
                for agent_id, was_replanned in copy_planner_info["replanned"].items():
                    if was_replanned and agent_id not in copy_planner_info["high_level_actions"]:
                        # Add a placeholder action for agents that replanned but have no action
                        copy_planner_info["high_level_actions"][agent_id] = (
                            "NoAction",
                            None,
                            "No action parsed from LLM response",
                        )
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)
            planner_infos.append(copy_planner_info)

            # Incremental log save (lightweight, just JSON)
            self._flush_planner_log(planner_infos)

            total_step_count += 1

        if (
            "print" in planner_info
            and len(planner_info["print"])
            and self.evaluation_runner_config.do_print
        ):
            rollout_print(planner_info["print"])

        if self.evaluation_runner_config.save_video:
            self.dvu._make_video(play=False, postfix=self.episode_filename)
            self._make_first_person_videos()

        self._log_planner_data(planner_infos)

        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime
        info |= planner_info
        if self.game_orchestrator and self.game_orchestrator.state:
            info["game_outcome"] = self.game_orchestrator.state.outcome
            # Emit a summary line on completion.
            if self.game_orchestrator.state.terminal:
                print(
                    f"[Game] Final outcome: {self.game_orchestrator.state.outcome}"
                )
        return info
