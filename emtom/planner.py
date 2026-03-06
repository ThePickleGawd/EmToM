"""
EmtomPlanner - EMTOM-specific LLM planner wrapper with optional vision support.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from hydra.utils import instantiate
from omegaconf import OmegaConf

from habitat_llm.planner import LLMPlanner

from emtom.vision import (
    build_candidate_frame_set,
    load_frame_as_data_url,
    parse_selector_response,
)


class EmtomPlanner(LLMPlanner):
    """LLMPlanner subclass with ReAct logging and benchmark vision support."""

    AGENT_COLORS = [
        "\033[94m",  # Blue
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
    ]
    RESET = "\033[0m"

    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        self._vision_mode = (
            str(getattr(self.env_interface.conf, "benchmark_observation_mode", "text")).lower()
            == "vision"
        )
        self._vision_config = getattr(self.env_interface.conf, "benchmark_vision", None)
        self._visual_context: Dict[str, Any] = {"turn": 0, "available_frames": []}
        self._selected_frames: List[Dict[str, Any]] = []
        self._selector_history: List[Dict[str, Any]] = []
        self._selector_call_count = 0
        self._selector_llm = None
        self._selector_prompt_template = ""
        if self._vision_mode:
            selector_llm_conf = OmegaConf.create(
                OmegaConf.to_container(self.planner_config.llm, resolve=True)
            )
            selector_llm_factory = instantiate(selector_llm_conf.llm)
            self._selector_llm = selector_llm_factory(selector_llm_conf)
            self._selector_prompt_template = self._load_selector_prompt_template()

    def reset(self):
        super().reset()
        self._visual_context = {"turn": 0, "available_frames": []}
        self._selected_frames = []
        self._selector_history = []
        self._selector_call_count = 0

    def set_visual_context(
        self,
        turn: int,
        available_frames: Sequence[Dict[str, Any]],
    ) -> None:
        self._visual_context = {
            "turn": int(turn),
            "available_frames": list(available_frames or []),
        }
        self._selected_frames = []

    def get_selector_metrics(self) -> Dict[str, Any]:
        model_name = None
        llm_provider = None
        if hasattr(self.planner_config, "llm"):
            model_name = getattr(self.planner_config.llm.generation_params, "model", None)
            llm_provider = getattr(self.planner_config.llm.llm, "_target_", "")

        return {
            "enabled": self._vision_mode and self._selector_llm is not None,
            "calls": self._selector_call_count,
            "model": model_name,
            "provider_target": llm_provider,
            "history": list(self._selector_history),
        }

    def _log_high_level_actions(
        self,
        high_level_actions: Dict[int, Any],
        thought: Optional[str] = None,
    ) -> None:
        for uid in sorted(high_level_actions.keys()):
            action_tuple = high_level_actions[uid]
            if action_tuple and action_tuple[0]:
                action_name, action_arg, _ = action_tuple
                color = self.AGENT_COLORS[uid % len(self.AGENT_COLORS)]

                if thought:
                    print(f"{thought}", flush=True)
                print(
                    f"{color}Agent_{uid}:{self.RESET} {action_name}[{action_arg}]",
                    flush=True,
                )

    def _vision_setting(self, name: str, default: Any) -> Any:
        if self._vision_config is None:
            return default
        return getattr(self._vision_config, name, default)

    def _load_selector_prompt_template(self) -> str:
        import habitat_llm

        instruct_name = str(self._vision_setting("selector_prompt_name", "emtom_frame_selector"))
        habitat_llm_dir = os.path.dirname(getattr(habitat_llm, "__file__", ""))
        try:
            instruct_cfg = OmegaConf.load(
                f"{habitat_llm_dir}/conf/instruct/{instruct_name}.yaml"
            )
            return str(getattr(instruct_cfg, "prompt", ""))
        except Exception:
            return (
                "You are selecting first-person images for an embodied EMTOM agent.\n\n"
                "Review the acting-agent context and the candidate frames attached below.\n"
                "Choose the {min_frames} to {max_frames} most decision-relevant frames for the next action.\n\n"
                "Acting-agent context:\n{context}\n\n"
                "Task instruction:\n{instruction}\n\n"
                "Return exactly one line in this format:\n"
                "SELECTED_FRAMES: frame_id_1, frame_id_2"
            )

    def _build_selector_prompt(
        self,
        instruction: str,
        candidate_handles: Sequence[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        min_frames = int(self._vision_setting("selector_min_frames", 1))
        max_frames = int(self._vision_setting("selector_max_frames", 5))
        selector_text = (self._selector_prompt_template or "").format(
            context=self.curr_prompt,
            instruction=instruction,
            min_frames=min_frames,
            max_frames=max_frames,
        ).rstrip()
        text_parts = [
            (
                "text",
                f"{selector_text}\n\nCandidate frames:\n",
            )
        ]

        for handle in candidate_handles:
            text_parts.append(
                (
                    "text",
                    (
                        f"- {handle['frame_id']} | turn={handle['turn']} | "
                        f"skill_step={handle['skill_step']} | sim_step={handle['sim_step']} | "
                        f"kind={handle['kind']}\n"
                    ),
                )
            )
            try:
                text_parts.append(("image", load_frame_as_data_url(handle)))
            except Exception as exc:
                text_parts.append(
                    (
                        "text",
                        f"[Warning: could not load {handle['frame_id']}: {exc}]\n",
                    )
                )

        text_parts.append(("text", "\nSELECTED_FRAMES:"))
        return text_parts

    def _select_visual_frames(
        self,
        instruction: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        available_frames = list(self._visual_context.get("available_frames", []))
        if not self._vision_mode or self._selector_llm is None or not available_frames:
            return [], {
                "turn": int(self._visual_context.get("turn", 0)),
                "available_frame_count": len(available_frames),
                "candidate_frame_count": 0,
                "selected_frame_ids": [],
                "raw_response": "",
            }

        min_select = int(self._vision_setting("selector_min_frames", 1))
        max_select = int(self._vision_setting("selector_max_frames", 5))
        max_candidates = int(self._vision_setting("selector_max_candidates", 12))

        candidate_handles = build_candidate_frame_set(available_frames, max_candidates=max_candidates)
        prompt_input = self._build_selector_prompt(instruction, candidate_handles)
        selector_response = self._selector_llm.generate(prompt_input)
        selected = parse_selector_response(
            selector_response,
            candidate_handles,
            min_select=min_select,
            max_select=max_select,
        )
        self._selector_call_count += 1

        selector_info = {
            "turn": int(self._visual_context.get("turn", 0)),
            "available_frame_count": len(available_frames),
            "candidate_frame_count": len(candidate_handles),
            "candidate_frame_ids": [handle["frame_id"] for handle in candidate_handles],
            "selected_frame_ids": [handle["frame_id"] for handle in selected],
            "raw_response": selector_response,
        }
        self._selector_history.append(selector_info)
        return selected, selector_info

    def _build_multimodal_prompt(
        self,
        selected_frames: Sequence[Dict[str, Any]],
    ) -> Union[str, List[Tuple[str, str]]]:
        if not selected_frames:
            return self.curr_prompt

        prompt_input: List[Tuple[str, str]] = [
            (
                "text",
                (
                    f"{self.curr_prompt}\n\n"
                    "Selected first-person visual observations for this turn are attached below.\n"
                    "Use them instead of any synthetic text summary of scene appearance.\n"
                ),
            )
        ]

        for handle in selected_frames:
            prompt_input.append(
                (
                    "text",
                    (
                        f"\nFrame {handle['frame_id']} "
                        f"(turn={handle['turn']}, skill_step={handle['skill_step']}, kind={handle['kind']}):\n"
                    ),
                )
            )
            prompt_input.append(("image", load_frame_as_data_url(handle)))

        return prompt_input

    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, Any],
    ):
        selected_frames: List[Dict[str, Any]] = []
        selector_info: Optional[Dict[str, Any]] = None
        if self._vision_mode:
            selected_frames, selector_info = self._select_visual_frames(instruction)
            self._selected_frames = selected_frames

        prompt_input = self._build_multimodal_prompt(selected_frames)

        if self.planner_config.get("constrained_generation", False):
            llm_response = self.llm.generate(
                prompt_input,
                self.stopword,
                generation_args={
                    "grammar_definition": self.build_response_grammar(
                        world_graph[self._agents[0].uid]
                    )
                },
            )
        else:
            llm_response = self.llm.generate(prompt_input, self.stopword)

        llm_response = self.format_response(llm_response, self.end_expression)

        info = {
            "llm_response": llm_response,
            "selected_frames": [dict(handle) for handle in selected_frames],
        }
        if selector_info is not None:
            info["selector"] = selector_info
        return info

    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, Any],
        verbose: bool = False,
    ):
        low_level_actions, planner_info, is_done = super().get_next_action(
            instruction, observations, world_graph, verbose=verbose
        )
        if self._vision_mode:
            planner_info["selected_frames"] = list(self._selected_frames)
            if self._selector_history:
                planner_info["selector"] = self._selector_history[-1]
        return low_level_actions, planner_info, is_done
