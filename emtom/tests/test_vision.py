from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from emtom.runner.benchmark import BenchmarkRunner
from emtom.runner.base import EMTOMBaseRunner
from emtom.utils.summarize import summarize_results
from emtom.vision import (
    VisualObservationStore,
    build_candidate_frame_set,
    load_frame_as_data_url,
    parse_selector_response,
)


class DummyRunner(EMTOMBaseRunner):
    def run(self, *args, **kwargs):
        raise NotImplementedError


def _make_handles(count: int):
    return [
        {
            "frame_id": f"frame_{idx}",
            "agent_id": "agent_0",
            "turn": 1,
            "frame_index": idx,
            "skill_step": idx,
            "sim_step": idx,
            "kind": "in_action",
            "path": f"/tmp/frame_{idx}.png",
        }
        for idx in range(count)
    ]


def test_textual_visual_summary_toggle():
    text_runner = DummyRunner(OmegaConf.create({"benchmark_observation_mode": "text"}))
    vision_runner = DummyRunner(OmegaConf.create({"benchmark_observation_mode": "vision"}))

    snapshots = ["[Step 30] kitchen_0: mug_0 (on counter_1)."]
    agents_passed = {"agent_1": ("kitchen_0", 30)}

    text_result = text_runner._append_textual_visual_summary("Successful execution!", snapshots, agents_passed)
    vision_result = vision_runner._append_textual_visual_summary("Successful execution!", snapshots, agents_passed)

    assert "Surroundings observed while acting" in text_result
    assert vision_result == "Successful execution!"


def test_visual_store_capture_and_data_url(tmp_path):
    store = VisualObservationStore(str(tmp_path))
    observations = {
        "agent_0_head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    captured = store.capture(
        observations=observations,
        agent_ids=["agent_0"],
        turn=2,
        skill_step=5,
        sim_step=17,
        kind="turn_end",
    )

    handle = captured["agent_0"][0]
    assert handle["frame_id"] == "agent_0_t0002_f0000"
    assert (tmp_path / "agent_0" / "turn_0002").exists()

    data_url = load_frame_as_data_url(handle)
    assert data_url.startswith("data:image/png;base64,")


def test_candidate_downsampling_and_selector_parsing():
    handles = _make_handles(8)
    candidates = build_candidate_frame_set(handles, max_candidates=3)

    assert len(candidates) == 3
    assert candidates[-1]["frame_id"] == "frame_7"

    response = "SELECTED_FRAMES: frame_2, frame_7"
    selected = parse_selector_response(response, handles, min_select=1, max_select=5)

    assert [handle["frame_id"] for handle in selected] == ["frame_2", "frame_7"]

    fallback = parse_selector_response("SELECTED_FRAMES:", candidates, min_select=1, max_select=5)
    assert fallback[-1]["frame_id"] == "frame_7"


def test_action_history_entry_includes_analysis_aliases():
    entry = BenchmarkRunner._make_action_history_entry(
        sim_step=4,
        turn=2,
        agent_id="agent_0",
        action="Navigate[table_22]",
        result="Successful execution!",
        mode="llm",
        skill_steps=12,
        selected_frames=["frame_1"],
        selector={"turn": 1},
    )

    assert entry["agent"] == "agent_0"
    assert entry["agent_id"] == "agent_0"
    assert entry["action"] == "Navigate[table_22]"
    assert entry["action_taken"] == "Navigate[table_22]"
    assert entry["result"] == "Successful execution!"
    assert entry["observation"] == "Successful execution!"
    assert entry["selected_frames"] == ["frame_1"]


def test_summarize_results_prefers_steps_field():
    summary = summarize_results(
        [
            {
                "task_id": "task-1",
                "task_title": "Example",
                "success": True,
                "steps": 6,
                "sim_steps": 99,
                "turns": 3,
                "evaluation": {
                    "completed_subtasks": [0],
                    "total_subtasks": 1,
                    "required_subtasks": 1,
                    "completed_required": 1,
                    "percent_complete": 1.0,
                    "percent_required_complete": 1.0,
                },
            }
        ]
    )

    assert summary["task_results"][0]["steps"] == 6
