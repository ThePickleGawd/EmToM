import json

from emtom.task_gen.runner import compute_calibration_stats, _write_sampled_tasks_summary


def _write_task(path, title, tom_level, calibration):
    path.write_text(
        json.dumps(
            {
                "task_id": title,
                "title": title,
                "task": title,
                "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
                "tom_level": tom_level,
                "calibration": calibration,
            }
        )
    )


def test_compute_calibration_stats_ignores_k_zero_and_merges_dirs(tmp_path):
    model = "gpt-5.2"
    seed_dir = tmp_path / "seed"
    out_dir = tmp_path / "out"
    seed_dir.mkdir()
    out_dir.mkdir()

    _write_task(
        seed_dir / "k0.json",
        "k0",
        0,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": True, "progress": 1.0},
            }
        ],
    )
    _write_task(
        seed_dir / "fail.json",
        "fail",
        1,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": False, "progress": 0.2},
            }
        ],
    )
    _write_task(
        out_dir / "pass.json",
        "pass",
        2,
        [
            {
                "run_mode": "standard",
                "agent_models": {"agent_0": model, "agent_1": model},
                "results": {"passed": True, "progress": 1.0},
            }
        ],
    )

    stats = compute_calibration_stats([str(seed_dir), str(out_dir)], model)

    assert stats["excluded_tom_level_zero"] == 1
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["total"] == 2


def test_write_sampled_tasks_summary_includes_gap_labels_and_analysis_files(tmp_path):
    model = "gpt-5.2"
    task_path = tmp_path / "task_1.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "gap-task",
                "title": "Gap Task",
                "task": "Move the bottle to the table.",
                "category": "cooperative",
                "num_agents": 2,
                "active_mechanics": ["room_restriction", "limited_bandwidth"],
                "problem_pddl": "(:goal (and (is_on_top bottle_1 table_1) (K agent_0 (is_open cabinet_1))))",
                "agent_actions": {"agent_0": ["Wait"], "agent_1": ["Wait"]},
                "calibration": [
                    {
                        "run_mode": "standard",
                        "agent_models": {"agent_0": model, "agent_1": model},
                        "steps": 12,
                        "results": {"passed": False, "progress": 0.33},
                        "trajectory": [
                            {
                                "turn": 1,
                                "agents": {
                                    "agent_0": {"action": "Navigate[kitchen_1]"},
                                    "agent_1": {"action": "Wait"},
                                },
                                "subtasks_completed": [],
                            }
                        ],
                    },
                    {
                        "run_mode": "baseline",
                        "agent_models": {"agent_0": model, "agent_1": model},
                        "steps": 8,
                        "results": {"passed": True, "progress": 1.0},
                        "trajectory": [
                            {
                                "turn": 1,
                                "agents": {
                                    "agent_0": {"action": "Communicate[agent_1, table_1]"},
                                    "agent_1": {"action": "Navigate[kitchen_1]"},
                                },
                                "subtasks_completed": ["placed bottle_1"],
                            }
                        ],
                    },
                ],
            }
        )
    )

    _write_sampled_tasks_summary(tmp_path, model=model)

    summary_text = (tmp_path / "SUMMARY.md").read_text()
    analysis_text = (tmp_path / "task_1_analysis.md").read_text()

    assert "GOOD_HARD_GAP" in summary_text
    assert "Standard: FAIL (progress=33%, steps=12)" in summary_text
    assert "Baseline: PASS (progress=100%, steps=8)" in summary_text
    assert "task_1_analysis.md" in summary_text

    assert "Gap label: GOOD_HARD_GAP" in analysis_text
    assert "Baseline solved while standard failed" in analysis_text
    assert "## Standard Trajectory Digest" in analysis_text
    assert "## Baseline Trajectory Digest" in analysis_text
