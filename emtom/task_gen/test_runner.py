import json

from emtom.task_gen.runner import compute_calibration_stats


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
