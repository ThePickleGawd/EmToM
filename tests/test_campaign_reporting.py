import json
from argparse import Namespace
from pathlib import Path

import pytest

from emtom.scripts import campaign as campaign_mod


def test_summary_payload_aggregates_literal_tom_with_probe_weighting():
    results = [
        {
            "task_id": "task_a",
            "category": "cooperative",
            "success": True,
            "steps": 8,
            "done": True,
            "evaluation": {
                "percent_complete": 1.0,
                "literal_tom_probe_score": 0.5,
                "literal_tom_probe_summary": {
                    "probe_count": 4,
                    "supported_probe_count": 4,
                    "passed_count": 2,
                },
            },
        },
        {
            "task_id": "task_b",
            "category": "mixed",
            "success": False,
            "steps": 10,
            "done": True,
            "evaluation": {
                "percent_complete": 0.5,
                "literal_tom_probe_score": 1.0,
                "literal_tom_probe_summary": {
                    "probe_count": 2,
                    "supported_probe_count": 2,
                    "passed_count": 2,
                },
            },
        },
    ]

    summary = campaign_mod._summary_payload(results, model="gpt-5.2")

    assert summary["pass_rate"] == 50.0
    assert summary["literal_tom_score"] == 66.7
    assert summary["literal_tom_task_count"] == 2
    assert summary["literal_tom_supported_probe_count"] == 6
    assert summary["literal_tom_passed_probe_count"] == 4
    assert summary["category_stats"]["cooperative"]["literal_tom_score"] == 50.0
    assert summary["category_stats"]["mixed"]["literal_tom_score"] == 100.0


def test_generate_report_writes_literal_tom_to_leaderboard(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True)
    leaderboard_file = results_dir / "leaderboard.json"

    monkeypatch.setattr(campaign_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(campaign_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(campaign_mod, "LEADERBOARD_FILE", leaderboard_file)

    run_dir = runs_dir / "gpt-5.2_text_cooperative"
    run_dir.mkdir()
    with open(run_dir / "benchmark_summary.json", "w") as f:
        json.dump(
            {
                "model": "gpt-5.2",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "pass_rate": 100.0,
                "results": [
                    {
                        "task_id": "task_a",
                        "title": "Task A",
                        "category": "cooperative",
                        "success": True,
                        "evaluation": {
                            "percent_complete": 1.0,
                            "literal_tom_probe_score": 0.75,
                            "literal_tom_probe_summary": {
                                "probe_count": 4,
                                "supported_probe_count": 4,
                                "passed_count": 3,
                            },
                        },
                    }
                ],
            },
            f,
            indent=2,
        )

    campaign = {
        "campaign_id": "active",
        "label": "Active Campaign",
        "status": "active",
        "models": ["gpt-5.2"],
        "modes": ["text"],
        "runs": {
            "gpt-5.2_text_cooperative": {
                "type": "solo",
                "status": "complete",
                "model": "gpt-5.2",
                "mode": "text",
                "category": "cooperative",
            }
        },
    }

    campaign_mod._generate_report(campaign)

    leaderboard = json.load(open(leaderboard_file))
    solo = leaderboard["solo"]["gpt-5.2_text"]
    assert solo["categories"]["cooperative"]["literal_tom_score"] == 75.0
    assert solo["overall"]["literal_tom_score"] == 75.0


def test_generate_report_includes_failed_runs_with_synced_summary(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True)
    leaderboard_file = results_dir / "leaderboard.json"

    monkeypatch.setattr(campaign_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(campaign_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(campaign_mod, "LEADERBOARD_FILE", leaderboard_file)

    run_dir = runs_dir / "gpt-5.2_text_cooperative"
    run_dir.mkdir()
    with open(run_dir / "benchmark_summary.json", "w") as f:
        json.dump(
            {
                "model": "gpt-5.2",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "pass_rate": 100.0,
                "results": [
                    {
                        "task_id": "task_a",
                        "title": "Task A",
                        "category": "cooperative",
                        "success": True,
                        "evaluation": {"percent_complete": 1.0},
                    }
                ],
            },
            f,
            indent=2,
        )

    campaign = {
        "campaign_id": "active",
        "label": "Active Campaign",
        "status": "active",
        "models": ["gpt-5.2"],
        "modes": ["text"],
        "runs": {
            "gpt-5.2_text_cooperative": {
                "type": "solo",
                "status": "failed",
                "model": "gpt-5.2",
                "mode": "text",
                "category": "cooperative",
            }
        },
    }

    campaign_mod._generate_report(campaign)

    leaderboard = json.load(open(leaderboard_file))
    assert leaderboard["solo"]["gpt-5.2_text"]["overall"]["total"] == 1


def test_collect_results_filters_stale_parallel_tasks(tmp_path, monkeypatch):
    runs_dir = tmp_path / "results" / "runs"
    runs_dir.mkdir(parents=True)
    monkeypatch.setattr(campaign_mod, "RUNS_DIR", runs_dir)

    output_dir = tmp_path / "outputs" / "resume"
    for task_id in ("task_a", "task_removed"):
        results_dir = output_dir / task_id / "benchmark-2agents" / "results"
        task_dir = results_dir / task_id
        (task_dir / "planner-log").mkdir(parents=True, exist_ok=True)
        with open(results_dir / "benchmark_summary.json", "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "task_id": task_id,
                            "title": task_id,
                            "category": "cooperative",
                            "success": True,
                            "evaluation": {"percent_complete": 1.0},
                        }
                    ]
                },
                f,
                indent=2,
            )
        (task_dir / "planner-log" / "planner-log-001.json").write_text("{}")

    campaign_mod._collect_results(
        str(output_dir),
        "gpt-5.2_text_cooperative",
        model="gpt-5.2",
        include_task_ids={"task_a"},
    )

    summary = json.load(open(runs_dir / "gpt-5.2_text_cooperative" / "benchmark_summary.json"))
    assert [result["task_id"] for result in summary["results"]] == ["task_a"]
    assert (runs_dir / "gpt-5.2_text_cooperative" / "tasks" / "task_a" / "planner-log.json").exists()
    assert not (runs_dir / "gpt-5.2_text_cooperative" / "tasks" / "task_removed").exists()


def test_cmd_run_interrupt_syncs_partial_results_and_resume_state(tmp_path, monkeypatch):
    project_root = tmp_path
    results_dir = project_root / "data" / "emtom" / "results"
    runs_dir = results_dir / "runs"
    leaderboard_file = results_dir / "leaderboard.json"
    campaign_file = results_dir / "campaign.json"
    outputs_dir = project_root / "outputs" / "emtom"
    tasks_dir = project_root / "tasks"

    runs_dir.mkdir(parents=True)
    tasks_dir.mkdir(parents=True)

    monkeypatch.setattr(campaign_mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(campaign_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(campaign_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(campaign_mod, "LEADERBOARD_FILE", leaderboard_file)
    monkeypatch.setattr(campaign_mod, "CAMPAIGN_FILE", campaign_file)
    monkeypatch.setattr(campaign_mod, "OUTPUTS_DIR", outputs_dir)

    for task_id in ("task_a", "task_b"):
        with open(tasks_dir / f"{task_id}.json", "w") as f:
            json.dump({"task_id": task_id, "category": "cooperative"}, f)

    run_key = "gpt-5.2_text_cooperative"
    with open(campaign_file, "w") as f:
        json.dump(
            {
                "campaign_id": "active",
                "label": "Active Campaign",
                "status": "active",
                "models": ["gpt-5.2"],
                "modes": ["text"],
                "task_counts": {"cooperative": 2},
                "task_total": 2,
                "tasks_dir": "tasks",
                "runs": {
                    run_key: {
                        "type": "solo",
                        "status": "pending",
                        "model": "gpt-5.2",
                        "mode": "text",
                        "category": "cooperative",
                    }
                },
            },
            f,
            indent=2,
        )

    def fake_execute(run_key_arg, run_def, benchmark_dir, output_dir, max_workers):
        results_dir = output_dir / "task_a" / "benchmark-2agents" / "results"
        task_dir = results_dir / "task_a"
        (task_dir / "planner-log").mkdir(parents=True, exist_ok=True)
        with open(results_dir / "benchmark_summary.json", "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "task_id": "task_a",
                            "title": "Task A",
                            "category": "cooperative",
                            "success": True,
                            "turns": 3,
                            "steps": 7,
                            "evaluation": {"percent_complete": 1.0},
                        }
                    ]
                },
                f,
                indent=2,
            )
        (task_dir / "planner-log" / "planner-log-001.json").write_text("{}")
        raise KeyboardInterrupt()

    monkeypatch.setattr(campaign_mod, "_execute_benchmark", fake_execute)

    with pytest.raises(KeyboardInterrupt):
        campaign_mod.cmd_run(Namespace(only=None, models=None, max_workers=2))

    campaign = json.load(open(campaign_file))
    run = campaign["runs"][run_key]
    assert run["status"] == "failed"
    assert Path(run["output_dir"]).exists()
    assert run["completed_tasks"] == ["task_a"]

    summary = json.load(open(runs_dir / run_key / "benchmark_summary.json"))
    assert [result["task_id"] for result in summary["results"]] == ["task_a"]

    leaderboard = json.load(open(leaderboard_file))
    assert leaderboard["solo"]["gpt-5.2_text"]["overall"]["total"] == 1


def test_archive_moves_active_campaign_into_archive_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs"
    archives_dir = results_dir / "archives"
    results_dir.mkdir()
    runs_dir.mkdir()

    campaign_file = results_dir / "campaign.json"
    leaderboard_file = results_dir / "leaderboard.json"
    with open(campaign_file, "w") as f:
        json.dump(
            {
                "campaign_id": "active",
                "label": "Active Campaign",
                "status": "active",
                "created_at": "2026-03-19T00:00:00+00:00",
                "updated_at": "2026-03-19T00:00:00+00:00",
                "models": ["gpt-5.2"],
                "modes": ["text"],
                "task_total": 1,
                "runs": {},
            },
            f,
            indent=2,
        )
    leaderboard_file.write_text("{}")
    (runs_dir / "example").mkdir()

    monkeypatch.setattr(campaign_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(campaign_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(campaign_mod, "ARCHIVES_DIR", archives_dir)
    monkeypatch.setattr(campaign_mod, "CAMPAIGN_FILE", campaign_file)
    monkeypatch.setattr(campaign_mod, "LEADERBOARD_FILE", leaderboard_file)

    campaign_mod.cmd_archive(
        Namespace(
            campaign_id="legacy_v1",
            label="Legacy v1",
            reason="benchmark semantics changed",
        )
    )

    archived_campaign = json.load(open(archives_dir / "legacy_v1" / "campaign.json"))
    assert archived_campaign["status"] == "archived"
    assert archived_campaign["campaign_id"] == "legacy_v1"
    assert archived_campaign["archive_reason"] == "benchmark semantics changed"
    assert not campaign_file.exists()
    assert (archives_dir / "legacy_v1" / "leaderboard.json").exists()
    assert (archives_dir / "legacy_v1" / "runs").exists()
