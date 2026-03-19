import json
from argparse import Namespace

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
