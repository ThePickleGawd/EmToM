from __future__ import annotations

import json
from pathlib import Path

from emtom.bulk_report import build_report


def test_build_report_uses_manifest_for_requested_tasks_and_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-03-31_12-00-00-generation"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "requested_tasks": 100,
                "model": "sonnet-4.6",
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "gpu0_slot0_cooperative_worker0001.log").write_text(
        "Tasks generated: 38\n",
        encoding="utf-8",
    )

    report = build_report([str(logs_dir)])

    assert report.models_used == ["sonnet-4.6"]
    assert report.requested_tasks_total == 100
    assert report.total_tasks == 38
    assert report.task_pass_rate == 38.0


def test_build_report_falls_back_to_launcher_log_for_requested_tasks_and_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-03-31_12-00-00-generation"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)

    (run_dir / "launcher.log").write_text(
        "\n".join(
            [
                "Bulk EMTOM Task Generation",
                "Model:              sonnet-4.6",
                "Total tasks:        100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "gpu0_slot0_cooperative_worker0001.log").write_text(
        "Tasks generated: 38\n",
        encoding="utf-8",
    )

    report = build_report([str(logs_dir)])

    assert report.models_used == ["sonnet-4.6"]
    assert report.requested_tasks_total == 100
    assert report.total_tasks == 38
    assert report.task_pass_rate == 38.0
