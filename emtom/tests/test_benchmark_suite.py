from __future__ import annotations

import json
from pathlib import Path

from emtom.evolve.benchmark_wrapper import BenchmarkResults
from emtom.scripts.benchmark_suite import (
    _build_suite_result,
    _parse_suite_model_results,
)


def test_parse_suite_model_results_falls_back_to_nested_parallel_outputs(tmp_path: Path) -> None:
    model_dir = tmp_path / "suite_model"
    task_a = model_dir / "task_a" / "benchmark-2agents" / "results"
    task_b = model_dir / "task_b" / "benchmark-3agents" / "results"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)

    task_a_summary = {
        "run_mode": "standard",
        "results": [
            {
                "task_id": "task_a",
                "title": "Task A",
                "success": True,
                "steps": 5,
                "turns": 5,
                "category": "cooperative",
                "evaluation": {"percent_complete": 1.0},
            }
        ],
    }
    task_b_summary = {
        "run_mode": "standard",
        "results": [
            {
                "task_id": "task_b",
                "title": "Task B",
                "success": False,
                "steps": 9,
                "turns": 9,
                "category": "mixed",
                "evaluation": {"percent_complete": 1.0 / 3.0},
            }
        ],
    }

    (task_a / "benchmark_summary.json").write_text(json.dumps(task_a_summary), encoding="utf-8")
    (task_b / "benchmark_summary.json").write_text(json.dumps(task_b_summary), encoding="utf-8")

    parsed = _parse_suite_model_results(str(model_dir), model="gpt-5.4")

    assert parsed.total == 2
    assert parsed.passed == 1
    assert parsed.failed == 1
    assert parsed.pass_rate == 50.0
    assert {result.task_id for result in parsed.results} == {"task_a", "task_b"}


def test_build_suite_result_marks_partial_when_results_exist_but_command_failed() -> None:
    parsed = BenchmarkResults(
        model="gpt-5.4",
        total=3,
        passed=1,
        failed=2,
        pass_rate=100.0 / 3.0,
    )

    result = _build_suite_result(
        model="gpt-5.4",
        output_dir="/tmp/out",
        return_code=1,
        parsed=parsed,
    )

    assert result.status == "partial"
    assert result.total == 3
    assert result.passed == 1
    assert result.failed == 2
