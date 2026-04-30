from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from emtom.benchmark_metrics import build_single_run_summary
from emtom.evolve.benchmark_wrapper import BenchmarkResults
import emtom.scripts.benchmark_suite as benchmark_suite
from emtom.scripts.benchmark_repeat import parse_args as parse_repeat_args
from emtom.scripts.benchmark_suite import (
    _build_suite_result,
    _parse_suite_model_summary,
    _parse_suite_model_results,
    parse_args,
    render_suite_summary,
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
    summary = build_single_run_summary(
        model="gpt-5.4",
        output_dir="/tmp/out",
        parsed=parsed,
        return_code=1,
    )

    result = _build_suite_result(
        model="gpt-5.4",
        output_dir="/tmp/out",
        return_code=1,
        summary=summary,
    )

    assert result.status == "partial"
    assert result.total == 3
    assert result.passed == 1
    assert result.failed == 2


def test_parse_suite_model_summary_reads_repeated_summary_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "suite_model"
    model_dir.mkdir()
    repeat_summary = {
        "summary_type": "benchmark_repeat",
        "model": "gpt-5.4",
        "num_times": 3,
        "k": 3,
        "completed_runs": 3,
        "scored_task_count": 5,
        "average_pass_rate": 40.0,
        "std_pass_rate": 10.0,
        "pass_at_k": 78.4,
        "pass_power_k": 6.4,
        "runs": [
            {
                "run_index": 1,
                "output_dir": str(model_dir / "run_1"),
                "status": "complete",
                "return_code": 0,
                "total": 5,
                "passed": 2,
                "failed": 3,
                "pass_rate": 40.0,
                "error": "",
            }
        ],
    }
    (model_dir / "benchmark_repeat_summary.json").write_text(
        json.dumps(repeat_summary),
        encoding="utf-8",
    )

    summary = _parse_suite_model_summary(str(model_dir), model="gpt-5.4")

    assert summary.num_times == 3
    assert summary.average_pass_rate == 40.0
    assert summary.pass_at_k == 78.4
    assert summary.pass_power_k == 6.4


def test_parse_args_normalizes_deepseek_alias() -> None:
    args = parse_args(
        [
            "--tasks-dir",
            "data/emtom/tasks",
            "--models",
            "deepseek",
            "gemini-flash",
        ]
    )

    assert args.models == ["deepseek-v3.2", "gemini-flash"]


def test_parse_args_defaults_num_times_to_three() -> None:
    args = parse_args(
        [
            "--tasks-dir",
            "data/emtom/tasks",
            "--models",
            "gpt-5.4",
        ]
    )

    assert args.num_times == 3


def test_parse_args_accepts_task_dir_alias_and_defaults_no_calibration() -> None:
    args = parse_args(
        [
            "--task-dir",
            "data/emtom/tasks",
            "--models",
            "gpt-5.4",
        ]
    )

    assert args.tasks_dir == "data/emtom/tasks"
    assert args.no_calibration is True


def test_benchmark_repeat_parse_args_defaults_num_times_to_three() -> None:
    args = parse_repeat_args(
        [
            "--tasks-dir",
            "data/emtom/tasks",
            "--model",
            "gpt-5.4",
            "--output-dir",
            "outputs/emtom/test",
        ]
    )

    assert args.num_times == 3


def test_benchmark_repeat_parse_args_accepts_task_dir_alias_and_defaults_no_calibration() -> None:
    args = parse_repeat_args(
        [
            "--task-dir",
            "data/emtom/tasks",
            "--model",
            "gpt-5.4",
            "--output-dir",
            "outputs/emtom/test",
        ]
    )

    assert args.tasks_dir == "data/emtom/tasks"
    assert args.no_calibration is True


def test_render_suite_summary_keeps_repeat_metric_columns_distinct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(benchmark_suite, "USE_COLOR", True)
    result = benchmark_suite.SuiteResult(
        model="gpt-5.4",
        status="complete",
        total=5,
        passed=2,
        failed=3,
        pass_rate=40.0,
        output_dir=str(tmp_path / "out"),
        return_code=0,
        num_times=3,
        completed_runs=3,
        average_pass_rate=40.0,
        std_pass_rate=10.0,
        pass_at_k=78.4,
        pass_power_k=6.4,
    )

    rendered = render_suite_summary([result], tmp_path, current_model="gpt-5.4")
    stripped = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    lines = [line for line in stripped.splitlines() if line.strip()]
    header = next(line for line in lines if "Pass@k" in line and "Pass^k" in line)
    row = next(line for line in lines if "gpt-5.4" in line and "complete" in line)

    assert header.split(" | ")[-2:] == ["    Pass@k", "    Pass^k"]
    assert row.split(" | ")[-2:] == ["     78.4%", "      6.4%"]


def test_parse_args_reports_concise_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--models", "deepseek"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Error: one of the arguments" in captured.err
    assert "--tasks-dir" in captured.err
    assert "--task" in captured.err
    assert "Hint: ./emtom/run_emtom.sh benchmark-suite --help" in captured.err
    assert "usage:" not in captured.err.lower()
