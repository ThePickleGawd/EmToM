from __future__ import annotations

from pathlib import Path

from emtom.scripts.benchmark_suite import SuiteResult, render_suite_summary


def test_render_suite_summary_lists_each_model_pass_rate() -> None:
    results = [
        SuiteResult(
            model="opus",
            status="complete",
            total=38,
            passed=9,
            failed=29,
            pass_rate=23.7,
            output_dir="/tmp/opus",
            return_code=0,
        ),
        SuiteResult(
            model="kimi-k2.5",
            status="complete",
            total=38,
            passed=4,
            failed=34,
            pass_rate=10.5,
            output_dir="/tmp/kimi",
            return_code=0,
        ),
    ]

    summary = render_suite_summary(results, Path("/tmp/tasks"))

    assert "BENCHMARK SUITE SUMMARY" in summary
    assert "opus" in summary
    assert "23.7%" in summary
    assert "kimi-k2.5" in summary
    assert "10.5%" in summary


def test_render_suite_summary_shows_current_model_and_counts() -> None:
    results = [
        SuiteResult(
            model="sonnet-4.6",
            status="running",
            total=12,
            passed=5,
            failed=7,
            pass_rate=41.7,
            output_dir="/tmp/sonnet",
            return_code=-1,
        ),
    ]

    summary = render_suite_summary(
        results,
        Path("/tmp/tasks"),
        current_model="sonnet-4.6",
    )

    assert "Current model: sonnet-4.6" in summary
    assert "running" in summary
    assert "5" in summary
    assert "12" in summary
    assert "41.7%" in summary
