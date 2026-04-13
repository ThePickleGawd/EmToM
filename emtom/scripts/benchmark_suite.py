#!/usr/bin/env python3
"""Run the same benchmark task folder across multiple models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from emtom.evolve.benchmark_wrapper import BenchmarkResults, parse_benchmark_results


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_EMTOM = PROJECT_ROOT / "emtom" / "run_emtom.sh"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"
USE_COLOR = sys.stdout.isatty()
SUMMARY_REFRESH_SECONDS = 600

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


@dataclass
class SuiteResult:
    model: str
    status: str
    total: int
    passed: int
    failed: int
    pass_rate: Optional[float]
    output_dir: str
    return_code: int
    error: str = ""


def _style(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    active_codes = [code for code in codes if code]
    if not active_codes:
        return text
    return "".join(active_codes) + text + RESET


def _resolve_tasks_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_task_file(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _format_rate(value: Optional[float]) -> str:
    return f"{value:.1f}%" if value is not None else "--"


def _status_style(status: str) -> tuple[str, ...]:
    if status == "complete":
        return (GREEN, BOLD)
    if status == "partial":
        return (YELLOW, BOLD)
    if status == "failed":
        return (RED, BOLD)
    if status == "running":
        return (CYAN, BOLD)
    return (DIM,)


def _rate_style(value: Optional[float], status: str) -> tuple[str, ...]:
    if value is None:
        return (DIM,)
    if status == "failed":
        return (RED, BOLD)
    if value >= 50:
        return (GREEN, BOLD)
    if value >= 25:
        return (YELLOW, BOLD)
    return (RED, BOLD)


def render_suite_summary(
    results: List[SuiteResult],
    tasks_dir: Path,
    current_model: Optional[str] = None,
) -> str:
    width = 88
    lines = [
        "",
        _style("=" * width, BOLD, CYAN),
        _style("BENCHMARK SUITE SUMMARY", BOLD, WHITE),
        _style("=" * width, BOLD, CYAN),
        f"Tasks dir: {_style(str(tasks_dir), WHITE)}",
    ]
    if current_model:
        lines.append(f"Current model: {_style(current_model, BOLD, MAGENTA)}")
    lines.extend([
        "",
        _style(f"{'Model':<24} {'Status':<10} {'Passed':>8} {'Total':>8} {'Pass rate':>10}", BOLD),
        _style("-" * width, DIM),
    ])
    for result in results:
        passed = str(result.passed) if result.total > 0 else "-"
        total = str(result.total) if result.total > 0 else "-"
        model_label = _style(result.model, BOLD, MAGENTA) if result.model == current_model else _style(result.model, WHITE)
        status_label = _style(f"{result.status:<10}", *_status_style(result.status))
        passed_label = _style(
            f"{passed:>8}",
            GREEN if result.passed > 0 else DIM,
            BOLD if result.passed > 0 else "",
        )
        total_label = _style(f"{total:>8}", CYAN if result.total > 0 else DIM)
        rate_label = _style(f"{_format_rate(result.pass_rate):>10}", *_rate_style(result.pass_rate, result.status))
        lines.append(
            f"{model_label:<24} {status_label} {passed_label} {total_label} {rate_label}"
        )
    return "\n".join(lines)


def _write_suite_summary(
    suite_dir: Path,
    tasks_dir: Path,
    args: argparse.Namespace,
    results: List[SuiteResult],
    current_model: Optional[str] = None,
) -> None:
    """Persist the current suite state for live monitoring."""
    summary_text = render_suite_summary(results, tasks_dir, current_model=current_model)
    print(summary_text)

    completed = sum(1 for result in results if result.status in {"complete", "partial", "failed"})
    pending = max(len(results) - completed, 0)

    summary_path = suite_dir / "suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks_dir": str(tasks_dir),
                "models": args.models,
                "observation_mode": args.observation_mode,
                "run_mode": args.run_mode,
                "category": args.category,
                "completed_models": completed,
                "pending_models": pending,
                "results": [asdict(result) for result in results],
            },
            f,
            indent=2,
        )

    live_text_path = suite_dir / "suite_summary.txt"
    live_text_path.write_text(summary_text + "\n", encoding="utf-8")
    print(f"\nSaved suite summary to: {summary_path}")


def _parse_nested_benchmark_results(output_dir: str, model: str) -> BenchmarkResults:
    """Aggregate per-task benchmark summaries written by parallel benchmark mode."""
    output_path = Path(output_dir)
    summary_files = sorted(output_path.glob("**/results/benchmark_summary.json"))
    if not summary_files:
        raise FileNotFoundError(
            f"No nested benchmark_summary.json found under '{output_dir}'."
        )

    total = 0
    passed = 0
    failed = 0
    all_results = []
    parse_errors = []
    for summary_file in summary_files:
        try:
            parsed = parse_benchmark_results(str(summary_file.parent.parent), model=model)
        except Exception as exc:
            parse_errors.append(f"{summary_file}: {exc}")
            continue
        total += parsed.total
        passed += parsed.passed
        failed += parsed.failed
        all_results.extend(parsed.results)

    if total == 0:
        details = parse_errors[0] if parse_errors else f"No parseable nested results under '{output_dir}'."
        raise RuntimeError(details)

    return BenchmarkResults(
        model=model,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=(passed / total * 100.0) if total else 0.0,
        results=all_results,
    )


def _parse_suite_model_results(output_dir: str, model: str) -> BenchmarkResults:
    """Parse either a direct benchmark output or a benchmark-suite model directory."""
    direct_error = ""
    try:
        return parse_benchmark_results(output_dir, model=model)
    except Exception as exc:
        direct_error = str(exc)

    try:
        return _parse_nested_benchmark_results(output_dir, model=model)
    except Exception as exc:
        if direct_error:
            raise RuntimeError(f"{direct_error} Nested parse also failed: {exc}") from exc
        raise


def _build_suite_result(
    model: str,
    output_dir: str,
    return_code: int,
    parsed: Optional[BenchmarkResults],
    error: str = "",
) -> SuiteResult:
    if parsed is None:
        return SuiteResult(
            model=model,
            status="failed",
            total=0,
            passed=0,
            failed=0,
            pass_rate=None,
            output_dir=output_dir,
            return_code=return_code,
            error=error or f"benchmark exited with code {return_code}",
        )

    status = "complete" if return_code == 0 else "partial"
    return SuiteResult(
        model=model,
        status=status,
        total=parsed.total,
        passed=parsed.passed,
        failed=parsed.failed,
        pass_rate=parsed.pass_rate,
        output_dir=output_dir,
        return_code=return_code,
        error=error,
    )


def _refresh_running_result(result: SuiteResult) -> SuiteResult:
    try:
        parsed = _parse_suite_model_results(result.output_dir, model=result.model)
    except Exception:
        return result
    return SuiteResult(
        model=result.model,
        status=result.status,
        total=parsed.total,
        passed=parsed.passed,
        failed=parsed.failed,
        pass_rate=parsed.pass_rate,
        output_dir=result.output_dir,
        return_code=result.return_code,
        error=result.error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one task folder across multiple models."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-dir", help="Task directory to benchmark.")
    source.add_argument("--task", dest="task_files", action="append", help="Specific task JSON to benchmark. Repeat for multiple tasks.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to benchmark.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers per model run.")
    parser.add_argument("--observation-mode", default="text", choices=["text", "vision"])
    parser.add_argument("--category", default=None, help="Optional category filter.")
    parser.add_argument("--run-mode", default="standard", choices=["standard", "baseline", "full_info"])
    parser.add_argument("--output-dir", default=None, help="Optional parent output directory for the suite.")
    parser.add_argument("--no-calibration", action="store_true", default=False)
    return parser.parse_args()


def _prepare_task_dir(args: argparse.Namespace, suite_dir: Path) -> tuple[Path, Optional[Path]]:
    if args.tasks_dir:
        tasks_dir = _resolve_tasks_dir(args.tasks_dir)
        if not tasks_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {tasks_dir}")
        return tasks_dir, None

    task_paths = [_resolve_task_file(path) for path in (args.task_files or [])]
    missing = [str(path) for path in task_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Task file(s) not found:\n" + "\n".join(missing))

    temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_suite_tasks_", dir=suite_dir))
    for task_path in task_paths:
        link_path = temp_dir / task_path.name
        if not link_path.exists():
            link_path.symlink_to(task_path)
    return temp_dir, temp_dir


def main() -> int:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suite_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / f"{timestamp}-benchmark-suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    try:
        tasks_dir, prepared_task_dir = _prepare_task_dir(args, suite_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results: List[SuiteResult] = [
        SuiteResult(
            model=model,
            status="pending",
            total=0,
            passed=0,
            failed=0,
            pass_rate=None,
            output_dir=str(suite_dir / model.replace("/", "_")),
            return_code=-1,
        )
        for model in args.models
    ]
    _write_suite_summary(suite_dir, tasks_dir, args, results)
    try:
        for idx, model in enumerate(args.models):
            model_output_dir = suite_dir / model.replace("/", "_")
            results[idx].status = "running"
            _write_suite_summary(suite_dir, tasks_dir, args, results, current_model=model)
            cmd = [
                str(RUN_EMTOM),
                "benchmark",
                "--tasks-dir",
                str(tasks_dir),
                "--model",
                model,
                "--observation-mode",
                args.observation_mode,
                "--benchmark-run-mode",
                args.run_mode,
                "--max-workers",
                str(args.max_workers),
                "--output-dir",
                str(model_output_dir),
            ]
            if args.category:
                cmd.extend(["--category", args.category])
            if args.no_calibration:
                cmd.append("--no-calibration")

            print()
            print(_style("=" * 72, BOLD, MAGENTA))
            print(f"Running benchmark for {_style(model, BOLD, MAGENTA)}")
            print(_style("=" * 72, BOLD, MAGENTA))
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            last_refresh = time.monotonic()
            return_code = proc.poll()
            while return_code is None:
                time.sleep(5)
                now = time.monotonic()
                if now - last_refresh >= SUMMARY_REFRESH_SECONDS:
                    results[idx] = _refresh_running_result(results[idx])
                    _write_suite_summary(suite_dir, tasks_dir, args, results, current_model=model)
                    last_refresh = now
                return_code = proc.poll()
            results[idx] = _refresh_running_result(results[idx])

            parsed = None
            parse_error = ""
            try:
                parsed = _parse_suite_model_results(str(model_output_dir), model=model)
            except Exception as exc:
                parse_error = str(exc)

            results[idx] = _build_suite_result(
                model=model,
                output_dir=str(model_output_dir),
                return_code=return_code,
                parsed=parsed,
                error=parse_error,
            )
            _write_suite_summary(suite_dir, tasks_dir, args, results, current_model=model)
    finally:
        if prepared_task_dir is not None and prepared_task_dir.exists():
            for child in prepared_task_dir.iterdir():
                child.unlink(missing_ok=True)
            prepared_task_dir.rmdir()

    _write_suite_summary(suite_dir, tasks_dir, args, results)

    return 0 if all(result.return_code == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
