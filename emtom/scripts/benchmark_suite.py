#!/usr/bin/env python3
"""Run the same benchmark task folder across multiple models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from emtom.evolve.benchmark_wrapper import parse_benchmark_results


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_EMTOM = PROJECT_ROOT / "emtom" / "run_emtom.sh"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"


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


def render_suite_summary(results: List[SuiteResult], tasks_dir: Path) -> str:
    lines = [
        "",
        "=" * 72,
        "BENCHMARK SUITE SUMMARY",
        "=" * 72,
        f"Tasks dir: {tasks_dir}",
        "",
        f"{'Model':<18} {'Status':<10} {'Passed':>8} {'Total':>8} {'Pass rate':>10}",
        "-" * 72,
    ]
    for result in results:
        passed = str(result.passed) if result.total > 0 else "-"
        total = str(result.total) if result.total > 0 else "-"
        lines.append(
            f"{result.model:<18} {result.status:<10} {passed:>8} {total:>8} {_format_rate(result.pass_rate):>10}"
        )
    return "\n".join(lines)


def _write_suite_summary(
    suite_dir: Path,
    tasks_dir: Path,
    args: argparse.Namespace,
    results: List[SuiteResult],
) -> None:
    """Persist the current suite state for live monitoring."""
    summary_text = render_suite_summary(results, tasks_dir)
    print(summary_text)

    completed = sum(1 for result in results if result.status in {"complete", "failed"})
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
            _write_suite_summary(suite_dir, tasks_dir, args, results)
            cmd = [
                str(RUN_EMTOM),
                "benchmark",
                "--tasks-dir",
                str(tasks_dir),
                "--model",
                model,
                "--observation-mode",
                args.observation_mode,
                "--run-mode",
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
            print("=" * 72)
            print(f"Running benchmark for {model}")
            print("=" * 72)
            return_code = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode

            parsed = None
            parse_error = ""
            try:
                parsed = parse_benchmark_results(str(model_output_dir), model=model)
            except Exception as exc:
                parse_error = str(exc)

            if parsed is not None:
                status = "complete" if return_code == 0 else "failed"
                results[idx] = SuiteResult(
                    model=model,
                    status=status,
                    total=parsed.total,
                    passed=parsed.passed,
                    failed=parsed.failed,
                    pass_rate=parsed.pass_rate,
                    output_dir=str(model_output_dir),
                    return_code=return_code,
                    error=parse_error,
                )
            else:
                results[idx] = SuiteResult(
                    model=model,
                    status="failed",
                    total=0,
                    passed=0,
                    failed=0,
                    pass_rate=None,
                    output_dir=str(model_output_dir),
                    return_code=return_code,
                    error=parse_error or f"benchmark exited with code {return_code}",
                )
            _write_suite_summary(suite_dir, tasks_dir, args, results)
    finally:
        if prepared_task_dir is not None and prepared_task_dir.exists():
            for child in prepared_task_dir.iterdir():
                child.unlink(missing_ok=True)
            prepared_task_dir.rmdir()

    _write_suite_summary(suite_dir, tasks_dir, args, results)

    return 0 if all(result.return_code == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
