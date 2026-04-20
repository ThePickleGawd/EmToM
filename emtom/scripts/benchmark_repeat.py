#!/usr/bin/env python3
"""Run the same benchmark command multiple times and aggregate reliability metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from emtom.benchmark_metrics import (
    BenchmarkRepeatRun,
    build_repeat_summary,
    write_repeat_summary,
)
from emtom.evolve.benchmark_wrapper import (
    BenchmarkResults,
    parse_benchmark_results,
    parse_parallel_benchmark_results,
    update_calibration_from_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_EMTOM = PROJECT_ROOT / "emtom" / "run_emtom.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat EMTOM benchmark runs and aggregate pass@k / pass^k metrics."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-dir", help="Task directory to benchmark.")
    source.add_argument("--task", help="Single task JSON to benchmark.")
    parser.add_argument("--model", required=True, help="Model to benchmark.")
    parser.add_argument("--output-dir", required=True, help="Parent output directory for all repeats.")
    parser.add_argument("--num-times", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--max-sim-steps", type=int, default=200000)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--team-model-map", default=None)
    parser.add_argument("--observation-mode", default="text", choices=["text", "vision"])
    parser.add_argument(
        "--benchmark-run-mode",
        default="standard",
        choices=["standard", "baseline", "full_info"],
    )
    parser.add_argument("--agent-type", default="robot", choices=["robot", "human"])
    parser.add_argument("--selector-min-frames", type=int, default=1)
    parser.add_argument("--selector-max-frames", type=int, default=5)
    parser.add_argument("--selector-max-candidates", type=int, default=12)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--no-calibration", action="store_true", default=False)
    return parser.parse_args()


def _parse_team_model_map(raw_value: Optional[str]) -> Optional[Dict[str, str]]:
    if not raw_value:
        return None

    mapping: Dict[str, str] = {}
    for entry in raw_value.split(","):
        token = entry.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid --team-model-map entry '{token}'. Expected team_0=sonnet,team_1=gpt-5."
            )
        team_id, model = token.split("=", 1)
        team_id = team_id.strip()
        model = model.strip()
        if not team_id or not model:
            raise ValueError(
                f"Invalid --team-model-map entry '{token}'. Team ID and model must be non-empty."
            )
        mapping[team_id] = model
    return mapping or None


def _load_task_dicts(task_file: Path) -> List[dict]:
    with open(task_file, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return [entry for entry in payload["tasks"] if isinstance(entry, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _matches_category(task_dict: dict, category: Optional[str]) -> bool:
    if not category:
        return True
    return str(task_dict.get("category", "cooperative")).strip().lower() == category


def _task_id_from_dict(task_dict: dict, fallback: str) -> str:
    task_id = str(task_dict.get("task_id", "")).strip()
    if task_id and task_id != "REPLACE_WITH_UNIQUE_ID":
        return task_id
    return fallback


def _collect_expected_task_ids(args: argparse.Namespace) -> List[str]:
    task_files: List[Path]
    if args.task:
        task_files = [Path(args.task)]
    else:
        task_files = sorted(Path(args.tasks_dir).glob("*.json"))

    expected_task_ids: List[str] = []
    seen = set()
    for task_file in task_files:
        try:
            task_dicts = _load_task_dicts(task_file)
        except Exception:
            continue

        for idx, task_dict in enumerate(task_dicts):
            if not _matches_category(task_dict, args.category):
                continue
            fallback = task_file.stem if len(task_dicts) == 1 else f"{task_file.stem}:{idx}"
            task_id = _task_id_from_dict(task_dict, fallback=fallback)
            if task_id in seen:
                continue
            seen.add(task_id)
            expected_task_ids.append(task_id)

    return expected_task_ids


def _build_run_command(args: argparse.Namespace, run_output_dir: Path) -> list[str]:
    cmd = [
        str(RUN_EMTOM),
        "benchmark",
        "--model",
        args.model,
        "--output-dir",
        str(run_output_dir),
        "--agent-type",
        args.agent_type,
        "--max-sim-steps",
        str(args.max_sim_steps),
        "--benchmark-run-mode",
        args.benchmark_run_mode,
        "--observation-mode",
        args.observation_mode,
        "--selector-min-frames",
        str(args.selector_min_frames),
        "--selector-max-frames",
        str(args.selector_max_frames),
        "--selector-max-candidates",
        str(args.selector_max_candidates),
        "--no-calibration",
    ]

    if args.tasks_dir:
        cmd.extend(["--tasks-dir", args.tasks_dir])
    else:
        cmd.extend(["--task", args.task])

    if args.max_llm_calls is not None:
        cmd.extend(["--max-llm-calls", str(args.max_llm_calls)])
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.num_gpus is not None:
        cmd.extend(["--num-gpus", str(args.num_gpus)])
    if args.category:
        cmd.extend(["--category", args.category])
    if args.team_model_map:
        cmd.extend(["--team-model-map", args.team_model_map])
    if args.video:
        cmd.append("--video")

    return cmd


def _parse_run_results(args: argparse.Namespace, run_output_dir: Path) -> BenchmarkResults:
    if args.task:
        return parse_benchmark_results(str(run_output_dir), args.model)
    if args.max_workers is not None:
        try:
            return parse_parallel_benchmark_results(str(run_output_dir), args.model)
        except Exception:
            return parse_benchmark_results(str(run_output_dir), args.model)
    return parse_benchmark_results(str(run_output_dir), args.model)


def _print_summary(summary_path: Path, summary) -> None:
    print("")
    print("==============================================")
    print("EMTOM REPEATED BENCHMARK SUMMARY")
    print("==============================================")
    print(f"Model: {summary.model}")
    print(f"Runs requested: {summary.num_times}")
    print(f"Runs completed with results: {summary.completed_runs}")
    if summary.average_pass_rate is not None:
        print(f"Average pass rate: {summary.average_pass_rate:.1f}%")
    else:
        print("Average pass rate: --")
    print(f"Pass-rate std dev: {summary.std_pass_rate:.1f}%")
    if summary.pass_at_k is not None:
        print(f"Pass@{summary.k}: {summary.pass_at_k:.1f}%")
    else:
        print(f"Pass@{summary.k}: --")
    if summary.pass_power_k is not None:
        print(f"Pass^{summary.k}: {summary.pass_power_k:.1f}%")
    else:
        print(f"Pass^{summary.k}: --")
    print(f"Results saved to: {summary_path}")


def main() -> int:
    args = parse_args()
    if args.num_times < 1:
        print("ERROR: --num-times must be at least 1.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_task_ids = _collect_expected_task_ids(args)

    run_specs = []
    for run_index in range(1, args.num_times + 1):
        run_output_dir = output_dir / f"run_{run_index}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_run_command(args, run_output_dir)
        print(
            f"[repeat] starting run {run_index}/{args.num_times}: {' '.join(cmd)}"
        )
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        run_specs.append((run_index, run_output_dir, proc))

    runs = []
    parsed_runs: Dict[int, BenchmarkResults] = {}
    exit_codes = []

    for run_index, run_output_dir, proc in run_specs:
        return_code = proc.wait()
        exit_codes.append(return_code)

        parsed = None
        error = ""
        try:
            parsed = _parse_run_results(args, run_output_dir)
        except Exception as exc:
            error = str(exc)

        if parsed is None:
            status = "failed"
            run = BenchmarkRepeatRun(
                run_index=run_index,
                output_dir=str(run_output_dir),
                status=status,
                return_code=return_code,
                error=error or f"benchmark exited with code {return_code}",
            )
        else:
            status = "complete" if return_code == 0 else "partial"
            run = BenchmarkRepeatRun(
                run_index=run_index,
                output_dir=str(run_output_dir),
                status=status,
                return_code=return_code,
                total=parsed.total,
                passed=parsed.passed,
                failed=parsed.failed,
                pass_rate=parsed.pass_rate,
                error=error,
            )
            parsed_runs[run_index] = parsed

        runs.append(run)
        if run.pass_rate is not None:
            print(
                f"[repeat] finished run {run_index}/{args.num_times}: "
                f"status={run.status} pass_rate={run.pass_rate:.1f}% "
                f"({run.passed}/{run.total})"
            )
        else:
            print(
                f"[repeat] finished run {run_index}/{args.num_times}: "
                f"status={run.status} error={run.error or 'unknown error'}"
            )

    summary = build_repeat_summary(
        model=args.model,
        num_times=args.num_times,
        runs=sorted(runs, key=lambda run: run.run_index),
        parsed_runs=parsed_runs,
        expected_task_ids=expected_task_ids,
    )
    summary_path = write_repeat_summary(
        output_dir,
        summary,
        extra_fields={
            "tasks_dir": args.tasks_dir,
            "task": args.task,
            "observation_mode": args.observation_mode,
            "run_mode": args.benchmark_run_mode,
            "category": args.category,
            "team_model_map_requested": args.team_model_map,
            "max_workers": args.max_workers,
        },
    )

    if not args.no_calibration and args.tasks_dir:
        calibration_source = None
        for run_index in sorted(parsed_runs.keys(), reverse=True):
            run = next((item for item in runs if item.run_index == run_index), None)
            if run and run.return_code == 0:
                calibration_source = parsed_runs[run_index]
                break
        if calibration_source is None and parsed_runs:
            calibration_source = parsed_runs[max(parsed_runs.keys())]

        if calibration_source is not None:
            team_model_map = _parse_team_model_map(args.team_model_map)
            update_calibration_from_benchmark(
                calibration_source,
                args.tasks_dir,
                team_model_map=team_model_map,
            )

    _print_summary(summary_path, summary)
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
