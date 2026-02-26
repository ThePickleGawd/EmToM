"""CLI entry point for parallel benchmark execution.

Called by run_emtom.sh when --max-workers is specified.
Runs one benchmark subprocess per task JSON with GPU round-robin,
then optionally writes calibration back to source task files.

Usage:
    python -m emtom.scripts.run_benchmark_parallel \
        --tasks-dir data/emtom/tasks \
        --model gpt-5.2 \
        --output-dir ./outputs/emtom/2026-02-25-benchmark \
        --max-workers 8

    # With team model map (competitive tasks):
    python -m emtom.scripts.run_benchmark_parallel \
        --tasks-dir data/emtom/tasks \
        --model gpt-5.2 \
        --output-dir ./outputs/emtom/2026-02-25-benchmark \
        --max-workers 8 \
        --team-model-map team_0=gpt-5.2,team_1=sonnet
"""

import argparse
import sys

from emtom.evolve.benchmark_wrapper import (
    run_benchmark_parallel,
    update_calibration_from_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark in parallel (one process per task)."
    )
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--no-video", action="store_true", default=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--team-model-map", default=None)
    parser.add_argument("--no-calibration", action="store_true", default=False)

    args = parser.parse_args()

    results = run_benchmark_parallel(
        tasks_dir=args.tasks_dir,
        model=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        no_video=args.no_video,
        category=args.category,
        team_model_map=args.team_model_map,
    )

    print(
        f"[parallel] Benchmark complete: {results.total} tasks "
        f"({results.passed} passed, {results.failed} failed, "
        f"pass_rate={results.pass_rate:.1f}%)"
    )

    if not args.no_calibration:
        team_model_map = None
        if args.team_model_map:
            team_model_map = {}
            for pair in args.team_model_map.split(","):
                k, v = pair.strip().split("=", 1)
                team_model_map[k.strip()] = v.strip()

        update_calibration_from_benchmark(
            results, args.tasks_dir, team_model_map=team_model_map
        )
        print(f"[parallel] Calibration updated in {args.tasks_dir}")


if __name__ == "__main__":
    main()
