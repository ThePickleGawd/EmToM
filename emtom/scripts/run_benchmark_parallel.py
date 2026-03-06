"""CLI entry point for parallel benchmark execution.

Called by run_emtom.sh when --max-workers is specified.
Runs one benchmark subprocess per task JSON with GPU round-robin.
Calibration is written per-task as each subprocess completes.

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

from emtom.evolve.benchmark_wrapper import run_benchmark_parallel


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
    parser.add_argument("--observation-mode", default="text", choices=["text", "vision"])
    parser.add_argument("--selector-min-frames", type=int, default=1)
    parser.add_argument("--selector-max-frames", type=int, default=5)
    parser.add_argument("--selector-max-candidates", type=int, default=12)

    args = parser.parse_args()

    results = run_benchmark_parallel(
        tasks_dir=args.tasks_dir,
        model=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        no_video=args.no_video,
        category=args.category,
        team_model_map=args.team_model_map,
        observation_mode=args.observation_mode,
        selector_min_frames=args.selector_min_frames,
        selector_max_frames=args.selector_max_frames,
        selector_max_candidates=args.selector_max_candidates,
        write_calibration=not args.no_calibration,
    )

    print(
        f"[parallel] Done: {results.total} tasks "
        f"({results.passed} passed, {results.failed} failed, "
        f"{results.pass_rate:.1f}%)"
    )


if __name__ == "__main__":
    main()
