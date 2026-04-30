"""Write benchmark results back into task JSONs as calibration metadata.

Usage:
    python -m emtom.scripts.update_calibration \
        --tasks-dir data/emtom/tasks \
        --benchmark-output-base ./outputs/emtom/2026-02-07-benchmark \
        --model gpt-5-mini

    # With team model map (competitive tasks):
    python -m emtom.scripts.update_calibration \
        --tasks-dir data/emtom/tasks \
        --benchmark-output-base ./outputs/emtom/2026-02-25-competitive \
        --model gpt-5.2 \
        --team-model-map team_0=gpt-5.2,team_1=sonnet
"""

import argparse

from emtom.evolve.benchmark_wrapper import (
    parse_benchmark_results,
    update_calibration_from_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Write benchmark results back into task JSONs as calibration metadata."
    )
    parser.add_argument("--tasks-dir", "--task-dir", dest="tasks_dir", required=True, help="Directory containing source task JSONs")
    parser.add_argument("--benchmark-output-base", required=True, help="Benchmark output base path (before -Nagents suffix)")
    parser.add_argument("--model", required=True, help="Model short name used in benchmark")
    parser.add_argument(
        "--team-model-map",
        default=None,
        help="Optional team->model mapping (e.g. 'team_0=gpt-5.2,team_1=sonnet')",
    )
    args = parser.parse_args()

    # Parse team model map if provided
    team_model_map = None
    if args.team_model_map:
        team_model_map = {}
        for pair in args.team_model_map.split(","):
            k, v = pair.strip().split("=", 1)
            team_model_map[k.strip()] = v.strip()

    results = parse_benchmark_results(args.benchmark_output_base, args.model)
    print(
        f"[calibration] Parsed {results.total} results "
        f"({results.passed} passed, {results.failed} failed)"
    )
    update_calibration_from_benchmark(results, args.tasks_dir, team_model_map=team_model_map)


if __name__ == "__main__":
    main()
