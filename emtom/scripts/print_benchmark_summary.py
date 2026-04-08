#!/usr/bin/env python3
"""Print a compact summary for a single benchmark run."""

from __future__ import annotations

import argparse

from emtom.evolve.benchmark_wrapper import (
    parse_benchmark_results,
    parse_parallel_benchmark_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact summary for one benchmark output.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--model", required=True, help="Benchmark model label.")
    parser.add_argument("--parallel", action="store_true", default=False, help="Parse as a parallel benchmark output.")
    args = parser.parse_args()

    if args.parallel:
        results = parse_parallel_benchmark_results(args.output_dir, args.model)
    else:
        results = parse_benchmark_results(args.output_dir, args.model)

    print("")
    print("==============================================")
    print("EMTOM BENCHMARK SUMMARY")
    print("==============================================")
    print(f"Model: {results.model}")
    print(f"Tasks: {results.total}")
    print(f"Passed: {results.passed}")
    print(f"Failed: {results.failed}")
    print(f"Pass rate: {results.pass_rate:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
