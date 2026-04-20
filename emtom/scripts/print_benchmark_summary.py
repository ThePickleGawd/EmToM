#!/usr/bin/env python3
"""Print a compact summary for a single benchmark run."""

from __future__ import annotations

import argparse

from emtom.benchmark_metrics import load_repeat_summary, repeat_summary_path
from emtom.evolve.benchmark_wrapper import (
    parse_benchmark_results,
    parse_parallel_benchmark_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact summary for one benchmark output.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--model", required=True, help="Benchmark model label.")
    parser.add_argument("--parallel", action="store_true", default=False, help="Parse as a parallel benchmark output.")
    parser.add_argument("--repeat", action="store_true", default=False, help="Parse as a repeated benchmark output.")
    args = parser.parse_args()

    if args.repeat or repeat_summary_path(args.output_dir).exists():
        summary = load_repeat_summary(args.output_dir)
        print("")
        print("==============================================")
        print("EMTOM REPEATED BENCHMARK SUMMARY")
        print("==============================================")
        print(f"Model: {summary.model}")
        print(f"Runs: {summary.num_times}")
        print(f"Completed runs: {summary.completed_runs}")
        print(f"Average pass rate: {summary.average_pass_rate:.1f}%" if summary.average_pass_rate is not None else "Average pass rate: --")
        print(f"Pass-rate std dev: {summary.std_pass_rate:.1f}%")
        print(f"Pass@{summary.k}: {summary.pass_at_k:.1f}%" if summary.pass_at_k is not None else f"Pass@{summary.k}: --")
        print(f"Pass^{summary.k}: {summary.pass_power_k:.1f}%" if summary.pass_power_k is not None else f"Pass^{summary.k}: --")
        return 0
    elif args.parallel:
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
