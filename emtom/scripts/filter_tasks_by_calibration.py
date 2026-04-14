#!/usr/bin/env python3
"""Filter task JSONs by calibration outcomes for a given model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from emtom.evolve.benchmark_wrapper import cal_passed, find_calibration_entry


def _require_entry(task_data: dict, model: str, run_mode: str) -> dict | None:
    return find_calibration_entry(task_data.get("calibration", []), model=model, run_mode=run_mode)


def _entry_passed(entry: dict | None) -> bool:
    return bool(entry and cal_passed(entry))


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy tasks matching calibration filters into a new directory.")
    parser.add_argument("--src-dir", required=True, help="Source directory with task JSONs.")
    parser.add_argument("--out-dir", required=True, help="Destination directory for filtered task JSONs.")
    parser.add_argument("--model", required=True, help="Model label to filter on, e.g. gpt-5.4.")
    parser.add_argument(
        "--standard",
        choices=("pass", "fail", "any"),
        default="fail",
        help="Required outcome for standard-mode calibration.",
    )
    parser.add_argument(
        "--baseline",
        choices=("pass", "fail", "missing", "ignore"),
        default="pass",
        help="Required outcome for baseline-mode calibration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    parser.add_argument(
        "--add-standard-pass",
        type=int,
        default=0,
        help="After selecting the main filtered pool, add up to this many standard-pass tasks "
             "that satisfy the baseline criterion.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: list[str] = []
    extra_standard_pass: list[str] = []
    rejected = {
        "missing_standard": 0,
        "wrong_standard": 0,
        "wrong_baseline": 0,
    }
    eligible_standard_pass: list[Path] = []

    for task_file in sorted(src_dir.glob("*.json")):
        try:
            task_data = json.loads(task_file.read_text())
        except Exception:
            continue

        standard_entry = _require_entry(task_data, args.model, "standard")
        baseline_entry = _require_entry(task_data, args.model, "baseline")

        if standard_entry is None:
            rejected["missing_standard"] += 1
            continue

        standard_passed = _entry_passed(standard_entry)
        baseline_ok = True
        if args.baseline == "pass" and not _entry_passed(baseline_entry):
            baseline_ok = False
        elif args.baseline == "fail" and _entry_passed(baseline_entry):
            baseline_ok = False
        elif args.baseline == "missing" and baseline_entry is not None:
            baseline_ok = False

        if standard_passed:
            if baseline_ok:
                eligible_standard_pass.append(task_file)
            if args.standard != "pass":
                rejected["wrong_standard"] += 1
                continue
        else:
            if args.standard == "pass":
                rejected["wrong_standard"] += 1
                continue

        if not baseline_ok:
            rejected["wrong_baseline"] += 1
            continue

        shutil.copy2(task_file, out_dir / task_file.name)
        selected.append(task_file.name)

    if args.standard != "pass" and args.add_standard_pass > 0:
        for task_file in eligible_standard_pass[: args.add_standard_pass]:
            shutil.copy2(task_file, out_dir / task_file.name)
            extra_standard_pass.append(task_file.name)

    manifest = {
        "source_dir": str(src_dir),
        "model": args.model,
        "criteria": {
            "standard": args.standard,
            "baseline": args.baseline,
        },
        "selected": len(selected),
        "extra_standard_pass": len(extra_standard_pass),
        "rejected": rejected,
        "tasks": selected,
        "added_standard_pass_tasks": extra_standard_pass,
    }
    (out_dir / "_filter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Selected {len(selected)} tasks into {out_dir}")
    print(json.dumps(rejected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
