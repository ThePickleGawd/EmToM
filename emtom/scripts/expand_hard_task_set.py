#!/usr/bin/env python3
"""Stage-gated expansion of a hard task directory using benchmark screening."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emtom.evolve.benchmark_wrapper import cal_passed, cal_progress


MANIFEST_NAME = "_expansion_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def iter_task_files(dir_path: Path):
    for path in sorted(dir_path.glob("*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        if "task_id" not in data:
            continue
        yield path, data


def latest_calibration_entry(task_data: dict[str, Any], model: str, run_mode: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for entry in task_data.get("calibration") or []:
        if str(entry.get("run_mode", "standard") or "standard") != run_mode:
            continue
        agent_models = entry.get("agent_models") or {}
        if not agent_models or set(agent_models.values()) != {model}:
            continue
        tested_at = entry.get("tested_at") or ""
        best_tested_at = best.get("tested_at") if best else ""
        if best is None or tested_at > (best_tested_at or ""):
            best = entry
    return best


def entry_outcome(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "tested_at": entry.get("tested_at"),
        "passed": cal_passed(entry),
        "progress": cal_progress(entry),
    }


def task_index(dir_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path, data in iter_task_files(dir_path):
        index[data["task_id"]] = {
            "path": str(path),
            "file_name": path.name,
            "task_id": data["task_id"],
            "title": data.get("title"),
            "category": data.get("category"),
            "tom_level": data.get("tom_level"),
            "data": data,
        }
    return index


def hardness_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    g52 = candidate.get("gpt52_standard")
    sonnet = candidate.get("sonnet_standard")
    std_screen = candidate.get("standard_screen")

    g52_pass = g52.get("passed") if g52 else None
    g52_progress = g52.get("progress") if g52 else None
    sonnet_pass = sonnet.get("passed") if sonnet else None
    sonnet_progress = sonnet.get("progress") if sonnet else None
    screen_progress = std_screen.get("progress") if std_screen else None

    return (
        0 if std_screen and not std_screen.get("passed") else 1,
        screen_progress if screen_progress is not None else 2.0,
        0 if g52_pass is False else 1 if g52_pass is None else 2,
        g52_progress if g52_progress is not None else 2.0,
        0 if sonnet_pass is False else 1 if sonnet_pass is None else 2,
        sonnet_progress if sonnet_progress is not None else 2.0,
        -(candidate.get("tom_level") or 0),
        candidate["task_id"],
    )


def manifest_path(out_dir: Path) -> Path:
    return out_dir / MANIFEST_NAME


def load_manifest(out_dir: Path) -> dict[str, Any]:
    return read_json(manifest_path(out_dir))


def save_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    write_json(manifest_path(out_dir), manifest)


def current_out_task_ids(out_dir: Path) -> set[str]:
    return set(task_index(out_dir).keys())


def add_verified_candidates(src_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> None:
    src_tasks = task_index(src_dir)
    out_ids = current_out_task_ids(out_dir)
    existing_verified = {item["task_id"] for item in manifest.get("verified_added", [])}

    for task_id, meta in sorted(src_tasks.items()):
        if task_id in out_ids or task_id in existing_verified:
            continue

        task_data = meta["data"]
        standard = entry_outcome(latest_calibration_entry(task_data, manifest["model"], "standard"))
        baseline = entry_outcome(latest_calibration_entry(task_data, manifest["model"], "baseline"))
        if not standard or standard["passed"] or not baseline or not baseline["passed"]:
            continue

        shutil.copy2(meta["path"], out_dir / meta["file_name"])
        manifest["verified_added"].append(
            {
                "task_id": task_id,
                "file_name": meta["file_name"],
                "title": meta["title"],
                "category": meta["category"],
                "tom_level": meta["tom_level"],
                "standard": standard,
                "baseline": baseline,
                "copied_at": utc_now(),
            }
        )
        out_ids.add(task_id)


def build_candidate_queue(src_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    src_tasks = task_index(src_dir)
    out_ids = current_out_task_ids(out_dir)
    candidates: list[dict[str, Any]] = []

    for task_id, meta in sorted(src_tasks.items()):
        if task_id in out_ids:
            continue

        task_data = meta["data"]
        standard = entry_outcome(latest_calibration_entry(task_data, manifest["model"], "standard"))
        baseline = entry_outcome(latest_calibration_entry(task_data, manifest["model"], "baseline"))
        g52_standard = entry_outcome(latest_calibration_entry(task_data, "gpt-5.2", "standard"))
        sonnet_standard = entry_outcome(latest_calibration_entry(task_data, "sonnet", "standard"))

        if standard and not standard["passed"] and baseline and baseline["passed"]:
            continue

        candidate = {
            "task_id": task_id,
            "file_name": meta["file_name"],
            "source_path": meta["path"],
            "title": meta["title"],
            "category": meta["category"],
            "tom_level": meta["tom_level"],
            "gpt54_standard": standard,
            "gpt54_baseline": baseline,
            "gpt52_standard": g52_standard,
            "sonnet_standard": sonnet_standard,
            "standard_screen": None,
            "baseline_screen": None,
            "pending_standard_batch_id": None,
            "pending_baseline_batch_id": None,
            "promoted": False,
        }
        candidates.append(candidate)

    candidates.sort(key=hardness_sort_key)
    for idx, candidate in enumerate(candidates, start=1):
        candidate["queue_rank"] = idx
    return candidates


def ensure_manifest(out_dir: Path, src_dir: Path, model: str, target_count: int, overwrite: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(out_dir)
    if path.exists() and not overwrite:
        return load_manifest(out_dir)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_dir": str(src_dir),
        "out_dir": str(out_dir),
        "model": model,
        "target_count": target_count,
        "initial_count": len(current_out_task_ids(out_dir)),
        "verified_added": [],
        "candidate_queue": [],
        "batches": [],
        "imports": [],
        "promoted_task_ids": [],
    }
    save_manifest(out_dir, manifest)
    return manifest


def candidate_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["task_id"]: candidate for candidate in manifest.get("candidate_queue", [])}


def next_batch_id(manifest: dict[str, Any], mode: str) -> str:
    prefix = f"{mode}_"
    n = 1
    for batch in manifest.get("batches", []):
        batch_id = batch.get("id", "")
        if batch_id.startswith(prefix):
            try:
                n = max(n, int(batch_id.split("_", 1)[1]) + 1)
            except Exception:
                continue
    return f"{mode}_{n:03d}"


def create_batch(manifest: dict[str, Any], out_dir: Path, mode: str, batch_dir: Path, count: int) -> dict[str, Any]:
    batch_id = next_batch_id(manifest, mode)
    candidates = manifest["candidate_queue"]

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("promoted"):
            continue
        if mode == "standard":
            if candidate.get("standard_screen") is not None or candidate.get("pending_standard_batch_id"):
                continue
            selected.append(candidate)
        elif mode == "baseline":
            standard_screen = candidate.get("standard_screen")
            baseline_screen = candidate.get("baseline_screen")
            if not standard_screen or standard_screen.get("passed"):
                continue
            if baseline_screen is not None or candidate.get("pending_baseline_batch_id"):
                continue
            selected.append(candidate)
        if len(selected) >= count:
            break

    if not selected:
        raise SystemExit(f"No eligible {mode} candidates remain.")

    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        shutil.copy2(candidate["source_path"], batch_dir / candidate["file_name"])
        if mode == "standard":
            candidate["pending_standard_batch_id"] = batch_id
        else:
            candidate["pending_baseline_batch_id"] = batch_id

    batch = {
        "id": batch_id,
        "mode": mode,
        "created_at": utc_now(),
        "batch_tasks_dir": str(batch_dir),
        "benchmark_output_dir": None,
        "task_ids": [candidate["task_id"] for candidate in selected],
        "task_files": [candidate["file_name"] for candidate in selected],
        "result_count": 0,
        "missing_task_ids": [],
    }
    manifest["batches"].append(batch)
    save_manifest(out_dir, manifest)
    return batch


def parse_benchmark_output(benchmark_output_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(benchmark_output_dir.rglob("benchmark_summary.json")):
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        for item in summary.get("results", []):
            task_id = item.get("task_id")
            if not task_id:
                continue
            evaluation = item.get("evaluation") or {}
            progress = evaluation.get("percent_complete")
            if progress is None:
                progress = evaluation.get("all_goal_percent_complete")
            results[task_id] = {
                "passed": bool(item.get("success")),
                "progress": progress,
                "summary_path": str(summary_path),
                "run_mode": item.get("run_mode"),
            }
    return results


def import_output_dir(manifest: dict[str, Any], out_dir: Path, benchmark_output_dir: Path) -> dict[str, Any]:
    candidates = candidate_map(manifest)
    results = parse_benchmark_output(benchmark_output_dir)
    imported = {"standard": 0, "baseline": 0}
    touched_task_ids: list[str] = []

    for task_id, outcome in results.items():
        candidate = candidates.get(task_id)
        if candidate is None:
            continue

        run_mode = outcome.get("run_mode")
        if run_mode not in {"standard", "baseline"}:
            continue

        record = {
            "batch_id": None,
            "screened_at": utc_now(),
            "benchmark_output_dir": str(benchmark_output_dir),
            "summary_path": outcome["summary_path"],
            "passed": outcome["passed"],
            "progress": outcome["progress"],
            "source": "import",
        }
        if run_mode == "standard":
            if candidate.get("standard_screen") is not None:
                continue
            candidate["standard_screen"] = record
            candidate["pending_standard_batch_id"] = None
            imported["standard"] += 1
        else:
            if candidate.get("baseline_screen") is not None:
                continue
            candidate["baseline_screen"] = record
            candidate["pending_baseline_batch_id"] = None
            imported["baseline"] += 1
        touched_task_ids.append(task_id)

    manifest.setdefault("imports", []).append(
        {
            "imported_at": utc_now(),
            "benchmark_output_dir": str(benchmark_output_dir),
            "imported": imported,
            "task_ids": touched_task_ids,
        }
    )
    save_manifest(out_dir, manifest)
    return {
        "benchmark_output_dir": str(benchmark_output_dir),
        "imported": imported,
        "task_ids": touched_task_ids,
    }


def ingest_batch(manifest: dict[str, Any], out_dir: Path, batch_id: str, benchmark_output_dir: Path) -> dict[str, Any]:
    batch = next((item for item in manifest.get("batches", []) if item.get("id") == batch_id), None)
    if batch is None:
        raise SystemExit(f"Unknown batch id: {batch_id}")

    mode = batch["mode"]
    results = parse_benchmark_output(benchmark_output_dir)
    candidates = candidate_map(manifest)
    missing: list[str] = []

    for task_id in batch["task_ids"]:
        candidate = candidates[task_id]
        outcome = results.get(task_id)
        if outcome is None:
            missing.append(task_id)
            if mode == "standard":
                candidate["pending_standard_batch_id"] = None
            else:
                candidate["pending_baseline_batch_id"] = None
            continue

        record = {
            "batch_id": batch_id,
            "screened_at": utc_now(),
            "benchmark_output_dir": str(benchmark_output_dir),
            "summary_path": outcome["summary_path"],
            "passed": outcome["passed"],
            "progress": outcome["progress"],
        }
        if mode == "standard":
            candidate["standard_screen"] = record
            candidate["pending_standard_batch_id"] = None
        else:
            candidate["baseline_screen"] = record
            candidate["pending_baseline_batch_id"] = None

    batch["benchmark_output_dir"] = str(benchmark_output_dir)
    batch["ingested_at"] = utc_now()
    batch["result_count"] = len(results)
    batch["missing_task_ids"] = missing
    save_manifest(out_dir, manifest)
    return batch


def finalize_promotions(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    src_dir = Path(manifest["source_dir"])
    out_ids = current_out_task_ids(out_dir)
    needed = manifest["target_count"] - len(out_ids)
    promoted_now: list[str] = []

    if needed <= 0:
        for candidate in manifest.get("candidate_queue", []):
            candidate["pending_standard_batch_id"] = None
            candidate["pending_baseline_batch_id"] = None
        save_manifest(out_dir, manifest)
        return {"needed": 0, "promoted_now": promoted_now}

    eligible = []
    for candidate in manifest.get("candidate_queue", []):
        if candidate.get("promoted"):
            continue
        standard_screen = candidate.get("standard_screen")
        baseline_screen = candidate.get("baseline_screen")
        if not standard_screen or standard_screen.get("passed"):
            continue
        if not baseline_screen or not baseline_screen.get("passed"):
            continue
        eligible.append(candidate)

    eligible.sort(key=hardness_sort_key)
    for candidate in eligible[:needed]:
        src_path = src_dir / candidate["file_name"]
        if candidate["task_id"] in out_ids:
            candidate["promoted"] = True
            continue
        shutil.copy2(src_path, out_dir / candidate["file_name"])
        candidate["promoted"] = True
        candidate["promoted_at"] = utc_now()
        manifest["promoted_task_ids"].append(candidate["task_id"])
        promoted_now.append(candidate["task_id"])
        out_ids.add(candidate["task_id"])

    if len(out_ids) >= manifest["target_count"]:
        for candidate in manifest.get("candidate_queue", []):
            candidate["pending_standard_batch_id"] = None
            candidate["pending_baseline_batch_id"] = None

    save_manifest(out_dir, manifest)
    return {"needed": max(0, manifest["target_count"] - len(out_ids)), "promoted_now": promoted_now}


def status_summary(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    candidates = manifest.get("candidate_queue", [])
    current_count = len(current_out_task_ids(out_dir))
    standard_failed = sum(1 for c in candidates if c.get("standard_screen") and not c["standard_screen"]["passed"])
    baseline_pass = sum(1 for c in candidates if c.get("baseline_screen") and c["baseline_screen"]["passed"])
    ready_to_promote = sum(
        1
        for c in candidates
        if c.get("standard_screen")
        and not c["standard_screen"]["passed"]
        and c.get("baseline_screen")
        and c["baseline_screen"]["passed"]
        and not c.get("promoted")
    )
    return {
        "current_count": current_count,
        "target_count": manifest["target_count"],
        "needed": max(0, manifest["target_count"] - current_count),
        "verified_added": len(manifest.get("verified_added", [])),
        "queue_size": len(candidates),
        "standard_screened_fail": standard_failed,
        "baseline_screened_pass": baseline_pass,
        "ready_to_promote": ready_to_promote,
        "promoted": len(manifest.get("promoted_task_ids", [])),
        "pending_standard": sum(1 for c in candidates if c.get("pending_standard_batch_id")),
        "pending_baseline": sum(1 for c in candidates if c.get("pending_baseline_batch_id")),
    }


def cmd_init(args: argparse.Namespace) -> int:
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    manifest = ensure_manifest(out_dir, src_dir, args.model, args.target_count, args.overwrite)
    add_verified_candidates(src_dir, out_dir, manifest)
    manifest["candidate_queue"] = build_candidate_queue(src_dir, out_dir, manifest)
    save_manifest(out_dir, manifest)
    print(json.dumps(status_summary(manifest, out_dir), indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    manifest = load_manifest(out_dir)
    print(json.dumps(status_summary(manifest, out_dir), indent=2))
    return 0


def cmd_make_batch(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    batch_dir = Path(args.batch_dir)
    manifest = load_manifest(out_dir)
    batch = create_batch(manifest, out_dir, args.mode, batch_dir, args.count)
    print(json.dumps(batch, indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    benchmark_output_dir = Path(args.benchmark_output_dir)
    manifest = load_manifest(out_dir)
    batch = ingest_batch(manifest, out_dir, args.batch_id, benchmark_output_dir)
    print(json.dumps(batch, indent=2))
    print(json.dumps(status_summary(manifest, out_dir), indent=2))
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    manifest = load_manifest(out_dir)
    result = finalize_promotions(manifest, out_dir)
    print(json.dumps(result, indent=2))
    print(json.dumps(status_summary(manifest, out_dir), indent=2))
    return 0


def cmd_import_output(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    benchmark_output_dir = Path(args.benchmark_output_dir)
    manifest = load_manifest(out_dir)
    result = import_output_dir(manifest, out_dir, benchmark_output_dir)
    print(json.dumps(result, indent=2))
    print(json.dumps(status_summary(manifest, out_dir), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand a hard benchmark task set by staged screening.")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Initialize manifest, copy verified tasks, and build the queue.")
    init_p.add_argument("--src-dir", required=True)
    init_p.add_argument("--out-dir", required=True)
    init_p.add_argument("--model", default="gpt-5.4")
    init_p.add_argument("--target-count", type=int, default=250)
    init_p.add_argument("--overwrite", action="store_true")
    init_p.set_defaults(func=cmd_init)

    status_p = sub.add_parser("status", help="Show current expansion state.")
    status_p.add_argument("--out-dir", required=True)
    status_p.set_defaults(func=cmd_status)

    make_batch_p = sub.add_parser("make-batch", help="Create a screening batch directory.")
    make_batch_p.add_argument("--out-dir", required=True)
    make_batch_p.add_argument("--mode", choices=("standard", "baseline"), required=True)
    make_batch_p.add_argument("--batch-dir", required=True)
    make_batch_p.add_argument("--count", type=int, required=True)
    make_batch_p.set_defaults(func=cmd_make_batch)

    ingest_p = sub.add_parser("ingest", help="Ingest results from a benchmark run.")
    ingest_p.add_argument("--out-dir", required=True)
    ingest_p.add_argument("--batch-id", required=True)
    ingest_p.add_argument("--benchmark-output-dir", required=True)
    ingest_p.set_defaults(func=cmd_ingest)

    finalize_p = sub.add_parser("finalize", help="Promote baseline-passing standard failures into the output set.")
    finalize_p.add_argument("--out-dir", required=True)
    finalize_p.set_defaults(func=cmd_finalize)

    import_p = sub.add_parser("import-output", help="Import screening results from an existing benchmark output dir.")
    import_p.add_argument("--out-dir", required=True)
    import_p.add_argument("--benchmark-output-dir", required=True)
    import_p.set_defaults(func=cmd_import_output)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
