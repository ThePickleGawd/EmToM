#!/usr/bin/env python3
"""
One-pass salvage pipeline for legacy EMTOM tasks under literal-ToM runtime semantics.

Flow:
1. Backup all source tasks into a timestamped subfolder under the source task dir.
2. Migrate each task to functional-success + literal-ToM-probe semantics.
3. Verify the regenerated physical-only golden trajectory.
4. Judge the migrated task once.
5. Keep only tasks that pass both verification and judge.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from emtom.cli.judge_task import run as judge_run
from emtom.scripts.migrate_literal_tom_tasks import migrate_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _copy_backup(tasks_dir: Path, backup_dir: Path) -> int:
    backup_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for task_path in sorted(tasks_dir.glob("*.json")):
        shutil.copy2(task_path, backup_dir / task_path.name)
        count += 1
    return count


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _salvage_one(
    task_path: Path,
    candidates_dir: Path,
    salvaged_dir: Path,
    *,
    judge_threshold: float,
    verify: bool,
    gpu_id: int | None,
) -> Dict[str, Any]:
    output_path = candidates_dir / task_path.name
    migration = migrate_task(task_path, output_path, verify=verify, gpu_id=gpu_id)
    entry: Dict[str, Any] = {
        "task": task_path.name,
        "migration": migration,
        "judge": None,
        "salvaged": False,
    }

    if migration.get("status") == "migrated":
        judge_result = judge_run(str(output_path), threshold=judge_threshold)
        judge_passed = bool(judge_result.get("success")) and bool(judge_result.get("data", {}).get("passed"))
        entry["judge"] = judge_result
        entry["salvaged"] = judge_passed
        if judge_passed:
            shutil.copy2(output_path, salvaged_dir / task_path.name)

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Salvage legacy tasks under literal-ToM runtime semantics")
    parser.add_argument(
        "--tasks-dir",
        default=str(PROJECT_ROOT / "data" / "emtom" / "tasks"),
        help="Source task directory",
    )
    parser.add_argument(
        "--work-dir",
        default=str(PROJECT_ROOT / "outputs" / "literal_tom_salvage_full"),
        help="Working directory for migrated candidates, salvaged tasks, and manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of source tasks to process (0 = all)",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=0.7,
        help="Judge threshold",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip simulator verification of regenerated goldens",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip creating a timestamped backup copy of source tasks",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of tasks to process concurrently",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for round-robin verification assignment",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    work_dir = Path(args.work_dir)
    candidates_dir = work_dir / "migrated_candidates"
    salvaged_dir = work_dir / "salvaged"
    manifest_path = work_dir / "salvage_manifest.json"

    _clean_dir(candidates_dir)
    _clean_dir(salvaged_dir)

    backup_dir = None
    backup_count = 0
    if not args.skip_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = tasks_dir / f"backup_pre_literal_tom_{stamp}"
        backup_count = _copy_backup(tasks_dir, backup_dir)

    task_paths = sorted(tasks_dir.glob("*.json"))
    if args.limit > 0:
        task_paths = task_paths[:args.limit]

    results: List[Dict[str, Any]] = []
    max_workers = max(1, args.max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _salvage_one,
                task_path,
                candidates_dir,
                salvaged_dir,
                judge_threshold=args.judge_threshold,
                verify=not args.skip_verify,
                gpu_id=(idx % max(1, args.num_gpus)),
            ): task_path
            for idx, task_path in enumerate(task_paths)
        }
        for future in as_completed(future_map):
            task_path = future_map[future]
            try:
                entry = future.result()
            except Exception as exc:
                entry = {
                    "task": task_path.name,
                    "migration": {
                        "task": str(task_path),
                        "status": "failed",
                        "stage": "worker_exception",
                        "error": str(exc),
                    },
                    "judge": None,
                    "salvaged": False,
                }
            migration = entry["migration"]
            judge_result = entry["judge"] or {}
            if entry["salvaged"]:
                print(f"[SALVAGED] {task_path.name}")
            elif migration.get("status") == "migrated":
                print(f"[DROPPED] {task_path.name}")
                summary = judge_result.get("data", {}).get("summary")
                if summary:
                    print(f"  {summary}")
                elif judge_result.get("error"):
                    print(f"  {judge_result['error']}")
            else:
                print(f"[FAILED] {task_path.name}")
                if migration.get("error"):
                    print(f"  {migration['error']}")
            results.append(entry)

    manifest = {
        "tasks_dir": str(tasks_dir),
        "work_dir": str(work_dir),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "backup_count": backup_count,
        "processed_count": len(task_paths),
        "migrated_count": sum(1 for r in results if r["migration"].get("status") == "migrated"),
        "salvaged_count": sum(1 for r in results if r["salvaged"]),
        "results": results,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
