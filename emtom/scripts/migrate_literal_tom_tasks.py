#!/usr/bin/env python3
"""
Deterministically migrate tasks to literal-ToM probe runtime semantics.

For each task:
1. Validate the canonical problem_pddl under the new runtime projection.
2. Backfill functional runtime metadata and literal-ToM probes.
3. Regenerate a physical-only golden trajectory.
4. Optionally verify the regenerated trajectory in simulator.
5. Write a manifest describing the migration outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _verify_task_file(task_file: Path, gpu_id: int | None = None) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "emtom.cli.verify_trajectory",
        str(task_file),
    ]
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    payload: Dict[str, Any]
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
        lines = stdout.splitlines()
        for start in range(len(lines)):
            candidate = "\n".join(lines[start:]).strip()
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if not payload:
            payload = {
                "success": False,
                "error": stdout or "verify_trajectory did not return JSON",
            }
    if stderr:
        payload["stderr"] = stderr[-4000:]
    payload["returncode"] = proc.returncode
    return payload


def migrate_task(task_path: Path, output_path: Path, verify: bool, gpu_id: int | None = None) -> Dict[str, Any]:
    from emtom.cli.validate_task import validate
    from emtom.pddl.planner import regenerate_golden_trajectory
    from emtom.pddl.runtime_projection import build_runtime_metadata

    task_data = _load_json(task_path)
    original_hash = hashlib.sha256(json.dumps(task_data, sort_keys=True).encode("utf-8")).hexdigest()

    validation = validate(task_data, scene_data=None)
    if not validation["success"]:
        return {
            "task": str(task_path),
            "status": "failed",
            "stage": "validate",
            "error": validation.get("error"),
        }

    task_data.update(build_runtime_metadata(task_data))
    regen = regenerate_golden_trajectory(task_data, scene_data=None, source="literal_tom_migration")
    _write_json(output_path, task_data)

    verify_result = None
    if verify:
        verify_result = _verify_task_file(output_path, gpu_id=gpu_id)
        verify_success = bool(verify_result.get("success", False))
        verify_valid = bool(verify_result.get("data", {}).get("valid", False))
        if not (verify_success and verify_valid):
            return {
                "task": str(task_path),
                "status": "failed",
                "stage": "verify_trajectory",
                "error": verify_result.get("error", "trajectory verification failed"),
                "verify_result": verify_result,
            }

    return {
        "task": str(task_path),
        "output": str(output_path),
        "status": "migrated",
        "runtime_semantics_version": task_data.get("runtime_semantics_version"),
        "functional_goal_pddl": task_data.get("functional_goal_pddl"),
        "literal_tom_probe_count": len(task_data.get("literal_tom_probes", [])),
        "epistemic_conjuncts_removed": task_data.get("epistemic_conjuncts_removed", 0),
        "trajectory_steps": regen.get("num_steps"),
        "original_hash": original_hash,
        "new_hash": hashlib.sha256(json.dumps(task_data, sort_keys=True).encode("utf-8")).hexdigest(),
        "verify_result": verify_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate tasks to literal-ToM runtime semantics")
    parser.add_argument(
        "--tasks-dir",
        default=str(PROJECT_ROOT / "data" / "emtom" / "tasks"),
        help="Input task directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "emtom" / "tasks_literal_tom_v1"),
        help="Output directory for migrated tasks",
    )
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "data" / "emtom" / "tasks_literal_tom_v1" / "migration_manifest.json"),
        help="Path to write migration manifest JSON",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip simulator trajectory verification after regenerating goldens",
    )
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    results: List[Dict[str, Any]] = []
    for task_path in sorted(tasks_dir.glob("*.json")):
        output_path = output_dir / task_path.name
        result = migrate_task(task_path, output_path, verify=not args.skip_verify)
        results.append(result)
        status = result.get("status", "unknown").upper()
        print(f"[{status}] {task_path.name}")
        if result.get("error"):
            print(f"  {result['error']}")

    manifest = {
        "runtime_semantics_version": "literal_tom_probe_v1",
        "tasks_dir": str(tasks_dir),
        "output_dir": str(output_dir),
        "results": results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
