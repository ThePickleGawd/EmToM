#!/usr/bin/env python3
"""
Canonicalize mechanic init facts and keep only fully working tasks.

This migration:
- backs up the current task root
- converts supported init-only mechanic predicates in problem_pddl :init into
  canonical mechanic_bindings
- strips those authored init-only mechanic facts from problem_pddl
- regenerates a deterministic golden trajectory
- keeps only tasks that pass validate-task, verify-pddl, and static verification

The output replaces the root task directory with survivors only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from emtom.cli.validate_task import validate
from emtom.cli.verify_pddl import run as verify_pddl_run
from emtom.pddl.problem_pddl import parse_problem_pddl, replace_init_in_problem_pddl
from emtom.pddl.planner import generate_deterministic_trajectory
from emtom.task_gen.static_verify import verify_task
from emtom.task_gen.task_generator import MechanicBinding


SUPPORTED_INIT_ONLY_PREDS = {
    "can_communicate",
    "is_restricted",
    "controls",
    "is_inverse",
    "mirrors",
}


def _iter_task_files(tasks_dir: Path) -> Iterable[Path]:
    for path in sorted(tasks_dir.glob("*.json")):
        if path.is_file():
            yield path


def _binding_to_dict(binding: Any) -> Dict[str, Any]:
    if isinstance(binding, MechanicBinding):
        return binding.to_dict()
    return dict(binding)


def _canonical_bindings(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for binding in task_data.get("mechanic_bindings", []) or []:
        if isinstance(binding, MechanicBinding):
            result.append(binding.to_dict())
        elif isinstance(binding, dict):
            result.append(copy.deepcopy(binding))
    return result


def _merge_room_restrictions(bindings: List[Dict[str, Any]], parsed) -> None:
    merged: Dict[str, set[str]] = defaultdict(set)
    preserved: List[Dict[str, Any]] = []
    for binding in bindings:
        if binding.get("mechanic_type") != "room_restriction":
            preserved.append(binding)
            continue
        for agent in binding.get("for_agents") or []:
            if not isinstance(agent, str):
                continue
            for room in binding.get("restricted_rooms") or []:
                if isinstance(room, str):
                    merged[agent].add(room)

    for lit in parsed.init_literals:
        if lit.predicate != "is_restricted" or lit.negated or len(lit.args) != 2:
            continue
        agent, room = lit.args
        merged[agent].add(room)

    for agent in sorted(merged):
        preserved.append(
            {
                "mechanic_type": "room_restriction",
                "restricted_rooms": sorted(merged[agent]),
                "for_agents": [agent],
            }
        )

    bindings[:] = preserved


def _merge_restricted_communication(bindings: List[Dict[str, Any]], parsed) -> None:
    merged: Dict[str, set[str]] = defaultdict(set)
    preserved: List[Dict[str, Any]] = []

    for binding in bindings:
        if binding.get("mechanic_type") != "restricted_communication":
            preserved.append(binding)
            continue
        for sender, targets in (binding.get("allowed_targets") or {}).items():
            if not isinstance(sender, str):
                continue
            for target in targets or []:
                if isinstance(target, str):
                    merged[sender].add(target)

    for lit in parsed.init_literals:
        if lit.predicate != "can_communicate" or lit.negated or len(lit.args) != 2:
            continue
        sender, target = lit.args
        merged[sender].add(target)

    if merged:
        preserved.append(
            {
                "mechanic_type": "restricted_communication",
                "allowed_targets": {
                    sender: sorted(targets)
                    for sender, targets in sorted(merged.items())
                    if targets
                },
            }
        )

    bindings[:] = preserved


def _merge_inverse_state(bindings: List[Dict[str, Any]], parsed) -> None:
    existing = {
        binding.get("trigger_object")
        for binding in bindings
        if binding.get("mechanic_type") == "inverse_state"
    }
    for lit in parsed.init_literals:
        if lit.predicate != "is_inverse" or lit.negated or len(lit.args) != 1:
            continue
        target = lit.args[0]
        if target in existing:
            continue
        bindings.append(
            {
                "mechanic_type": "inverse_state",
                "trigger_object": target,
            }
        )
        existing.add(target)


def _merge_state_mirroring(bindings: List[Dict[str, Any]], parsed) -> None:
    existing = {
        (binding.get("trigger_object"), binding.get("target_object"))
        for binding in bindings
        if binding.get("mechanic_type") == "state_mirroring"
    }
    for lit in parsed.init_literals:
        if lit.predicate != "mirrors" or lit.negated or len(lit.args) != 2:
            continue
        trigger, target = lit.args
        key = (trigger, target)
        if key in existing:
            continue
        bindings.append(
            {
                "mechanic_type": "state_mirroring",
                "trigger_object": trigger,
                "target_object": target,
                "target_state": "is_open",
            }
        )
        existing.add(key)


def _merge_remote_control(bindings: List[Dict[str, Any]], parsed) -> None:
    existing = {
        (binding.get("trigger_object"), binding.get("target_object"))
        for binding in bindings
        if binding.get("mechanic_type") == "remote_control"
    }
    for lit in parsed.init_literals:
        if lit.predicate != "controls" or lit.negated or len(lit.args) != 2:
            continue
        trigger, target = lit.args
        key = (trigger, target)
        if key in existing:
            continue
        bindings.append(
            {
                "mechanic_type": "remote_control",
                "trigger_object": trigger,
                "target_object": target,
                "target_state": "is_unlocked",
            }
        )
        existing.add(key)


def _strip_supported_init_only_facts(problem_pddl: str) -> str:
    parsed = parse_problem_pddl(problem_pddl)
    kept_exprs = [
        lit.to_pddl()
        for lit in parsed.init_literals
        if lit.predicate not in SUPPORTED_INIT_ONLY_PREDS
    ]
    kept_exprs.extend(expr.to_pddl() for expr in parsed.epistemic_init)
    new_init_pddl = "\n    ".join(kept_exprs)
    return replace_init_in_problem_pddl(problem_pddl, new_init_pddl)


def migrate_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(task_data)
    parsed = parse_problem_pddl(updated["problem_pddl"])
    bindings = _canonical_bindings(updated)

    _merge_room_restrictions(bindings, parsed)
    _merge_restricted_communication(bindings, parsed)
    _merge_inverse_state(bindings, parsed)
    _merge_state_mirroring(bindings, parsed)
    _merge_remote_control(bindings, parsed)

    updated["mechanic_bindings"] = bindings
    updated["problem_pddl"] = _strip_supported_init_only_facts(updated["problem_pddl"])
    return updated


def _verify_pddl_dict(task_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(task_data, tmp)
        tmp_path = tmp.name
    try:
        result = verify_pddl_run(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return bool(result.get("success")), result


def _process_one(task_path: str) -> Dict[str, Any]:
    path = Path(task_path)
    try:
        with path.open() as f:
            original = json.load(f)

        migrated = migrate_task(original)
        regenerated = generate_deterministic_trajectory(migrated)
        migrated["golden_trajectory"] = regenerated["trajectory"]

        validate_result = validate(migrated, None)
        if not validate_result.get("success"):
            return {
                "task_file": path.name,
                "kept": False,
                "stage": "validate-task",
                "error": validate_result.get("error"),
            }

        pddl_ok, pddl_result = _verify_pddl_dict(migrated)
        if not pddl_ok:
            return {
                "task_file": path.name,
                "kept": False,
                "stage": "verify-pddl",
                "error": pddl_result.get("error"),
            }

        static_result = verify_task(path.name, migrated, None, strict_object_ids=False)
        if not static_result.valid:
            return {
                "task_file": path.name,
                "kept": False,
                "stage": "verify-static",
                "error": static_result.errors[0] if static_result.errors else "static verification failed",
            }

        return {
            "task_file": path.name,
            "kept": True,
            "task_data": migrated,
        }
    except Exception as exc:
        return {
            "task_file": path.name,
            "kept": False,
            "stage": "exception",
            "error": str(exc),
        }


def backup_tasks(tasks_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = tasks_dir / f"backup_pre_canonical_mechanics_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for task_file in _iter_task_files(tasks_dir):
        shutil.copy2(task_file, backup_dir / task_file.name)
    return backup_dir


def run(tasks_dir: Path, max_workers: int) -> Dict[str, Any]:
    backup_dir = backup_tasks(tasks_dir)
    source_files = list(_iter_task_files(backup_dir))
    staging_dir = tasks_dir / f".canonical_mechanics_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    kept = 0
    dropped = 0
    failures: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, str(task_path)): task_path.name
            for task_path in source_files
        }
        for future in as_completed(futures):
            result = future.result()
            if result.get("kept"):
                kept += 1
                out_path = staging_dir / result["task_file"]
                with out_path.open("w") as f:
                    json.dump(result["task_data"], f, indent=2)
                    f.write("\n")
            else:
                dropped += 1
                failures.append(result)

    for task_file in _iter_task_files(tasks_dir):
        task_file.unlink()
    for staged_file in sorted(staging_dir.glob("*.json")):
        shutil.move(str(staged_file), tasks_dir / staged_file.name)
    staging_dir.rmdir()

    manifest = {
        "tasks_dir": str(tasks_dir),
        "backup_dir": str(backup_dir),
        "source_count": len(source_files),
        "kept": kept,
        "dropped": dropped,
        "max_workers": max_workers,
        "failures": failures,
    }
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "outputs" / f"canonical_mechanics_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="data/emtom/tasks")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    result = run(Path(args.tasks_dir), max_workers=args.max_workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
