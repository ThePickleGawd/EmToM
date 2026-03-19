#!/usr/bin/env python3
"""
Canonicalize room restrictions so mechanic_bindings is the single source.

For each task JSON:
- read explicit `(is_restricted agent_X room_Y)` facts from problem_pddl :init
- merge them into room_restriction mechanic_bindings
- remove those explicit facts from problem_pddl

The task's semantics are preserved, but authored room restrictions live only in
mechanic_bindings going forward.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from emtom.pddl.problem_pddl import parse_problem_pddl, replace_init_in_problem_pddl


def _iter_task_files(tasks_dir: Path) -> Iterable[Path]:
    for path in sorted(tasks_dir.glob("*.json")):
        if path.is_file():
            yield path


def _restriction_map_from_problem(task_data: Dict[str, Any]) -> Dict[str, set[str]]:
    parsed = parse_problem_pddl(task_data.get("problem_pddl", ""))
    restrictions: Dict[str, set[str]] = defaultdict(set)
    for lit in parsed.init_literals:
        if lit.predicate != "is_restricted" or lit.negated or len(lit.args) != 2:
            continue
        agent, room = lit.args
        restrictions[agent].add(room)
    return restrictions


def _restriction_map_from_bindings(task_data: Dict[str, Any]) -> Dict[str, set[str]]:
    restrictions: Dict[str, set[str]] = defaultdict(set)
    for binding in task_data.get("mechanic_bindings", []):
        if not isinstance(binding, dict):
            continue
        if binding.get("mechanic_type") != "room_restriction":
            continue
        rooms = binding.get("restricted_rooms") or []
        agents = binding.get("for_agents") or []
        for agent in agents:
            if not isinstance(agent, str):
                continue
            for room in rooms:
                if isinstance(room, str):
                    restrictions[agent].add(room)
    return restrictions


def _canonical_room_restriction_bindings(
    task_data: Dict[str, Any],
    merged: Dict[str, set[str]],
) -> List[Dict[str, Any]]:
    preserved: List[Dict[str, Any]] = []
    for binding in task_data.get("mechanic_bindings", []):
        if not isinstance(binding, dict):
            continue
        if binding.get("mechanic_type") == "room_restriction":
            continue
        preserved.append(copy.deepcopy(binding))

    for agent in sorted(merged):
        rooms = sorted(merged[agent])
        if not rooms:
            continue
        preserved.append(
            {
                "mechanic_type": "room_restriction",
                "restricted_rooms": rooms,
                "for_agents": [agent],
            }
        )
    return preserved


def _strip_is_restricted_from_problem_pddl(task_data: Dict[str, Any]) -> str:
    parsed = parse_problem_pddl(task_data.get("problem_pddl", ""))
    kept_exprs = [
        lit.to_pddl()
        for lit in parsed.init_literals
        if lit.predicate != "is_restricted"
    ]
    kept_exprs.extend(expr.to_pddl() for expr in parsed.epistemic_init)
    new_init_pddl = "\n    ".join(kept_exprs)
    return replace_init_in_problem_pddl(task_data["problem_pddl"], new_init_pddl)


def migrate_task(task_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    pddl_restrictions = _restriction_map_from_problem(task_data)
    binding_restrictions = _restriction_map_from_bindings(task_data)

    if not pddl_restrictions:
        return task_data, False

    merged = defaultdict(set)
    for source in (binding_restrictions, pddl_restrictions):
        for agent, rooms in source.items():
            merged[agent].update(rooms)

    updated = copy.deepcopy(task_data)
    updated["mechanic_bindings"] = _canonical_room_restriction_bindings(updated, merged)
    updated["problem_pddl"] = _strip_is_restricted_from_problem_pddl(updated)
    return updated, True


def backup_tasks(tasks_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = tasks_dir / f"backup_pre_room_restriction_single_source_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for task_file in _iter_task_files(tasks_dir):
        shutil.copy2(task_file, backup_dir / task_file.name)
    return backup_dir


def run(tasks_dir: Path, dry_run: bool) -> Dict[str, Any]:
    backup_dir = None if dry_run else backup_tasks(tasks_dir)
    migrated = 0
    unchanged = 0

    for task_file in _iter_task_files(tasks_dir):
        with task_file.open() as f:
            task_data = json.load(f)
        updated, changed = migrate_task(task_data)
        if changed:
            migrated += 1
            if not dry_run:
                with task_file.open("w") as f:
                    json.dump(updated, f, indent=2)
                    f.write("\n")
        else:
            unchanged += 1

    return {
        "tasks_dir": str(tasks_dir),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "migrated": migrated,
        "unchanged": unchanged,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="data/emtom/tasks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(Path(args.tasks_dir), dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
