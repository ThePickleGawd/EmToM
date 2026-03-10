#!/usr/bin/env python3
"""
Migrate non_verified_pddl tasks by injecting missing :init facts.

Three phases:
  1. cache-scenes  — Load each unique scene via habitat-sim (GPU, parallel)
  2. patch         — Inject is_in_room / agent_in_room into problem_pddl (CPU, fast)
  3. verify        — Run FD solver on patched tasks (CPU, parallel)

Usage:
    # Full pipeline (needs GPU for phase 1):
    python -m emtom.scripts.migrate_tasks

    # If scene cache already exists, skip to patch+verify:
    python -m emtom.scripts.migrate_tasks --skip-cache

    # Dry run (no writes):
    python -m emtom.scripts.migrate_tasks --dry-run

    # Patch only, skip verify (fast):
    python -m emtom.scripts.migrate_tasks --skip-cache --skip-verify
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks" / "non_verified_pddl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks"
FAILED_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks" / "migration_failed"
SCENE_CACHE_DIR = PROJECT_ROOT / "data" / "emtom" / "scene_cache"


# ---------------------------------------------------------------------------
# Phase 1: Cache scenes
# ---------------------------------------------------------------------------

def _get_gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], text=True, stderr=subprocess.DEVNULL,
        )
        return len(out.strip().splitlines())
    except Exception:
        return 1


def _load_one_scene(scene_id: str, gpu_id: int, cache_dir: Path) -> Dict[str, Any]:
    """Load a single scene via subprocess and cache the result."""
    cache_file = cache_dir / f"{scene_id}.json"
    if cache_file.exists():
        return {"scene_id": scene_id, "status": "cached", "error": None}

    with tempfile.TemporaryDirectory(prefix=f"scene_{scene_id}_") as tmpdir:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, "-m", "emtom.cli.new_scene",
            "2",  # num_agents (minimum; furniture-room map is agent-count independent)
            "--working-dir", tmpdir,
            "--scene-id", scene_id,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env=env, cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            return {"scene_id": scene_id, "status": "timeout", "error": "scene load timed out (300s)"}

        scene_file = Path(tmpdir) / "current_scene.json"
        if not scene_file.exists():
            # Try to parse error from stdout
            stderr_tail = (proc.stderr or "")[-500:]
            stdout_tail = (proc.stdout or "")[-500:]
            return {
                "scene_id": scene_id,
                "status": "error",
                "error": f"No current_scene.json produced. stderr: {stderr_tail} stdout: {stdout_tail}",
            }

        # Copy to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scene_file, cache_file)

    return {"scene_id": scene_id, "status": "ok", "error": None}


def cache_scenes(scene_ids: List[str], cache_dir: Path, num_gpus: int) -> List[Dict[str, Any]]:
    """Load all unique scenes in parallel across GPUs."""
    # Filter already-cached
    to_load = [s for s in scene_ids if not (cache_dir / f"{s}.json").exists()]
    already_cached = len(scene_ids) - len(to_load)
    if already_cached:
        print(f"  {already_cached} scenes already cached, {len(to_load)} to load")

    if not to_load:
        return [{"scene_id": s, "status": "cached", "error": None} for s in scene_ids]

    results: List[Dict[str, Any]] = []
    # Scene loading requires GPU+GL context, run sequentially per GPU but
    # parallelize across GPUs.  Each subprocess gets its own GL context.
    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        futures = {}
        for idx, sid in enumerate(to_load):
            gpu = idx % num_gpus
            fut = pool.submit(_load_one_scene, sid, gpu, cache_dir)
            futures[fut] = sid

        for fut in as_completed(futures):
            res = fut.result()
            tag = "OK" if res["status"] == "ok" else res["status"].upper()
            print(f"    [{tag}] {res['scene_id']}" + (f" — {res['error']}" if res.get("error") else ""))
            results.append(res)

    # Include pre-cached
    loaded_ids = {r["scene_id"] for r in results}
    for sid in scene_ids:
        if sid not in loaded_ids:
            results.append({"scene_id": sid, "status": "cached", "error": None})

    return results


# ---------------------------------------------------------------------------
# Phase 2: Patch PDDL
# ---------------------------------------------------------------------------

def _load_scene_cache(scene_id: str, cache_dir: Path) -> Optional[Dict[str, Any]]:
    cache_file = cache_dir / f"{scene_id}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        return json.load(f)


def _build_room_maps(scene_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build furniture->room and object->room from scene cache."""
    furn_to_room: Dict[str, str] = {}
    for room, furns in scene_data.get("furniture_in_rooms", {}).items():
        for furn in furns:
            furn_to_room[furn] = room

    obj_to_furn: Dict[str, str] = {}
    for furn, objs in scene_data.get("objects_on_furniture", {}).items():
        for obj in objs:
            obj_to_furn[obj] = furn

    obj_to_room: Dict[str, str] = {}
    for obj, furn in obj_to_furn.items():
        if furn in furn_to_room:
            obj_to_room[obj] = furn_to_room[furn]

    return furn_to_room, obj_to_room


def _parse_init_predicates(pddl: str) -> List[Tuple[str, List[str]]]:
    """Quick parse of :init predicates as (predicate, [args]) tuples."""
    import re
    lower = pddl.lower()
    init_start = lower.find("(:init")
    if init_start < 0:
        return []

    # Find matching paren
    depth = 0
    init_end = init_start
    for i in range(init_start, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                init_end = i
                break

    init_body = pddl[init_start + 6:init_end]

    preds = []
    for m in re.finditer(r"\((\w+)\s+([^()]+)\)", init_body):
        pred = m.group(1)
        args = m.group(2).split()
        preds.append((pred, args))
    return preds


def _parse_objects_section(pddl: str) -> Dict[str, str]:
    """Parse :objects section into {name: type}."""
    import re
    lower = pddl.lower()
    obj_start = lower.find("(:objects")
    if obj_start < 0:
        return {}

    depth = 0
    obj_end = obj_start
    for i in range(obj_start, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                obj_end = i
                break

    body = pddl[obj_start + 9:obj_end].strip()
    tokens = body.split()

    objects: Dict[str, str] = {}
    pending: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            if i + 1 < len(tokens):
                typ = tokens[i + 1]
                for name in pending:
                    objects[name] = typ
                pending = []
                i += 2
                continue
        pending.append(tok)
        i += 1
    for name in pending:
        objects[name] = "object"
    return objects


def _get_restricted_rooms(pddl: str, agent_id: str) -> Set[str]:
    """Get rooms that are restricted for a specific agent."""
    restricted = set()
    for pred, args in _parse_init_predicates(pddl):
        if pred == "is_restricted" and len(args) == 2 and args[0] == agent_id:
            restricted.add(args[1])
    return restricted


def _find_init_close_paren(pddl: str) -> int:
    """Find the index of the closing ')' of the (:init ...) section."""
    lower = pddl.lower()
    init_start = lower.find("(:init")
    if init_start < 0:
        raise ValueError("No (:init section found")

    depth = 0
    for i in range(init_start, len(pddl)):
        if pddl[i] == "(":
            depth += 1
        elif pddl[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced parens in (:init ...)")


def patch_task(
    task_data: Dict[str, Any],
    scene_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Patch a task's problem_pddl with missing is_in_room and agent_in_room facts.

    Returns (patched_task_data, report_entry).
    """
    pddl = task_data["problem_pddl"]
    objects = _parse_objects_section(pddl)
    init_preds = _parse_init_predicates(pddl)
    furn_to_room, obj_to_room = _build_room_maps(scene_data)

    # What's already grounded
    existing_is_in_room: Set[str] = set()
    existing_agent_in_room: Set[str] = set()
    parent_of: Dict[str, str] = {}  # object -> furniture (from is_on_top/is_inside)

    for pred, args in init_preds:
        if pred == "is_in_room" and len(args) == 2:
            existing_is_in_room.add(args[0])
        if pred == "agent_in_room" and len(args) == 2:
            existing_agent_in_room.add(args[0])
        if pred in ("is_on_top", "is_inside") and len(args) == 2:
            parent_of[args[0]] = args[1]

    # Determine what needs patching
    rooms = [name for name, typ in objects.items() if typ == "room"]
    agents = sorted(name for name, typ in objects.items() if typ == "agent")
    furniture_ids = [name for name, typ in objects.items() if typ == "furniture"]
    object_ids = [name for name, typ in objects.items() if typ == "object"]

    new_facts: List[str] = []
    warnings: List[str] = []

    # Patch furniture is_in_room
    for fid in furniture_ids:
        if fid in existing_is_in_room:
            continue
        room = furn_to_room.get(fid)
        if room and room in rooms:
            new_facts.append(f"(is_in_room {fid} {room})")
        else:
            warnings.append(f"furniture {fid} not found in scene cache")

    # Patch object is_in_room
    for oid in object_ids:
        if oid in existing_is_in_room:
            continue
        # Try 1: derive from parent furniture in :init
        parent = parent_of.get(oid)
        room = None
        if parent:
            # Parent furniture might already be resolved or in scene cache
            room = furn_to_room.get(parent)
        # Try 2: scene cache direct lookup
        if not room:
            room = obj_to_room.get(oid)
        if room and room in rooms:
            new_facts.append(f"(is_in_room {oid} {room})")
            # Also ensure parent furniture has is_in_room
            if parent and parent not in existing_is_in_room:
                parent_room = furn_to_room.get(parent)
                if parent_room and parent_room in rooms:
                    fact = f"(is_in_room {parent} {parent_room})"
                    if fact not in new_facts:
                        new_facts.append(fact)
        else:
            warnings.append(f"object {oid} room not resolvable")

    # Patch agent_in_room
    agent_room_strategies: Dict[str, str] = {}
    for agent_id in agents:
        if agent_id in existing_agent_in_room:
            continue
        restricted = _get_restricted_rooms(pddl, agent_id)
        available = [r for r in rooms if r not in restricted]
        if not available:
            warnings.append(f"{agent_id} restricted from all rooms")
            # Fallback: first room
            available = rooms

        # Pick the first available room
        room = available[0]
        new_facts.append(f"(agent_in_room {agent_id} {room})")
        strategy = "first_non_restricted" if restricted else "first_room"
        agent_room_strategies[agent_id] = f"{strategy}:{room}"

    # Insert into PDDL string
    if new_facts:
        init_close = _find_init_close_paren(pddl)
        insertion = "\n    " + "\n    ".join(new_facts) + "\n  "
        patched_pddl = pddl[:init_close] + insertion + pddl[init_close:]
        task_data = dict(task_data)
        task_data["problem_pddl"] = patched_pddl

    report = {
        "added_facts": len(new_facts),
        "added_is_in_room": sum(1 for f in new_facts if "is_in_room" in f),
        "added_agent_in_room": sum(1 for f in new_facts if "agent_in_room" in f),
        "agent_room_strategies": agent_room_strategies,
        "warnings": warnings,
    }
    return task_data, report


def validate_patched_pddl(task_data: Dict[str, Any]) -> List[str]:
    """Run structural validation on patched problem_pddl. Returns error list."""
    from emtom.pddl.problem_pddl import parse_problem_pddl, validate_problem_pddl_self_contained

    parsed = parse_problem_pddl(task_data["problem_pddl"])
    return validate_problem_pddl_self_contained(
        parsed, num_agents=task_data.get("num_agents", 2),
    )


# ---------------------------------------------------------------------------
# Phase 3: Verify (FD solver)
# ---------------------------------------------------------------------------

def _verify_one_task(task_path: str) -> Dict[str, Any]:
    """Run verify_pddl on a single task file. Returns result dict."""
    try:
        from emtom.cli.verify_pddl import run
        result = run(task_path)
        return {
            "file": task_path,
            "success": result.get("success", False),
            "tom_level": result.get("data", {}).get("tom_level"),
            "tom_reasoning": result.get("data", {}).get("tom_reasoning"),
            "error": result.get("error"),
        }
    except Exception as e:
        return {"file": task_path, "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Migrate non_verified_pddl tasks")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--failed-dir", type=Path, default=FAILED_DIR)
    parser.add_argument("--scene-cache-dir", type=Path, default=SCENE_CACHE_DIR)
    parser.add_argument("--skip-cache", action="store_true", help="Skip scene caching (assumes cache exists)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip FD solver verification")
    parser.add_argument("--dry-run", action="store_true", help="No writes, just report")
    parser.add_argument("--verify-workers", type=int, default=8, help="Parallel verify workers")
    args = parser.parse_args()

    start_time = time.time()

    # Scan input tasks
    task_files = sorted(args.input_dir.glob("*.json"))
    if not task_files:
        print(f"No tasks found in {args.input_dir}")
        return

    print(f"Found {len(task_files)} tasks in {args.input_dir}")

    # Load all tasks and group by scene_id
    tasks: List[Tuple[Path, Dict[str, Any]]] = []
    scene_to_tasks: Dict[str, List[int]] = collections.defaultdict(list)
    for tf in task_files:
        with open(tf) as f:
            td = json.load(f)
        idx = len(tasks)
        tasks.append((tf, td))
        scene_to_tasks[td.get("scene_id", "UNKNOWN")].append(idx)

    unique_scenes = sorted(scene_to_tasks.keys())
    print(f"Unique scenes: {len(unique_scenes)}")

    # ---- Phase 1: Cache scenes ----
    if not args.skip_cache:
        num_gpus = _get_gpu_count()
        print(f"\n=== Phase 1: Cache scenes ({len(unique_scenes)} scenes, {num_gpus} GPUs) ===")
        cache_results = cache_scenes(unique_scenes, args.scene_cache_dir, num_gpus)
        failed_scenes = {r["scene_id"] for r in cache_results if r["status"] not in ("ok", "cached")}
        if failed_scenes:
            print(f"  WARNING: {len(failed_scenes)} scenes failed to load: {failed_scenes}")
    else:
        print("\n=== Phase 1: Skipped (--skip-cache) ===")
        failed_scenes = set()
        # Check which scenes are actually cached
        for sid in unique_scenes:
            if not (args.scene_cache_dir / f"{sid}.json").exists():
                failed_scenes.add(sid)
        if failed_scenes:
            print(f"  WARNING: {len(failed_scenes)} scenes missing from cache: {failed_scenes}")

    # ---- Phase 2: Patch PDDL ----
    print(f"\n=== Phase 2: Patch PDDL ===")
    patched_dir = args.output_dir / "_migration_staging" if not args.dry_run else None
    if patched_dir:
        patched_dir.mkdir(parents=True, exist_ok=True)

    patch_results: List[Dict[str, Any]] = []
    patched_files: List[Path] = []

    for tf, td in tasks:
        fname = tf.name
        scene_id = td.get("scene_id", "UNKNOWN")
        entry: Dict[str, Any] = {"file": fname, "scene_id": scene_id}

        if scene_id in failed_scenes:
            entry["status"] = "scene_cache_miss"
            entry["error"] = f"Scene {scene_id} not in cache"
            patch_results.append(entry)
            continue

        scene_data = _load_scene_cache(scene_id, args.scene_cache_dir)
        if scene_data is None:
            entry["status"] = "scene_cache_miss"
            entry["error"] = f"Scene cache file missing for {scene_id}"
            patch_results.append(entry)
            continue

        try:
            patched_td, patch_report = patch_task(td, scene_data)
            entry.update(patch_report)
        except Exception as e:
            entry["status"] = "patch_error"
            entry["error"] = str(e)
            patch_results.append(entry)
            continue

        # Structural validation
        try:
            errors = validate_patched_pddl(patched_td)
        except Exception as e:
            entry["status"] = "validate_error"
            entry["error"] = str(e)
            patch_results.append(entry)
            continue

        if errors:
            entry["status"] = "validate_fail"
            entry["error"] = "; ".join(errors)
            patch_results.append(entry)
            continue

        entry["status"] = "patched"

        if not args.dry_run and patched_dir:
            out_file = patched_dir / fname
            with open(out_file, "w") as f:
                json.dump(patched_td, f, indent=2)
            patched_files.append(out_file)

        patch_results.append(entry)

    patched_ok = sum(1 for r in patch_results if r["status"] == "patched")
    patched_fail = sum(1 for r in patch_results if r["status"] != "patched")
    print(f"  Patched: {patched_ok}, Failed: {patched_fail}")

    for r in patch_results:
        if r["status"] != "patched":
            print(f"    [{r['status'].upper()}] {r['file']}: {r.get('error', '')[:120]}")

    # ---- Phase 3: Verify ----
    if args.skip_verify or args.dry_run:
        print(f"\n=== Phase 3: Skipped {'(--dry-run)' if args.dry_run else '(--skip-verify)'} ===")
        verify_results = []
    else:
        print(f"\n=== Phase 3: Verify PDDL ({len(patched_files)} tasks, {args.verify_workers} workers) ===")
        verify_results = []

        with ProcessPoolExecutor(max_workers=args.verify_workers) as pool:
            futures = {
                pool.submit(_verify_one_task, str(pf)): pf
                for pf in patched_files
            }
            done_count = 0
            for fut in as_completed(futures):
                res = fut.result()
                verify_results.append(res)
                done_count += 1
                tag = "PASS" if res["success"] else "FAIL"
                fname = Path(res["file"]).name
                tom = f" tom={res['tom_level']}" if res.get("tom_level") is not None else ""
                err = f" — {res['error'][:80]}" if not res["success"] and res.get("error") else ""
                if done_count % 10 == 0 or not res["success"]:
                    print(f"    [{tag}] {fname}{tom}{err}  ({done_count}/{len(patched_files)})")

        verified_ok = sum(1 for r in verify_results if r["success"])
        verified_fail = sum(1 for r in verify_results if not r["success"])
        print(f"  Verified: {verified_ok}, Failed: {verified_fail}")

    # ---- Phase 4: Move results ----
    if args.dry_run:
        print(f"\n=== Phase 4: Skipped (--dry-run) ===")
    else:
        print(f"\n=== Phase 4: Move results ===")
        args.failed_dir.mkdir(parents=True, exist_ok=True)

        # Build set of verified files (or all patched if verify was skipped)
        if verify_results:
            verified_set = {Path(r["file"]).name for r in verify_results if r["success"]}
            failed_verify_set = {Path(r["file"]).name for r in verify_results if not r["success"]}
        else:
            verified_set = {pf.name for pf in patched_files}
            failed_verify_set = set()

        moved_ok = 0
        moved_fail = 0

        for pf in patched_files:
            if pf.name in verified_set:
                dest = args.output_dir / pf.name
                shutil.move(str(pf), str(dest))
                moved_ok += 1

                # Update tom_level/tom_reasoning if verify provided them
                if verify_results:
                    vr = next((r for r in verify_results if Path(r["file"]).name == pf.name), None)
                    if vr and vr.get("tom_level") is not None:
                        with open(dest) as f:
                            td = json.load(f)
                        td["tom_level"] = vr["tom_level"]
                        if vr.get("tom_reasoning"):
                            td["tom_reasoning"] = vr["tom_reasoning"]
                        with open(dest, "w") as f:
                            json.dump(td, f, indent=2)
            else:
                dest = args.failed_dir / pf.name
                shutil.move(str(pf), str(dest))
                moved_fail += 1

        # Move tasks that failed patching to failed dir too
        for r in patch_results:
            if r["status"] != "patched":
                src = args.input_dir / r["file"]
                if src.exists():
                    dest = args.failed_dir / r["file"]
                    shutil.copy2(str(src), str(dest))

        # Clean up staging dir
        if patched_dir and patched_dir.exists():
            try:
                patched_dir.rmdir()
            except OSError:
                pass

        print(f"  Moved to output: {moved_ok}")
        print(f"  Moved to failed: {moved_fail}")
        patch_fail_count = sum(1 for r in patch_results if r["status"] != "patched")
        if patch_fail_count:
            print(f"  Copied to failed (patch errors): {patch_fail_count}")

    # ---- Report ----
    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Migration complete in {elapsed:.1f}s")
    print(f"{'=' * 50}")

    summary = {
        "total": len(tasks),
        "patched": patched_ok,
        "patch_failed": patched_fail,
    }
    if verify_results:
        summary["verified"] = sum(1 for r in verify_results if r["success"])
        summary["verify_failed"] = sum(1 for r in verify_results if not r["success"])

    print(json.dumps(summary, indent=2))

    # Write detailed report
    report = {
        "summary": summary,
        "elapsed_seconds": round(elapsed, 1),
        "patch_results": patch_results,
        "verify_results": verify_results,
    }
    report_path = PROJECT_ROOT / "outputs" / "migration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report: {report_path}")


if __name__ == "__main__":
    main()
