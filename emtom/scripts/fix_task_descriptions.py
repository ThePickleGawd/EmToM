#!/usr/bin/env python3
"""
Fix task text and secrets so the public task stays high-level while private
secrets use explicit scene IDs for goal-critical references.

For each task:
1. Parse the functional goal to extract target states (open/close/placement).
2. Compare against the task description and agent_secrets.
3. Use an LLM to rewrite the task/secrets to match the public-vague /
   secret-explicit split.
4. Write the fixed task back.

Usage:
    python emtom/scripts/fix_task_descriptions.py --tasks-dir data/emtom/tasks
    python emtom/scripts/fix_task_descriptions.py --task data/emtom/tasks/foo.json
    python emtom/scripts/fix_task_descriptions.py --tasks-dir data/emtom/tasks --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "description_fix_manifests"


def _parse_functional_goal(task: Dict[str, Any]) -> str:
    """Get the functional goal PDDL, deriving it if needed."""
    func_goal = task.get("functional_goal_pddl", "")
    if func_goal:
        return func_goal
    try:
        from emtom.pddl.runtime_projection import project_runtime_from_problem
        proj = project_runtime_from_problem(task["problem_pddl"])
        return proj.functional_goal_pddl or ""
    except Exception:
        return ""


def _extract_goal_states(func_goal: str) -> Dict[str, List[Tuple[str, ...]]]:
    """Extract target states from functional goal PDDL."""
    states: Dict[str, List[Tuple[str, ...]]] = {
        "is_open": [],
        "is_closed": [],
        "is_on_top": [],
        "is_inside": [],
        "is_on_floor": [],
        "is_in_room": [],
        "is_clean": [],
        "is_dirty": [],
        "is_filled": [],
        "is_empty": [],
        "is_powered_on": [],
        "has_item": [],
    }
    for pred in states:
        for m in re.finditer(rf"\({pred}\s+([\w\s]+?)\)", func_goal):
            args = tuple(m.group(1).strip().split())
            states[pred].append(args)
    return {k: v for k, v in states.items() if v}


def _describe_goal_states(states: Dict[str, List[Tuple[str, ...]]]) -> str:
    """Build a human-readable summary of goal states."""
    lines = []
    for pred, args_list in sorted(states.items()):
        for args in args_list:
            if pred == "is_open":
                lines.append(f"- {args[0]} must be OPEN at the end")
            elif pred == "is_closed":
                lines.append(f"- {args[0]} must be CLOSED at the end")
            elif pred == "is_on_top":
                lines.append(f"- {args[0]} must be ON TOP of {args[1]} at the end")
            elif pred == "is_inside":
                lines.append(f"- {args[0]} must be INSIDE {args[1]} at the end")
            elif pred == "is_in_room":
                lines.append(f"- {args[0]} must be in room {args[1]} at the end")
            elif pred == "is_on_floor":
                lines.append(f"- {args[0]} must be on the floor in {args[1]} at the end")
            else:
                lines.append(f"- {pred}({', '.join(args)})")
    return "\n".join(lines)


def _get_scene_context(task: Dict[str, Any]) -> str:
    """Extract relevant scene context for disambiguation."""
    pddl = task.get("problem_pddl", "")
    init_match = re.search(r":init\s*\((.*?)\)\s*\(:goal", pddl, re.DOTALL)
    if not init_match:
        return ""
    init_block = init_match.group(1)
    # Extract object locations
    on_top = re.findall(r"\(is_on_top (\S+) (\S+)\)", init_block)
    inside = re.findall(r"\(is_inside (\S+) (\S+)\)", init_block)
    lines = []
    for obj, furn in on_top:
        lines.append(f"{obj} is on top of {furn}")
    for obj, furn in inside:
        lines.append(f"{obj} is inside {furn}")
    return "\n".join(lines)


def _collect_goal_ids(func_goal: str) -> List[str]:
    ids = sorted(set(re.findall(r"\b[a-z_]+_\d+\b", func_goal)))
    return ids


def _contains_any_id(text: str, ids: List[str]) -> bool:
    lowered = text.lower()
    return any(obj_id.lower() in lowered for obj_id in ids)


def _count_ids_in_text(text: str, ids: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for obj_id in ids if obj_id.lower() in lowered)


def _needs_fix(task: Dict[str, Any], func_goal: str) -> Tuple[bool, List[str]]:
    """Check if a task description needs fixing. Returns (needs_fix, reasons)."""
    desc = task.get("task", "").lower()
    reasons = []
    goal_ids = _collect_goal_ids(func_goal)

    epistemic_leak_phrases = (
        "must know",
        "knowledge is required",
        "knowing requirement",
        "so you know",
        "explicitly informed",
        "whole team agrees",
    )
    for phrase in epistemic_leak_phrases:
        if phrase in desc:
            reasons.append(f"task text still frames epistemic success directly: '{phrase}'")

    # Public task can stay high-level, but it should not directly contradict
    # the functional goal by using "seal" when the goal is to leave something open.
    if "seal" in desc and re.findall(r"\(is_open (\w+)\)", func_goal):
        reasons.append("desc uses 'seal' but PDDL has open goals — contradictory")

    public_id_count = _count_ids_in_text(task.get("task", ""), goal_ids)
    if public_id_count > 0:
        reasons.append("task text leaks exact goal-critical scene IDs")

    for agent_id, secret_val in task.get("agent_secrets", {}).items():
        if isinstance(secret_val, list):
            secret_text = " ".join(str(x) for x in secret_val)
        else:
            secret_text = str(secret_val)
        secret_lower = secret_text.lower()
        if (
            any(phrase in secret_lower for phrase in ("the cabinet", "the door", "the fridge", "the drawer"))
            and not _contains_any_id(secret_text, goal_ids)
        ):
            reasons.append(f"{agent_id} secret still uses ambiguous furniture references")
        if goal_ids and not _contains_any_id(secret_text, goal_ids):
            reasons.append(f"{agent_id} secret does not name any exact goal-critical IDs")
        for phrase in epistemic_leak_phrases:
            if phrase in secret_lower:
                reasons.append(f"{agent_id} secret still frames epistemic success directly: '{phrase}'")
        if re.search(r"\bso\s+agent_\d+\s+(?:knows|can know|will know|understands)\b", secret_lower):
            reasons.append(f"{agent_id} secret still frames epistemic success directly: 'so agent_'")

    return bool(reasons), reasons


REWRITE_PROMPT = """You are fixing a multi-agent task to match this benchmark rule:
- The public `task` should stay high-level and non-leaking.
- The private `agent_secrets` should use exact scene IDs for goal-critical targets.

The PDDL functional goal defines the EXACT target end-state. The public task and
agent secrets must communicate the task correctly without duplicating the same level
of detail.

## Current task description
{task_desc}

## Category
{category}

## PDDL functional goal states (ground truth)
{goal_states}

## Scene context (what's where at the start)
{scene_context}

## Issues found
{issues}

## Current agent secrets
{secrets_json}

## Goal-critical IDs that should appear in private secrets when relevant
{goal_ids}

## Rules
1. Rewrite the public `task` so it stays high-level and natural. It may describe the shared end-state without naming exact IDs.
2. Do NOT put exact scene IDs into the public `task` unless absolutely unavoidable.
3. Rewrite `agent_secrets` so goal-critical objects, furniture, rooms, and hidden constraints use exact scene IDs whenever the agent needs precise grounding.
4. Keep the target predicate explicit. Do not replace open/close/on-top-of goals with vague wording like "adjust", "seal", "configure", or "set correctly."
5. For competitive tasks: keep the global `task` neutral about team ownership, but secrets should still name exact IDs/states for the relevant branch each agent cares about.
6. Preserve the original secret structure: if a secret is currently a list of strings, return a list of strings; if it is currently a string, return a string.
7. Keep the overall theme/narrative.
8. A good secret may say an agent cannot enter `bedroom_1` but knows that `cup_1` is on `table_7` in that room. That kind of exact private grounding is desirable.
9. Do NOT describe `K()` knowledge as a runtime success condition. Remove phrases like "must know", "knowing requirement", "explicitly informed", or "the team must agree." If confirmation matters, phrase it as private information flow or coordination help, not as a success criterion.

Respond with ONLY valid JSON:
{{
  "task": "<fixed task description>",
  "agent_secrets": {{"agent_0": "<fixed secret with same type as input>", ...}}
}}
"""


def fix_task(
    task: Dict[str, Any],
    llm,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fix a single task's description. Returns the changes dict."""
    func_goal = _parse_functional_goal(task)
    if not func_goal:
        return {"status": "skipped", "reason": "no functional goal"}

    needs, reasons = _needs_fix(task, func_goal)
    if not needs:
        return {"status": "ok", "reason": "description already clear"}

    states = _extract_goal_states(func_goal)
    goal_desc = _describe_goal_states(states)
    scene_ctx = _get_scene_context(task)
    secrets = copy.deepcopy(task.get("agent_secrets", {}))
    goal_ids = _collect_goal_ids(func_goal)

    prompt = REWRITE_PROMPT.format(
        task_desc=task.get("task", ""),
        category=task.get("category", "unknown"),
        goal_states=goal_desc,
        scene_context=scene_ctx[:1500],
        issues="\n".join(f"- {r}" for r in reasons),
        secrets_json=json.dumps(secrets, indent=2),
        goal_ids=", ".join(goal_ids) if goal_ids else "(none)",
    )

    if dry_run:
        return {
            "status": "dry_run",
            "reasons": reasons,
            "goal_states": goal_desc,
        }

    # Call LLM
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.generate(prompt, request_timeout=30)
            break
        except Exception as exc:
            if attempt >= max_retries:
                return {"status": "error", "reason": str(exc)}
            time.sleep(2 ** attempt)

    # Parse response
    json_match = re.search(r"\{[\s\S]*\}", response or "")
    if not json_match:
        return {"status": "error", "reason": "failed to parse LLM response"}

    try:
        fix_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"status": "error", "reason": f"JSON parse error: {e}"}

    changes = {}
    new_task = fix_data.get("task", "")
    if new_task and new_task != task.get("task", ""):
        changes["task"] = {"old": task.get("task", ""), "new": new_task}

    new_secrets = fix_data.get("agent_secrets", {})
    if new_secrets:
        secret_changes = {}
        for agent_id, new_secret in new_secrets.items():
            old_secret = secrets.get(agent_id, "")
            if new_secret and new_secret != old_secret:
                secret_changes[agent_id] = {"old": old_secret, "new": new_secret}
        if secret_changes:
            changes["agent_secrets"] = secret_changes

    return {
        "status": "fixed" if changes else "unchanged",
        "reasons": reasons,
        "changes": changes,
        "fix_data": fix_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix task/secrets to match public-vague and secret-explicit grounding")
    parser.add_argument("--tasks-dir", default=str(PROJECT_ROOT / "data" / "emtom" / "tasks"))
    parser.add_argument("--task", default=None, help="Fix a single task file")
    parser.add_argument("--dry-run", action="store_true", help="Only report issues, don't fix")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model for rewriting")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--backup", action="store_true", help="Backup root task JSONs before rewriting")
    args = parser.parse_args()

    # Collect task files
    if args.task:
        task_files = [Path(args.task)]
    else:
        task_files = sorted(Path(args.tasks_dir).glob("*.json"))

    print(f"Processing {len(task_files)} tasks (dry_run={args.dry_run})")

    if args.backup and not args.dry_run and not args.task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(args.tasks_dir) / f"backup_pre_public_task_secret_id_fix_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for task_path in task_files:
            shutil.copy2(task_path, backup_dir / task_path.name)
        print(f"Backup: {backup_dir}")

    # Create LLM client
    thread_state = local()

    def _get_llm():
        if args.dry_run:
            return None
        llm = getattr(thread_state, "llm", None)
        if llm is None:
            from habitat_llm.llm import instantiate_llm

            llm = instantiate_llm(
                "openai_chat",
                generation_params={"model": args.model, "temperature": 0.0, "max_tokens": 3000},
            )
            thread_state.llm = llm
        return llm

    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = Counter()
    manifest = []

    def _process_one(task_path: Path):
        with open(task_path) as f:
            task = json.load(f)
        result = fix_task(task, _get_llm(), dry_run=args.dry_run)
        return task_path, task, result

    if args.dry_run or args.max_workers <= 1:
        futures_list = [_process_one(p) for p in task_files]
    else:
        futures_list = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(_process_one, p): p for p in task_files}
            for future in as_completed(future_map):
                try:
                    futures_list.append(future.result())
                except Exception as exc:
                    p = future_map[future]
                    futures_list.append((p, None, {"status": "error", "reason": str(exc)}))

    for task_path, task, result in futures_list:
        status = result.get("status", "unknown")
        results[status] += 1
        entry = {"file": str(task_path.name), "status": status}
        if result.get("reasons"):
            entry["reasons"] = result["reasons"]

        if status == "fixed" and not args.dry_run:
            fix_data = result.get("fix_data", {})
            # Apply fixes
            if fix_data.get("task"):
                task["task"] = fix_data["task"]
            if fix_data.get("agent_secrets"):
                for agent_id, new_secret in fix_data["agent_secrets"].items():
                    if agent_id in task.get("agent_secrets", {}):
                        task["agent_secrets"][agent_id] = new_secret
            # Write back
            with open(task_path, "w") as f:
                json.dump(task, f, indent=2)
                f.write("\n")
            print(f"[FIXED] {task_path.name}")
        elif status == "dry_run":
            print(f"[NEEDS FIX] {task_path.name}")
            for r in result.get("reasons", []):
                print(f"  - {r}")
        elif status == "ok":
            pass  # silent for OK tasks
        elif status == "error":
            print(f"[ERROR] {task_path.name}: {result.get('reason', '?')}")
        elif status == "skipped":
            pass

        manifest.append(entry)

    print(f"\n=== Summary ===")
    for status, count in results.most_common():
        print(f"  {status}: {count}")

    # Write manifest
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"description_fix_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        json.dump({"results": manifest, "summary": dict(results)}, f, indent=2)
        f.write("\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
