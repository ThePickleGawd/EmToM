#!/usr/bin/env python3
"""
Fix ambiguous task descriptions by grounding them against the functional PDDL goal.

For each task:
1. Parse the functional goal to extract target states (open/close/placement).
2. Compare against the task description and agent_secrets.
3. Use an LLM to rewrite the description to be unambiguous.
4. Write the fixed task back.

Usage:
    python emtom/scripts/fix_task_descriptions.py --tasks-dir data/emtom/tasks
    python emtom/scripts/fix_task_descriptions.py --task data/emtom/tasks/foo.json
    python emtom/scripts/fix_task_descriptions.py --tasks-dir data/emtom/tasks --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _needs_fix(task: Dict[str, Any], func_goal: str) -> Tuple[bool, List[str]]:
    """Check if a task description needs fixing. Returns (needs_fix, reasons)."""
    desc = task.get("task", "").lower()
    reasons = []

    # Check ambiguous words
    ambig_words = [
        "adjust", "specified", "correct state", "reset state",
        "particular pattern", "correct configuration", "designated state",
    ]
    for word in ambig_words:
        if word in desc:
            reasons.append(f"ambiguous word: '{word}'")

    # Check open/close goals have explicit directives
    open_goals = re.findall(r"\(is_open (\w+)\)", func_goal)
    close_goals = re.findall(r"\(is_closed (\w+)\)", func_goal)

    for obj in open_goals:
        m = re.match(r"([a-z_]+?)_\d+", obj)
        if not m:
            continue
        otype = m.group(1).replace("_", " ")
        # Check desc explicitly says to open this type
        if not re.search(rf"(?:open|leave.*open).*{otype}|{otype}.*(?:open|leave.*open)", desc):
            reasons.append(f"PDDL wants {obj} OPEN but desc doesn't say 'open'")

    for obj in close_goals:
        m = re.match(r"([a-z_]+?)_\d+", obj)
        if not m:
            continue
        otype = m.group(1).replace("_", " ")
        if not re.search(rf"(?:close|shut).*{otype}|{otype}.*(?:close|shut)", desc):
            reasons.append(f"PDDL wants {obj} CLOSED but desc doesn't say 'close'")

    # Check if "seal" is used (strongly implies close)
    if "seal" in desc and open_goals:
        reasons.append("desc uses 'seal' but PDDL has open goals — contradictory")

    return bool(reasons), reasons


REWRITE_PROMPT = """You are fixing a multi-agent task description to be unambiguous.

The PDDL functional goal defines the EXACT target end-state. The task description
and agent secrets must clearly communicate these targets to agents in natural language,
without using object IDs.

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

## Rules
1. Rewrite the `task` field so every open/close/placement goal is EXPLICITLY stated.
   Say "leave the kitchen cabinet open" not "adjust the kitchen cabinet" or "seal the cabinet."
2. When multiple same-type furniture exist, add a distinguishing feature:
   "the cabinet with the canned food on top" or "the cabinet near the fridge."
3. Do NOT include object IDs (cabinet_27, bottle_3, etc.) in `task` or secrets.
4. For competitive tasks: keep the global `task` neutral (don't reveal team targets),
   but fix each team's secrets to explicitly state their target states.
5. For cooperative/mixed tasks: the `task` field should state all shared goals explicitly.
6. Keep the overall theme/narrative — just make the target states unambiguous.
7. Keep secrets concise. Only fix the parts that reference target states ambiguously.

Respond with ONLY valid JSON:
{{
  "task": "<fixed task description>",
  "agent_secrets": {{"agent_0": "<fixed or unchanged>", ...}}
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
    secrets = task.get("agent_secrets", {})
    # Flatten list secrets to strings
    secrets_clean = {}
    for k, v in secrets.items():
        if isinstance(v, list):
            secrets_clean[k] = " ".join(str(x) for x in v)
        else:
            secrets_clean[k] = str(v)

    prompt = REWRITE_PROMPT.format(
        task_desc=task.get("task", ""),
        category=task.get("category", "unknown"),
        goal_states=goal_desc,
        scene_context=scene_ctx[:1500],
        issues="\n".join(f"- {r}" for r in reasons),
        secrets_json=json.dumps(secrets_clean, indent=2),
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
            old_secret = secrets_clean.get(agent_id, "")
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
    parser = argparse.ArgumentParser(description="Fix ambiguous task descriptions")
    parser.add_argument("--tasks-dir", default=str(PROJECT_ROOT / "data" / "emtom" / "tasks"))
    parser.add_argument("--task", default=None, help="Fix a single task file")
    parser.add_argument("--dry-run", action="store_true", help="Only report issues, don't fix")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model for rewriting")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    # Collect task files
    if args.task:
        task_files = [Path(args.task)]
    else:
        task_files = sorted(Path(args.tasks_dir).glob("*.json"))

    print(f"Processing {len(task_files)} tasks (dry_run={args.dry_run})")

    # Create LLM client
    llm = None
    if not args.dry_run:
        from habitat_llm.llm import instantiate_llm
        llm = instantiate_llm(
            "openai_chat",
            generation_params={"model": args.model, "temperature": 0.0, "max_tokens": 3000},
        )

    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = Counter()
    manifest = []

    def _process_one(task_path: Path):
        with open(task_path) as f:
            task = json.load(f)
        result = fix_task(task, llm, dry_run=args.dry_run)
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
    manifest_path = Path(args.tasks_dir) / "description_fix_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"results": manifest, "summary": dict(results)}, f, indent=2)
        f.write("\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
