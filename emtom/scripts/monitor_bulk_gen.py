#!/usr/bin/env python3
"""Monitor bulk generation run: report new tasks, flag missing/weak ToM,
and surface non-ToM failure modes from judge logs."""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

TASK_DIR = Path("data/emtom/tasks")
LOG_DIR = Path("outputs/bulk_gen_logs")
STATE_FILE = Path("/tmp/emtom_monitor_state.json")

# Judge criteria — these are the non-ToM failure modes we want to flag
NON_TOM_CRITERIA = [
    "agent_necessity",
    "secret_quality",
    "task_naturalness",
    "narrative_consistency",
    "goal_relevance",
    "mechanic_utilization",
    "pddl_solvability",
    "task_interdependence",
    "goal_opposition",
    "subgoal_tension",
]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_task_count": 0, "seen_tasks": [], "last_check": None}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_latest_log_dir():
    """Find the most recent bulk gen log directory."""
    if not LOG_DIR.exists():
        return None
    dirs = sorted(LOG_DIR.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        if d.is_dir() and "bulk-generate" in d.name:
            return d
    return None


def check_task_tom(task_path):
    """Check a task JSON for ToM quality. Returns (issues, task_info)."""
    issues = []
    try:
        data = json.loads(task_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"Failed to read: {e}"], {}

    info = {
        "title": data.get("title", "unknown"),
        "category": data.get("category", "unknown"),
        "tom_level": data.get("tom_level"),
        "num_agents": data.get("num_agents", 0),
        "mechanics": data.get("active_mechanics", []),
    }

    # Check tom_level
    tom_level = data.get("tom_level")
    if tom_level is None or tom_level == 0:
        issues.append(f"NO ToM: tom_level={tom_level}")

    # Check tom_reasoning
    tom_reasoning = data.get("tom_reasoning", "")
    if not tom_reasoning or len(tom_reasoning.strip()) < 50:
        issues.append(f"WEAK tom_reasoning: {len(tom_reasoning)} chars")

    # Check for weak literal ToM patterns
    if tom_reasoning:
        weak_patterns = [
            r"simply (relay|tell|inform|communicate)",
            r"just (need|has) to (tell|say|communicate|inform)",
            r"straightforward (relay|communication|report)",
        ]
        for pat in weak_patterns:
            if re.search(pat, tom_reasoning, re.IGNORECASE):
                issues.append(f"LITERAL ToM pattern: '{pat}' in tom_reasoning")
                break

    # Check problem_pddl for K() operator (epistemic goals)
    problem_pddl = data.get("problem_pddl", "")
    has_k_goal = "(K " in problem_pddl or "(K(" in problem_pddl
    if not has_k_goal and tom_level and tom_level >= 1:
        issues.append(f"tom_level={tom_level} but no K() in problem_pddl")

    # Check active_mechanics — tasks with no mechanics rarely need ToM
    mechanics = data.get("active_mechanics", [])
    bindings = data.get("mechanic_bindings", [])
    if not mechanics and not bindings:
        issues.append("No active mechanics or bindings")

    return issues, info


def scan_log_failures(log_dir):
    """Parse log files for judge failure patterns."""
    failures = {}
    if not log_dir or not log_dir.exists():
        return failures

    for log_file in log_dir.glob("*.log"):
        content = log_file.read_text()

        # Count judge FAILs and extract failure criteria
        fail_matches = re.findall(
            r"\[Judge\] Result: FAIL \(score: ([\d.]+)\) \[failures: (\d+)\]",
            content,
        )
        pass_matches = re.findall(
            r"\[Judge\] Result: PASS",
            content,
        )

        # Extract "Improve X (score)" suggestions which indicate failing criteria
        improve_matches = re.findall(
            r"Improve (\w+) \(([\d.]+)\)",
            content,
        )

        # Count submitted tasks
        submitted = re.findall(r"Submitted: (\d+)/", content)
        last_submitted = int(submitted[-1]) if submitted else 0

        # Count iterations
        iteration_matches = re.findall(r"Iteration (\d+)/", content)
        last_iteration = int(iteration_matches[-1]) if iteration_matches else 0

        slot_name = log_file.stem
        failures[slot_name] = {
            "judge_fails": len(fail_matches),
            "judge_passes": len(pass_matches),
            "submitted": last_submitted,
            "iteration": last_iteration,
            "failing_criteria": {},
        }

        for criterion, score in improve_matches:
            if criterion not in failures[slot_name]["failing_criteria"]:
                failures[slot_name]["failing_criteria"][criterion] = 0
            failures[slot_name]["failing_criteria"][criterion] += 1

    return failures


def main():
    state = load_state()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"  BULK GEN MONITOR — {now}")
    print(f"{'='*60}")

    # Check if bulk gen is still running
    import subprocess

    ps = subprocess.run(
        ["pgrep", "-f", "bulk_generate"],
        capture_output=True,
        text=True,
    )
    running = bool(ps.stdout.strip())
    print(f"\n  Status: {'RUNNING' if running else 'STOPPED'}")

    # Count all tasks
    all_tasks = sorted(TASK_DIR.glob("*.json"))
    total = len(all_tasks)
    seen = set(state.get("seen_tasks", []))
    new_tasks = [t for t in all_tasks if t.name not in seen]

    print(f"  Total tasks: {total} (+{len(new_tasks)} new since last check)")

    # Analyze new tasks
    tom_issues = []
    non_tom_issues = []
    good_tasks = []

    for task_path in new_tasks:
        issues, info = check_task_tom(task_path)
        if issues:
            tom_related = [i for i in issues if "ToM" in i or "tom_" in i or "K()" in i]
            other = [i for i in issues if i not in tom_related]

            if tom_related:
                tom_issues.append((task_path.name, info, tom_related))
            if other:
                non_tom_issues.append((task_path.name, info, other))
            if not tom_related:
                good_tasks.append((task_path.name, info))
        else:
            good_tasks.append((task_path.name, info))

    # Report ToM issues
    if tom_issues:
        print(f"\n  {'!'*50}")
        print(f"  TASKS WITHOUT PROPER ToM ({len(tom_issues)}):")
        print(f"  {'!'*50}")
        for name, info, issues in tom_issues:
            title = info.get("title", "?")[:50]
            cat = info.get("category", "?")
            print(f"\n    {name[:60]}...")
            print(f"      Title: {title} ({cat})")
            for issue in issues:
                print(f"      -> {issue}")

    # Report non-ToM issues
    if non_tom_issues:
        print(f"\n  NON-TOM ISSUES ({len(non_tom_issues)}):")
        for name, info, issues in non_tom_issues:
            title = info.get("title", "?")[:50]
            print(f"    {title}: {', '.join(issues)}")

    # Report good tasks
    if good_tasks:
        print(f"\n  GOOD TASKS ({len(good_tasks)}):")
        for name, info in good_tasks[:5]:  # Show first 5
            title = info.get("title", "?")[:50]
            cat = info.get("category", "?")
            tom = info.get("tom_level", "?")
            print(f"    [{cat}] K={tom} {title}")
        if len(good_tasks) > 5:
            print(f"    ... and {len(good_tasks) - 5} more")

    # Scan logs for failure mode distribution
    log_dir = get_latest_log_dir()
    if log_dir:
        failures = scan_log_failures(log_dir)

        # Aggregate
        total_fails = sum(f["judge_fails"] for f in failures.values())
        total_passes = sum(f["judge_passes"] for f in failures.values())
        total_submitted = sum(f["submitted"] for f in failures.values())
        total_iterations = sum(f["iteration"] for f in failures.values())

        criteria_totals = {}
        for slot_data in failures.values():
            for criterion, count in slot_data["failing_criteria"].items():
                criteria_totals[criterion] = criteria_totals.get(criterion, 0) + count

        print(f"\n  LOG ANALYSIS ({log_dir.name}):")
        print(f"    Judge: {total_passes} pass, {total_fails} fail")
        print(f"    Submitted: {total_submitted} tasks across {len(failures)} slots")
        print(f"    Total iterations consumed: {total_iterations}")

        if criteria_totals:
            print(f"\n    FAILURE CRITERIA DISTRIBUTION:")
            for criterion, count in sorted(
                criteria_totals.items(), key=lambda x: -x[1]
            ):
                is_tom = criterion in ("pddl_solvability", "tom_depth")
                marker = " [ToM-related]" if is_tom else ""
                print(f"      {criterion}: {count}{marker}")

    # Category distribution
    cats = {}
    for t in all_tasks:
        try:
            d = json.loads(t.read_text())
            c = d.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        except Exception:
            pass

    if cats:
        print(f"\n  CATEGORY DISTRIBUTION:")
        for cat, count in sorted(cats.items()):
            print(f"    {cat}: {count}")

    # Update state
    state["seen_tasks"] = [t.name for t in all_tasks]
    state["last_task_count"] = total
    state["last_check"] = now
    save_state(state)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
