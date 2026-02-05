#!/usr/bin/env python3
"""Monitor bulk EMTOM task generation progress.

Parses log files from bulk_generate.sh and displays a live dashboard.

Usage:
    ./emtom/bulk_monitor.sh                           # Auto-detect latest run
    ./emtom/bulk_monitor.sh outputs/emtom/2026-02-03_19-22-36-bulk-generate
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def parse_log(log_path: Path) -> dict:
    """Parse a single generation log file."""
    info = {
        "file": log_path.name,
        "gpu": "",
        "category": "",
        "iteration": 0,
        "max_iterations": 0,
        "submitted": 0,
        "target": 0,
        "status": "unknown",
        "judge_passes": 0,
        "judge_fails": 0,
        "test_passes": 0,
        "test_fails": 0,
        "last_action": "",
        "fail_reason": "",
    }

    # Extract GPU/category from filename: gpu0_slot1_competitive.log
    match = re.match(r"gpu(\d+)_slot(\d+)_(\w+)\.log", log_path.name)
    if match:
        info["gpu"] = match.group(1)
        info["category"] = match.group(3)

    try:
        with open(log_path, "r", errors="replace") as f:
            # Read from end for efficiency on large files
            f.seek(0, 2)
            file_size = f.tell()
            # Read last 20KB for status
            read_size = min(file_size, 20_000)
            f.seek(max(0, file_size - read_size))
            tail = f.read()

            # Also read first 2KB for config
            f.seek(0)
            head = f.read(2000)
    except (OSError, IOError):
        info["status"] = "error"
        return info

    if not tail.strip():
        info["status"] = "starting"
        return info

    # Find latest iteration line
    iter_matches = list(re.finditer(
        r"Iteration (\d+)/(\d+) \| Submitted: (\d+)/(\d+)", tail
    ))
    if iter_matches:
        last = iter_matches[-1]
        info["iteration"] = int(last.group(1))
        info["max_iterations"] = int(last.group(2))
        info["submitted"] = int(last.group(3))
        info["target"] = int(last.group(4))

    # Count judge results in tail
    info["judge_passes"] = len(re.findall(r"\[Judge\] Result: PASS", tail))
    info["judge_fails"] = len(re.findall(r"\[Judge\] Result: FAIL", tail))

    # Count test results
    info["test_passes"] = len(re.findall(r"Result: PASSED", tail))
    info["test_fails"] = len(re.findall(r"Result: FAILED", tail))

    # Determine status (check failure before "Done!" since the shell wrapper
    # prints "Done!" even when the agent failed)
    if "Agent FAILED:" in tail:
        info["status"] = "failed"
        fail_match = re.search(r"Agent FAILED: (.+)", tail)
        if fail_match:
            info["fail_reason"] = fail_match.group(1)[:80]
    elif "Done!" in tail[-200:]:
        if info["submitted"] > 0 or re.search(r"Submitted: [1-9]", tail):
            info["status"] = "done"
        else:
            info["status"] = "done_empty"
    elif iter_matches:
        info["status"] = "running"
    else:
        info["status"] = "starting"

    # Last agent action
    action_matches = list(re.finditer(r"Action: (\w+)\[", tail))
    if action_matches:
        info["last_action"] = action_matches[-1].group(1)

    return info


def find_latest_bulk_dir(project_root: Path):
    """Find the most recent bulk-generate output directory."""
    outputs_dir = project_root / "outputs" / "emtom"
    if not outputs_dir.exists():
        return None
    bulk_dirs = sorted(outputs_dir.glob("*-bulk-generate"), reverse=True)
    return bulk_dirs[0] if bulk_dirs else None


def render_dashboard(log_dir, infos):
    """Render the monitoring dashboard."""
    # Clear screen
    print("\033[2J\033[H", end="")

    total = len(infos)
    running = sum(1 for i in infos if i["status"] == "running")
    done = sum(1 for i in infos if i["status"] == "done")
    done_empty = sum(1 for i in infos if i["status"] == "done_empty")
    failed = sum(1 for i in infos if i["status"] == "failed")
    starting = sum(1 for i in infos if i["status"] == "starting")
    total_submitted = sum(i["submitted"] for i in infos)
    total_target = sum(i["target"] for i in infos)

    print(f"{BOLD}EMTOM Bulk Generation Monitor{NC}")
    print(f"{DIM}{log_dir}{NC}")
    print(f"{DIM}Updated: {time.strftime('%H:%M:%S')}{NC}")
    print()

    # Summary bar
    parts = []
    if running:
        parts.append(f"{CYAN}{running} running{NC}")
    if done:
        parts.append(f"{GREEN}{done} done{NC}")
    if done_empty:
        parts.append(f"{YELLOW}{done_empty} done(0 tasks){NC}")
    if failed:
        parts.append(f"{RED}{failed} failed{NC}")
    if starting:
        parts.append(f"{DIM}{starting} starting{NC}")
    print(f"Processes: {' | '.join(parts)}  ({total} total)")

    if total_target > 0:
        print(f"Tasks:     {BOLD}{total_submitted}/{total_target}{NC} submitted")
    print()

    # Table header
    header = f"{'GPU':>3}  {'Category':<13} {'Status':<14} {'Progress':<16} {'Submitted':<11} {'Tests':<10} {'Judge':<10} {'Last Tool':<15}"
    print(f"{BOLD}{header}{NC}")
    print("─" * len(header))

    # Sort by GPU then category
    for info in sorted(infos, key=lambda x: (x["gpu"], x["category"])):
        gpu = info["gpu"]
        cat = info["category"]

        # Status with color
        status = info["status"]
        if status == "running":
            status_str = f"{CYAN}running{NC}"
        elif status == "done":
            status_str = f"{GREEN}done{NC}"
        elif status == "done_empty":
            status_str = f"{YELLOW}done (0){NC}"
        elif status == "failed":
            status_str = f"{RED}failed{NC}"
        elif status == "starting":
            status_str = f"{DIM}starting{NC}"
        else:
            status_str = f"{DIM}unknown{NC}"

        # Progress bar
        if info["max_iterations"] > 0:
            pct = info["iteration"] / info["max_iterations"]
            bar_width = 10
            filled = int(pct * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress = f"{bar} {pct:>3.0%}"
        else:
            progress = f"{'░' * 10}  - "

        # Submitted
        if info["target"] > 0:
            submitted = f"{info['submitted']}/{info['target']}"
        else:
            submitted = "-"

        # Test results
        if info["test_passes"] or info["test_fails"]:
            tests = f"{GREEN}{info['test_passes']}P{NC}/{RED}{info['test_fails']}F{NC}"
        else:
            tests = f"{DIM}-{NC}"

        # Judge results
        if info["judge_passes"] or info["judge_fails"]:
            judge = f"{GREEN}{info['judge_passes']}P{NC}/{RED}{info['judge_fails']}F{NC}"
        else:
            judge = f"{DIM}-{NC}"

        last_action = info["last_action"] or "-"

        print(
            f"{gpu:>3}  {cat:<13} {status_str:<23} {progress:<16} {submitted:<11} {tests:<19} {judge:<19} {last_action:<15}"
        )

    # Show failure reasons
    failures = [i for i in infos if i["fail_reason"]]
    if failures:
        print()
        print(f"{RED}{BOLD}Failures:{NC}")
        for info in failures:
            print(f"  GPU {info['gpu']} {info['category']}: {info['fail_reason']}")

    print()
    print(f"{DIM}Press Ctrl+C to exit{NC}")


def main():
    parser = argparse.ArgumentParser(description="Monitor EMTOM bulk generation")
    parser.add_argument(
        "log_dir",
        nargs="?",
        help="Path to bulk-generate log directory (auto-detects latest if omitted)",
    )
    parser.add_argument(
        "--interval", "-n", type=float, default=5,
        help="Refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print once and exit (no live refresh)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.log_dir:
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
    else:
        log_dir = find_latest_bulk_dir(project_root)
        if log_dir is None:
            print(f"{RED}No bulk-generate output found in outputs/emtom/{NC}")
            sys.exit(1)

    if not log_dir.exists():
        print(f"{RED}Directory not found: {log_dir}{NC}")
        sys.exit(1)

    log_files = sorted(log_dir.glob("gpu*.log"))
    if not log_files:
        print(f"{RED}No log files found in {log_dir}{NC}")
        sys.exit(1)

    try:
        while True:
            infos = [parse_log(f) for f in log_files]
            render_dashboard(log_dir, infos)

            if args.once:
                break

            # Exit if all processes are done
            all_done = all(
                i["status"] in ("done", "done_empty", "failed")
                for i in infos
            )
            if all_done:
                print(f"{GREEN}{BOLD}All processes finished.{NC}")
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
