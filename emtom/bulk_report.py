#!/usr/bin/env python3
"""Post-run report for bulk_generate.sh.

Parses log files from one or more bulk generation runs, classifies
errors, and prints a colored terminal summary.

Usage:
    python -m emtom.bulk_report <log_dir> [<log_dir2> ...]
    python -m emtom.bulk_report outputs/bulk_gen_logs/2026-02-03_17-52-37-bulk-generate

Called automatically by bulk_generate.sh after all processes finish.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── ANSI colors ──

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
MAGENTA = "\033[0;35m"
CYAN = "\033[0;36m"
WHITE = "\033[1;37m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

# Strip ANSI escape sequences for parsing log content
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


# ── Error classification ──

ERROR_CATEGORIES = {
    "environment": {
        "label": "Environment / Sim",
        "patterns": [
            r"navmesh",
            r"PathFinder",
            r"timed out",
            r"timeout",
            r"CUDA",
            r"out of memory",
            r"OOM",
            r"scene.*issue",
        ],
    },
    "evaluator_bug": {
        "label": "Evaluator Bug",
        "patterns": [
            r"ao_link_map",
            r"is_open\(\).*unexpected keyword",
            r"is_closed\(\).*unexpected keyword",
            r"evaluator bug",
            r"evaluation.*crash",
            r"predicate.*error",
        ],
    },
    "task_design": {
        "label": "Task Design",
        "patterns": [
            r"success.condition.*cannot",
            r"cannot express",
            r"schema.*mismatch",
            r"unsupported",
            r"violates requirements",
            r"constraint.*mismatch",
        ],
    },
    "tooling": {
        "label": "Tooling / Sandbox",
        "patterns": [
            r"command filter",
            r"sandbox",
            r"tool.*availability",
            r"apply_patch.*missing",
            r"Shell command",
        ],
    },
}


def classify_error(fail_reason: str) -> str:
    """Classify an Agent FAILED reason into an error category."""
    lower = fail_reason.lower()
    for cat_key, cat_info in ERROR_CATEGORIES.items():
        for pattern in cat_info["patterns"]:
            if re.search(pattern, lower):
                return cat_key
    return "other"


# ── Log parsing ──

@dataclass
class ProcessResult:
    """Parsed result from one generation process log."""
    log_file: str
    gpu: int
    slot: int
    category: str
    tasks_generated: int = 0
    iterations_used: int = 0
    max_iterations: int = 0
    failed: bool = False
    fail_reason: str = ""
    error_category: str = ""
    task_paths: List[str] = field(default_factory=list)
    process_exited_ok: bool = True


def parse_log(log_path: Path) -> ProcessResult:
    """Parse a single bulk generation log file."""
    # Extract GPU/slot/category from filename: gpu0_slot1_cooperative.log
    stem = log_path.stem
    m = re.match(r"gpu(\d+)_slot(\d+)_(\w+)", stem)
    if m:
        gpu, slot, category = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        gpu, slot, category = -1, -1, "unknown"

    result = ProcessResult(
        log_file=str(log_path),
        gpu=gpu,
        slot=slot,
        category=category,
    )

    try:
        content = log_path.read_text(errors="replace")
    except Exception:
        result.process_exited_ok = False
        return result

    clean = strip_ansi(content)

    # Parse "Tasks generated: N"
    tasks_match = re.findall(r"Tasks generated:\s*(\d+)", clean)
    if tasks_match:
        result.tasks_generated = int(tasks_match[-1])

    # Parse task file paths: "  - data/emtom/tasks/foo.json"
    for path_m in re.finditer(r"^\s+-\s+(.*\.json)\s*$", clean, re.MULTILINE):
        result.task_paths.append(path_m.group(1).strip())

    # Parse iteration progress: "Iteration N/M | Submitted: X/Y"
    iter_matches = re.findall(r"Iteration\s+(\d+)/(\d+)\s*\|", clean)
    if iter_matches:
        last_iter, max_iter = iter_matches[-1]
        result.iterations_used = int(last_iter)
        result.max_iterations = int(max_iter)

    # Parse agent failures
    fail_matches = re.findall(r"Agent FAILED:\s*(.+)", clean)
    if fail_matches:
        result.failed = True
        result.fail_reason = fail_matches[-1].strip()
        result.error_category = classify_error(result.fail_reason)

    # If no tasks generated and no explicit fail, mark as failed
    if result.tasks_generated == 0 and not result.failed:
        # Check for tracebacks or other hard errors
        if "Traceback" in clean or "Error:" in clean:
            result.failed = True
            # Try to extract a meaningful error line
            tb_lines = re.findall(r"(?:Error|Exception):\s*(.+)", clean)
            if tb_lines:
                result.fail_reason = tb_lines[-1].strip()[:200]
                result.error_category = classify_error(result.fail_reason)
            else:
                result.fail_reason = "(process error — check log)"
                result.error_category = "other"

    return result


@dataclass
class RunReport:
    """Aggregated report for one or more bulk generation runs."""
    log_dirs: List[str]
    results: List[ProcessResult] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    @property
    def total_processes(self) -> int:
        return len(self.results)

    @property
    def total_tasks(self) -> int:
        return sum(r.tasks_generated for r in self.results)

    @property
    def processes_with_tasks(self) -> int:
        return sum(1 for r in self.results if r.tasks_generated > 0)

    @property
    def processes_failed(self) -> int:
        return sum(1 for r in self.results if r.failed)

    @property
    def process_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.processes_with_tasks / self.total_processes * 100

    def tasks_by_category(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.results:
            counts[r.category] = counts.get(r.category, 0) + r.tasks_generated
        return counts

    def errors_by_category(self) -> Dict[str, List[ProcessResult]]:
        cats: Dict[str, List[ProcessResult]] = {}
        for r in self.results:
            if r.failed:
                key = r.error_category or "other"
                cats.setdefault(key, []).append(r)
        return cats

    def avg_iterations(self) -> float:
        iters = [r.iterations_used for r in self.results if r.iterations_used > 0]
        return sum(iters) / len(iters) if iters else 0.0


# ── Printing ──

def _bar(pct: float, width: int = 30) -> str:
    """Render a colored progress bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct >= 70:
        color = GREEN
    elif pct >= 40:
        color = YELLOW
    else:
        color = RED
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    remaining_min = int(minutes % 60)
    return f"{hours}h {remaining_min}m"


def print_report(report: RunReport) -> None:
    """Print the colored terminal report."""
    w = 76  # table width

    # ── Header ──
    print()
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    print(f"{BOLD}{CYAN}  BULK GENERATION REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    print()

    # ── Overview ──
    pct = report.process_success_rate
    print(f"  {BOLD}Runs parsed{RESET}       : {len(report.log_dirs)}")
    print(f"  {BOLD}Total processes{RESET}   : {report.total_processes}")
    if report.wall_clock_seconds > 0:
        print(f"  {BOLD}Wall-clock time{RESET}   : {_fmt_duration(report.wall_clock_seconds)}")
    print(f"  {BOLD}Avg iterations{RESET}    : {report.avg_iterations():.0f}")
    print()

    # ── Pass/fail bar ──
    print(f"  {BOLD}Process outcomes:{RESET}")
    print(f"    {_bar(pct)} {pct:.1f}%")
    print(
        f"    {GREEN}produced tasks: {report.processes_with_tasks}{RESET}  "
        f"{RED}failed: {report.processes_failed}{RESET}  "
        f"{DIM}total: {report.total_processes}{RESET}"
    )
    print()

    # ── Tasks summary ──
    print(f"{BOLD}{WHITE}  {'─' * (w - 4)}{RESET}")
    print(f"  {BOLD}TASKS GENERATED: {GREEN}{report.total_tasks}{RESET}")
    print(f"{BOLD}{WHITE}  {'─' * (w - 4)}{RESET}")
    print()

    by_cat = report.tasks_by_category()
    if by_cat:
        cat_header = f"  {BOLD}{'Category':<16} {'Tasks':>8} {'Share':>10}{RESET}"
        print(cat_header)
        print(f"  {DIM}{'─' * 36}{RESET}")
        total = max(report.total_tasks, 1)
        for cat in sorted(by_cat.keys()):
            count = by_cat[cat]
            share = count / total * 100
            color = GREEN if count > 0 else DIM
            print(f"  {color}{cat:<16} {count:>8} {share:>9.1f}%{RESET}")
        print()

    # ── Error breakdown ──
    errors = report.errors_by_category()
    if errors:
        total_errors = sum(len(v) for v in errors.values())
        print(f"{BOLD}{RED}  {'─' * (w - 4)}{RESET}")
        print(f"  {BOLD}{RED}ERRORS: {total_errors} process(es) failed{RESET}")
        print(f"{BOLD}{RED}  {'─' * (w - 4)}{RESET}")
        print()

        err_header = f"  {BOLD}{'Category':<22} {'Count':>7} {'Share':>10}{RESET}"
        print(err_header)
        print(f"  {DIM}{'─' * 42}{RESET}")

        for cat_key in sorted(errors.keys(), key=lambda k: -len(errors[k])):
            results_list = errors[cat_key]
            count = len(results_list)
            share = count / total_errors * 100
            label = ERROR_CATEGORIES.get(cat_key, {}).get("label", cat_key.title())
            color = RED if count >= 5 else YELLOW
            print(f"  {color}{label:<22} {count:>7} {share:>9.1f}%{RESET}")

        print()

        # Show top unique error messages
        print(f"  {BOLD}Top error messages:{RESET}")
        all_reasons: Dict[str, int] = {}
        for cat_results in errors.values():
            for r in cat_results:
                # Truncate and normalize
                reason = r.fail_reason[:120]
                all_reasons[reason] = all_reasons.get(reason, 0) + 1

        sorted_reasons = sorted(all_reasons.items(), key=lambda x: -x[1])
        for i, (reason, count) in enumerate(sorted_reasons[:8]):
            cat = classify_error(reason)
            cat_label = ERROR_CATEGORIES.get(cat, {}).get("label", "Other")
            color = RED if cat in ("environment", "evaluator_bug") else YELLOW
            print(f"    {color}{count:>3}x{RESET} [{DIM}{cat_label}{RESET}] {reason}")

        print()

    # ── Per-GPU summary ──
    gpus: Dict[int, List[ProcessResult]] = {}
    for r in report.results:
        gpus.setdefault(r.gpu, []).append(r)

    if len(gpus) > 1:
        print(f"  {BOLD}Per-GPU breakdown:{RESET}")
        gpu_header = f"  {'GPU':<6} {'Procs':>7} {'Tasks':>7} {'Failed':>8} {'Rate':>8}"
        print(f"  {BOLD}{gpu_header.strip()}{RESET}")
        print(f"  {DIM}{'─' * 40}{RESET}")
        for gpu_id in sorted(gpus.keys()):
            gpu_results = gpus[gpu_id]
            procs = len(gpu_results)
            tasks = sum(r.tasks_generated for r in gpu_results)
            fails = sum(1 for r in gpu_results if r.failed)
            rate = (tasks / procs * 100) if procs > 0 else 0
            color = GREEN if rate > 50 else (YELLOW if rate > 0 else RED)
            print(
                f"  {color}GPU {gpu_id:<3} {procs:>7} {tasks:>7} "
                f"{fails:>8} {rate:>7.0f}%{RESET}"
            )
        print()

    # ── Task files ──
    all_paths = []
    for r in report.results:
        all_paths.extend(r.task_paths)
    if all_paths:
        print(f"  {BOLD}{GREEN}Generated task files ({len(all_paths)}):{RESET}")
        for p in sorted(set(all_paths)):
            print(f"    {CYAN}{p}{RESET}")
        print()

    # ── Footer ──
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    print()


# ── Entry point ──

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk generation report — parses log dirs and prints summary"
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        help="One or more bulk generation log directories",
    )
    parser.add_argument(
        "--wall-clock",
        type=float,
        default=0.0,
        help="Total wall-clock seconds (passed by bulk_generate.sh)",
    )
    return parser.parse_args()


def build_report(log_dirs: List[str], wall_clock: float = 0.0) -> RunReport:
    """Build a report from one or more log directories."""
    report = RunReport(log_dirs=log_dirs, wall_clock_seconds=wall_clock)

    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"{YELLOW}Warning: log dir not found: {log_dir}{RESET}", file=sys.stderr)
            continue
        for log_file in sorted(log_path.glob("*.log")):
            result = parse_log(log_file)
            report.results.append(result)

    return report


def main():
    args = parse_args()
    report = build_report(args.log_dirs, wall_clock=args.wall_clock)
    if not report.results:
        print(f"{RED}No log files found in: {', '.join(args.log_dirs)}{RESET}")
        sys.exit(1)
    print_report(report)


if __name__ == "__main__":
    main()
