#!/usr/bin/env python3
"""
Campaign system for organized benchmark evaluation.

Manages a single campaign definition (which models, modes, matchups to run),
orchestrates benchmark runs, and aggregates results into a clean directory.

The campaign lives at data/emtom/results/:
  campaign.json          — definition + run status
  leaderboard.json       — aggregated results matrix
  runs/                  — clean per-run results (summaries + planner logs)

Individual benchmark runs still land in outputs/emtom/ as usual.
After each run completes, the interesting bits are copied into data/emtom/results/runs/.

Usage:
    python -m emtom.scripts.campaign add --models gpt-5.2 kimi-k2.5 --modes text vision
    python -m emtom.scripts.campaign run
    python -m emtom.scripts.campaign run --only kimi-k2.5_text_cooperative
    python -m emtom.scripts.campaign status
    python -m emtom.scripts.campaign report
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "emtom" / "results"
CAMPAIGN_FILE = RESULTS_DIR / "campaign.json"
RUNS_DIR = RESULTS_DIR / "runs"
TASKS_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"

CATEGORIES = ["cooperative", "competitive", "mixed"]


# ---------------------------------------------------------------------------
# Campaign definition
# ---------------------------------------------------------------------------

def _count_tasks_by_category(tasks_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tf in tasks_dir.glob("*.json"):
        try:
            with open(tf) as f:
                cat = json.load(f).get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        except (json.JSONDecodeError, OSError):
            pass
    return counts


def _derive_runs(
    models: List[str],
    modes: List[str],
    matchups: List[Tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Derive the full set of run keys from models, modes, and matchups."""
    runs: Dict[str, Dict[str, Any]] = {}

    for model in models:
        for mode in modes:
            # Cooperative and mixed: one run each
            for cat in ["cooperative", "mixed"]:
                key = f"{model}_{mode}_{cat}"
                runs[key] = {
                    "model": model,
                    "mode": mode,
                    "category": cat,
                    "type": "solo",
                    "status": "pending",
                }

            # Competitive: matchup runs
            for model_a, model_b in matchups:
                if model not in (model_a, model_b):
                    continue
                # Only generate from the pair once (the loop over models would double it)
                if model != model_a:
                    continue
                for direction in ["forward", "swap"]:
                    if direction == "forward":
                        t0, t1 = model_a, model_b
                    else:
                        t0, t1 = model_b, model_a
                    key = f"{model_a}_vs_{model_b}_{mode}_competitive"
                    if direction == "swap":
                        key += "_swap"
                    runs[key] = {
                        "model_a": model_a,
                        "model_b": model_b,
                        "team_0": t0,
                        "team_1": t1,
                        "mode": mode,
                        "category": "competitive",
                        "type": "matchup",
                        "direction": direction,
                        "status": "pending",
                    }

    return runs


def cmd_add(args: argparse.Namespace) -> None:
    """Add models/modes to the campaign. Creates campaign.json if it doesn't exist."""
    # Load existing campaign if present
    existing: Dict[str, Any] = {}
    if CAMPAIGN_FILE.exists():
        with open(CAMPAIGN_FILE) as f:
            existing = json.load(f)

    # Merge models and modes with existing (additive)
    prev_models = existing.get("models", [])
    prev_modes = existing.get("modes", [])
    models = list(dict.fromkeys(prev_models + args.models))  # dedupe, preserve order
    modes = list(dict.fromkeys(prev_modes + args.modes))

    # All pairwise matchups from the full model list
    matchups = list(combinations(models, 2))

    # Derive the full run set from merged models/modes
    new_runs = _derive_runs(models, modes, matchups)

    # Preserve status of all existing runs (complete, failed, pending with output_dir, etc.)
    existing_runs = existing.get("runs", {})
    for key, run_def in new_runs.items():
        if key in existing_runs:
            new_runs[key] = existing_runs[key]

    task_counts = _count_tasks_by_category(TASKS_DIR)

    campaign = {
        "created_at": existing.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "modes": modes,
        "competitive_matchups": [list(m) for m in matchups],
        "tasks_dir": str(TASKS_DIR.relative_to(PROJECT_ROOT)),
        "task_counts": task_counts,
        "task_total": sum(task_counts.values()),
        "runs": new_runs,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAMPAIGN_FILE, "w") as f:
        json.dump(campaign, f, indent=2)

    added_models = [m for m in args.models if m not in prev_models]
    added_modes = [m for m in args.modes if m not in prev_modes]
    pending = sum(1 for r in new_runs.values() if r["status"] == "pending")
    complete = sum(1 for r in new_runs.values() if r["status"] == "complete")
    print(f"Campaign updated: {len(new_runs)} runs ({complete} complete, {pending} pending)")
    if added_models:
        print(f"  Added models: {added_models}")
    if added_modes:
        print(f"  Added modes: {added_modes}")
    print(f"  All models: {models}")
    print(f"  All modes: {modes}")
    print(f"  Matchups: {[f'{a} vs {b}' for a, b in matchups]}")
    print(f"  Tasks: {campaign['task_total']} ({task_counts})")
    print(f"  Saved to: {CAMPAIGN_FILE}")


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def _run_benchmark(
    run_key: str,
    run_def: Dict[str, Any],
    tasks_dir: Path,
    max_workers: int,
) -> Tuple[bool, str]:
    """Execute a single benchmark run via run_emtom.sh. Returns (success, output_dir)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUTS_DIR / f"{timestamp}-campaign-{run_key}"

    mode = run_def["mode"]
    category = run_def["category"]

    if run_def["type"] == "solo":
        model = run_def["model"]
        cmd = [
            str(PROJECT_ROOT / "emtom" / "run_emtom.sh"), "benchmark",
            "--model", model,
            "--tasks-dir", str(tasks_dir),
            "--category", category,
            "--observation-mode", mode,
            "--max-workers", str(max_workers),
            "--output-dir", str(output_dir),
            "--no-video",
        ]
    else:
        # Matchup: use team_0's model as the "base" model, override via team-model-map
        team_0 = run_def["team_0"]
        team_1 = run_def["team_1"]
        team_map = f"team_0={team_0},team_1={team_1}"
        # Use team_0 as the default model (team_model_map overrides per-team)
        cmd = [
            str(PROJECT_ROOT / "emtom" / "run_emtom.sh"), "benchmark",
            "--model", team_0,
            "--tasks-dir", str(tasks_dir),
            "--category", category,
            "--observation-mode", mode,
            "--max-workers", str(max_workers),
            "--output-dir", str(output_dir),
            "--team-model-map", team_map,
            "--no-video",
        ]

    print(f"\n{'=' * 60}")
    print(f"Running: {run_key}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    success = result.returncode == 0
    tag = "DONE" if success else "FAIL"
    print(f"\n[{tag}] {run_key} ({elapsed:.0f}s)")

    return success, str(output_dir)


def _collect_results(output_dir: str, run_key: str) -> None:
    """Copy benchmark summary and planner logs into data/emtom/results/runs/."""
    out_path = Path(output_dir)
    dest = RUNS_DIR / run_key
    dest.mkdir(parents=True, exist_ok=True)

    # Find benchmark_summary.json — check multiple locations
    summary_found = False
    for pattern in [
        out_path / "results" / "benchmark_summary.json",
        *sorted(out_path.glob("*-*agents/results/benchmark_summary.json")),
    ]:
        if pattern.exists():
            shutil.copy2(str(pattern), str(dest / "benchmark_summary.json"))
            summary_found = True
            # Also copy per-task planner logs
            results_dir = pattern.parent
            tasks_dest = dest / "tasks"
            tasks_dest.mkdir(exist_ok=True)
            for task_dir in sorted(results_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                log_dir = task_dir / "planner-log"
                if not log_dir.exists():
                    continue
                log_files = sorted(log_dir.glob("planner-log-*.json"))
                if log_files:
                    task_dest = tasks_dest / task_dir.name
                    task_dest.mkdir(exist_ok=True)
                    shutil.copy2(str(log_files[-1]), str(task_dest / "planner-log.json"))
            break  # Use first found

    # Also check parallel output layout (per-task subdirs at top level)
    if not summary_found:
        # Parallel mode: results are in {output_dir}/{task_stem}/benchmark-Nagents/results/
        tasks_dest = dest / "tasks"
        tasks_dest.mkdir(exist_ok=True)
        for task_wrapper in sorted(out_path.iterdir()):
            if not task_wrapper.is_dir() or task_wrapper.name in ("logs",):
                continue
            for bench_dir in task_wrapper.iterdir():
                if not bench_dir.is_dir() or not bench_dir.name.startswith("benchmark-"):
                    continue
                results_dir = bench_dir / "results"
                if not results_dir.exists():
                    continue
                sf = results_dir / "benchmark_summary.json"
                if sf.exists() and not summary_found:
                    # Use first summary as template
                    shutil.copy2(str(sf), str(dest / "benchmark_summary.json"))
                    summary_found = True
                for task_dir in sorted(results_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    log_dir = task_dir / "planner-log"
                    if not log_dir.exists():
                        continue
                    log_files = sorted(log_dir.glob("planner-log-*.json"))
                    if log_files:
                        td = tasks_dest / task_dir.name
                        td.mkdir(exist_ok=True)
                        shutil.copy2(str(log_files[-1]), str(td / "planner-log.json"))

    if not summary_found:
        print(f"  WARNING: no benchmark_summary.json found in {output_dir}")


def _aggregate_parallel_summaries(output_dir: str, model: str) -> Optional[Dict[str, Any]]:
    """Aggregate per-task benchmark summaries from parallel output layout."""
    out_path = Path(output_dir)
    all_results = []

    for task_wrapper in sorted(out_path.iterdir()):
        if not task_wrapper.is_dir() or task_wrapper.name in ("logs",):
            continue
        for bench_dir in task_wrapper.iterdir():
            if not bench_dir.is_dir() or not bench_dir.name.startswith("benchmark-"):
                continue
            sf = bench_dir / "results" / "benchmark_summary.json"
            if sf.exists():
                try:
                    with open(sf) as f:
                        summary = json.load(f)
                    all_results.extend(summary.get("results", []))
                except (json.JSONDecodeError, OSError):
                    pass

    if not all_results:
        return None

    passed = sum(1 for r in all_results if r.get("success"))
    failed = sum(1 for r in all_results if not r.get("skipped") and not r.get("success"))
    total = len(all_results)

    return {
        "model": model,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
        "results": all_results,
    }


def cmd_run(args: argparse.Namespace) -> None:
    """Run pending benchmark runs."""
    if not CAMPAIGN_FILE.exists():
        print("No campaign found. Run 'campaign add --models ...' first.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    runs = campaign["runs"]
    tasks_dir = PROJECT_ROOT / campaign["tasks_dir"]
    max_workers = args.max_workers

    # Filter to specific run if --only specified
    if args.only:
        if args.only not in runs:
            print(f"Run '{args.only}' not found. Available runs:")
            for key in sorted(runs.keys()):
                print(f"  {key} [{runs[key]['status']}]")
            sys.exit(1)
        to_run = {args.only: runs[args.only]}
    else:
        to_run = {k: v for k, v in runs.items() if v["status"] == "pending"}

    if not to_run:
        print("No pending runs. Use 'campaign status' to see all runs.")
        return

    print(f"Running {len(to_run)} benchmark(s) (max_workers={max_workers}):")
    for key in sorted(to_run.keys()):
        r = to_run[key]
        if r["type"] == "solo":
            print(f"  {key}: {r['model']} / {r['mode']} / {r['category']}")
        else:
            print(f"  {key}: {r['team_0']} vs {r['team_1']} / {r['mode']}")

    for key in sorted(to_run.keys()):
        run_def = to_run[key]
        success, output_dir = _run_benchmark(key, run_def, tasks_dir, max_workers)

        # Re-read campaign.json before updating so we don't clobber
        # changes made by concurrent 'campaign add' or other processes.
        with open(CAMPAIGN_FILE) as f:
            campaign = json.load(f)

        # Update only this run's entry
        campaign["runs"][key]["output_dir"] = output_dir
        if success:
            campaign["runs"][key]["status"] = "complete"
            campaign["runs"][key]["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Collect results into clean dir
            _collect_results(output_dir, key)

            # Write aggregated summary for parallel runs if needed
            dest = RUNS_DIR / key
            if not (dest / "benchmark_summary.json").exists():
                model = run_def.get("model", run_def.get("team_0", ""))
                agg = _aggregate_parallel_summaries(output_dir, model)
                if agg:
                    with open(dest / "benchmark_summary.json", "w") as f:
                        json.dump(agg, f, indent=2)
        else:
            campaign["runs"][key]["status"] = "failed"
            campaign["runs"][key]["failed_at"] = datetime.now(timezone.utc).isoformat()

        # Persist after each run
        campaign["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(CAMPAIGN_FILE, "w") as f:
            json.dump(campaign, f, indent=2)

    # Auto-generate report (re-read final state)
    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)
    _generate_report(campaign)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Print campaign status."""
    if not CAMPAIGN_FILE.exists():
        print("No campaign found. Run 'campaign add --models ...' first.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    runs = campaign["runs"]
    by_status = {"pending": 0, "complete": 0, "failed": 0, "running": 0}
    for r in runs.values():
        s = r.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1

    print(f"Campaign: {len(runs)} runs")
    print(f"  Complete: {by_status['complete']}")
    print(f"  Pending:  {by_status['pending']}")
    print(f"  Failed:   {by_status['failed']}")
    print(f"  Models:   {campaign['models']}")
    print(f"  Modes:    {campaign['modes']}")
    print()

    # Group by type for display
    solo_runs = {k: v for k, v in runs.items() if v.get("type") == "solo"}
    matchup_runs = {k: v for k, v in runs.items() if v.get("type") == "matchup"}

    if solo_runs:
        print("Solo runs (cooperative + mixed):")
        for key in sorted(solo_runs.keys()):
            r = solo_runs[key]
            status = r["status"].upper()
            pad = " " * max(0, 10 - len(status))
            print(f"  [{status}]{pad} {key}")

    if matchup_runs:
        print("\nCompetitive matchups:")
        for key in sorted(matchup_runs.keys()):
            r = matchup_runs[key]
            status = r["status"].upper()
            pad = " " * max(0, 10 - len(status))
            print(f"  [{status}]{pad} {key}")


# ---------------------------------------------------------------------------
# Report / Leaderboard
# ---------------------------------------------------------------------------

def _read_run_summary(run_key: str) -> Optional[Dict[str, Any]]:
    """Read benchmark_summary.json from the results dir."""
    sf = RUNS_DIR / run_key / "benchmark_summary.json"
    if not sf.exists():
        return None
    try:
        with open(sf) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _generate_report(campaign: Dict[str, Any]) -> None:
    """Generate leaderboard.json from completed runs."""
    runs = campaign["runs"]

    # Solo results: model × mode × category
    solo_results: Dict[str, Dict[str, Any]] = {}
    for key, run_def in runs.items():
        if run_def.get("type") != "solo" or run_def.get("status") != "complete":
            continue
        summary = _read_run_summary(key)
        if not summary:
            continue

        model = run_def["model"]
        mode = run_def["mode"]
        category = run_def["category"]
        model_mode = f"{model}_{mode}"

        if model_mode not in solo_results:
            solo_results[model_mode] = {"model": model, "mode": mode, "categories": {}}
        solo_results[model_mode]["categories"][category] = {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "pass_rate": summary.get("pass_rate", 0),
        }

    # Compute overall pass rates
    for model_mode, data in solo_results.items():
        total = sum(c["total"] for c in data["categories"].values())
        passed = sum(c["passed"] for c in data["categories"].values())
        data["overall"] = {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
        }

    # Matchup results
    matchup_results: Dict[str, Dict[str, Any]] = {}
    for key, run_def in runs.items():
        if run_def.get("type") != "matchup" or run_def.get("status") != "complete":
            continue
        summary = _read_run_summary(key)
        if not summary:
            continue

        model_a = run_def["model_a"]
        model_b = run_def["model_b"]
        mode = run_def["mode"]
        direction = run_def["direction"]
        pair_key = f"{model_a}_vs_{model_b}_{mode}"

        if pair_key not in matchup_results:
            matchup_results[pair_key] = {
                "model_a": model_a,
                "model_b": model_b,
                "mode": mode,
            }

        # Count wins from per-task results
        wins_t0 = 0
        wins_t1 = 0
        draws = 0
        for r in summary.get("results", []):
            eval_data = r.get("evaluation", {})
            winner = eval_data.get("winner")
            if winner == "team_0":
                wins_t0 += 1
            elif winner == "team_1":
                wins_t1 += 1
            else:
                draws += 1

        team_0 = run_def["team_0"]
        team_1 = run_def["team_1"]
        matchup_results[pair_key][direction] = {
            "team_0": team_0,
            "team_1": team_1,
            "team_0_wins": wins_t0,
            "team_1_wins": wins_t1,
            "draws": draws,
            "total": wins_t0 + wins_t1 + draws,
        }

    # Compute combined win rates for matchups
    for pair_key, data in matchup_results.items():
        fwd = data.get("forward", {})
        swap = data.get("swap", {})
        # In forward: team_0=model_a, team_1=model_b
        # In swap: team_0=model_b, team_1=model_a
        a_wins = fwd.get("team_0_wins", 0) + swap.get("team_1_wins", 0)
        b_wins = fwd.get("team_1_wins", 0) + swap.get("team_0_wins", 0)
        total = fwd.get("total", 0) + swap.get("total", 0)
        data["combined"] = {
            "model_a_wins": a_wins,
            "model_b_wins": b_wins,
            "draws": fwd.get("draws", 0) + swap.get("draws", 0),
            "total": total,
            "model_a_win_rate": (a_wins / total * 100) if total > 0 else 0,
        }

    leaderboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": campaign["models"],
        "modes": campaign["modes"],
        "solo": solo_results,
        "matchups": matchup_results,
    }

    leaderboard_file = RESULTS_DIR / "leaderboard.json"
    with open(leaderboard_file, "w") as f:
        json.dump(leaderboard, f, indent=2)

    # Print leaderboard
    print(f"\n{'=' * 70}")
    print("LEADERBOARD")
    print(f"{'=' * 70}")

    if solo_results:
        # Table header
        cats = ["cooperative", "mixed"]
        header = f"{'Model':<30} {'Overall':>8}"
        for cat in cats:
            header += f" {cat[:5]:>8}"
        print(header)
        print("-" * len(header))

        for model_mode in sorted(solo_results.keys()):
            data = solo_results[model_mode]
            overall = data["overall"]
            row = f"{model_mode:<30} {overall['pass_rate']:>7.1f}%"
            for cat in cats:
                cat_data = data["categories"].get(cat, {})
                pr = cat_data.get("pass_rate", 0)
                row += f" {pr:>7.1f}%"
            print(row)

    if matchup_results:
        print(f"\n{'Competitive Matchups':<40} {'A wins':>8} {'B wins':>8} {'Draws':>8} {'A rate':>8}")
        print("-" * 72)
        for pair_key in sorted(matchup_results.keys()):
            data = matchup_results[pair_key]
            c = data.get("combined", {})
            label = f"{data['model_a']} vs {data['model_b']} ({data['mode']})"
            print(
                f"{label:<40} {c.get('model_a_wins', 0):>8} "
                f"{c.get('model_b_wins', 0):>8} {c.get('draws', 0):>8} "
                f"{c.get('model_a_win_rate', 0):>7.1f}%"
            )

    print(f"\nSaved to: {leaderboard_file}")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate leaderboard from completed runs."""
    if not CAMPAIGN_FILE.exists():
        print("No campaign found. Run 'campaign add --models ...' first.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    _generate_report(campaign)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Campaign benchmark manager",
        prog="python -m emtom.scripts.campaign",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add models/modes to the campaign")
    p_add.add_argument("--models", nargs="+", required=True, help="Models to add")
    p_add.add_argument("--modes", nargs="+", default=["text"], help="Observation modes to add (text, vision)")

    # run
    p_run = sub.add_parser("run", help="Run pending benchmarks")
    p_run.add_argument("--only", default=None, help="Run only this specific run key")
    p_run.add_argument("--max-workers", type=int, default=50, help="Parallel workers per run")

    # status
    sub.add_parser("status", help="Show campaign status")

    # report
    sub.add_parser("report", help="Generate leaderboard from completed runs")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
