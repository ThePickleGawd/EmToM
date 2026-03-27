#!/usr/bin/env python3
"""
Campaign system for organized benchmark evaluation.

One active campaign tracks all models, modes, and matchups.
Invalidated campaigns can be archived for the visualizer and replaced
with a fresh active campaign.

The active campaign lives at data/emtom/results/:
  campaign.json          — definition + run status + per-run completed_tasks
  leaderboard.json       — aggregated results matrix
  runs/                  — clean per-run results (summaries + planner logs)

Archived campaigns live at data/emtom/results/archives/<campaign_id>/.

Usage:
    python -m emtom.scripts.campaign add --models gpt-5.2 kimi-k2.5 --modes text vision
    python -m emtom.scripts.campaign archive --campaign-id pre_literal_tom --label "Pre literal ToM"
    python -m emtom.scripts.campaign run
    python -m emtom.scripts.campaign run --only kimi-k2.5_text_cooperative
    python -m emtom.scripts.campaign status
    python -m emtom.scripts.campaign list
    python -m emtom.scripts.campaign report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "emtom" / "results"
CAMPAIGN_FILE = RESULTS_DIR / "campaign.json"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.json"
RUNS_DIR = RESULTS_DIR / "runs"
ARCHIVES_DIR = RESULTS_DIR / "archives"
TASKS_DIR = PROJECT_ROOT / "data" / "emtom" / "tasks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "emtom"

CATEGORIES = ["cooperative", "competitive", "mixed"]


# ---------------------------------------------------------------------------
# Campaign definition
# ---------------------------------------------------------------------------

SUMMARY_CORE_KEYS = {
    "model",
    "llm_provider",
    "benchmark_observation_mode",
    "run_mode",
    "task_category_filter",
    "team_model_map_requested",
    "team_model_map_resolved",
    "total",
    "passed",
    "failed",
    "skipped",
    "pass_rate",
    "literal_tom_score",
    "literal_tom_task_count",
    "literal_tom_probe_count",
    "literal_tom_supported_probe_count",
    "literal_tom_passed_probe_count",
    "category_stats",
    "results",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_campaign_metadata(
    campaign: Dict[str, Any],
    *,
    campaign_id: str = "active",
    label: Optional[str] = None,
    status: str = "active",
) -> Dict[str, Any]:
    campaign["campaign_id"] = campaign.get("campaign_id", campaign_id)
    campaign["label"] = campaign.get("label", label or "Active Campaign")
    campaign["status"] = campaign.get("status", status)
    return campaign


def _archive_campaign_dir(campaign_id: str) -> Path:
    return ARCHIVES_DIR / campaign_id


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _literal_tom_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate literal-ToM scores from per-task evaluation payloads."""
    scored_task_count = 0
    fallback_score_sum = 0.0
    probe_count = 0
    supported_probe_count = 0
    passed_probe_count = 0

    for result in results:
        if result.get("skipped"):
            continue
        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict):
            continue

        probe_summary = evaluation.get("literal_tom_probe_summary")
        if isinstance(probe_summary, dict):
            raw_probe_count = _coerce_number(probe_summary.get("probe_count")) or 0.0
            raw_supported = _coerce_number(probe_summary.get("supported_probe_count")) or 0.0
            raw_passed = _coerce_number(probe_summary.get("passed_count")) or 0.0
            probe_count += int(raw_probe_count)
            supported_probe_count += int(raw_supported)
            passed_probe_count += int(raw_passed)
            if raw_supported > 0:
                scored_task_count += 1
                continue

        score = _coerce_number(evaluation.get("literal_tom_probe_score"))
        if score is not None:
            scored_task_count += 1
            fallback_score_sum += score

    score_pct: Optional[float] = None
    if supported_probe_count > 0:
        score_pct = passed_probe_count / supported_probe_count * 100
    elif scored_task_count > 0:
        score_pct = fallback_score_sum / scored_task_count * 100

    return {
        "literal_tom_score": round(score_pct, 1) if score_pct is not None else None,
        "literal_tom_task_count": scored_task_count,
        "literal_tom_probe_count": probe_count,
        "literal_tom_supported_probe_count": supported_probe_count,
        "literal_tom_passed_probe_count": passed_probe_count,
    }


def _category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build simple per-category aggregate stats for campaign summaries."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if result.get("skipped"):
            continue
        category = result.get("category", "unknown")
        grouped.setdefault(category, []).append(result)

    stats: Dict[str, Dict[str, Any]] = {}
    for category, cat_results in grouped.items():
        total = len(cat_results)
        passed = sum(1 for r in cat_results if r.get("success"))
        timed_out = sum(1 for r in cat_results if not r.get("done", True))
        avg_steps = sum(float(r.get("steps", 0) or 0) for r in cat_results) / total if total else 0.0
        avg_progress = 0.0
        for result in cat_results:
            evaluation = result.get("evaluation")
            if isinstance(evaluation, dict):
                avg_progress += float(evaluation.get("percent_complete", 0.0) or 0.0)
        avg_progress = avg_progress / total if total else 0.0

        cat_stats = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total * 100 if total else 0.0,
            "avg_progress": round(avg_progress, 3),
            "avg_steps": round(avg_steps, 1),
            "timed_out": timed_out,
        }
        cat_stats.update(_literal_tom_metrics(cat_results))
        stats[category] = cat_stats

    return stats


def _summary_payload(
    results: List[Dict[str, Any]],
    *,
    model: str = "",
    base_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized benchmark summary payload from per-task results."""
    summary = dict(base_summary or {})
    total = len(results)
    skipped = sum(1 for r in results if r.get("skipped"))
    passed = sum(1 for r in results if r.get("success"))
    evaluated = total - skipped
    failed = sum(1 for r in results if not r.get("skipped") and not r.get("success"))

    summary.update({
        "model": model or summary.get("model", ""),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": (passed / evaluated * 100) if evaluated > 0 else 0,
        "category_stats": _category_stats(results),
        "results": results,
    })
    summary.update(_literal_tom_metrics(results))
    return summary


def _load_campaign(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _list_archived_campaigns() -> List[Dict[str, Any]]:
    archives: List[Dict[str, Any]] = []
    if not ARCHIVES_DIR.exists():
        return archives
    for archive_dir in sorted(ARCHIVES_DIR.iterdir(), reverse=True):
        if not archive_dir.is_dir():
            continue
        campaign = _load_campaign(archive_dir / "campaign.json")
        if not campaign:
            continue
        archives.append({
            "campaign_id": campaign.get("campaign_id", archive_dir.name),
            "label": campaign.get("label", archive_dir.name),
            "status": campaign.get("status", "archived"),
            "created_at": campaign.get("created_at"),
            "updated_at": campaign.get("updated_at"),
            "archived_at": campaign.get("archived_at"),
            "archive_reason": campaign.get("archive_reason", ""),
            "task_total": campaign.get("task_total", 0),
            "models": campaign.get("models", []),
            "modes": campaign.get("modes", []),
        })
    return archives

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


def _get_task_ids_by_category(tasks_dir: Path, category: str) -> Dict[str, Path]:
    """Return {task_id: file_path} for all tasks matching category."""
    result: Dict[str, Path] = {}
    for tf in tasks_dir.glob("*.json"):
        try:
            with open(tf) as f:
                data = json.load(f)
            if data.get("category") == category:
                tid = data.get("task_id", "")
                if tid:
                    result[tid] = tf
        except (json.JSONDecodeError, OSError):
            pass
    return result


def _backfill_completed_tasks(run_key: str) -> List[str]:
    """Extract task_ids from an existing benchmark_summary.json for backfill."""
    summary = _read_run_summary(run_key)
    if not summary:
        return []
    return [r["task_id"] for r in summary.get("results", []) if r.get("task_id")]


def _merge_results(
    summary_path: Path,
    new_summary: Dict[str, Any],
    model: str,
) -> None:
    """Merge new per-task results into an existing benchmark_summary.json."""
    existing: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    all_results = existing.get("results", [])
    # Deduplicate: new results replace any existing entries with same task_id
    new_results = new_summary.get("results", [])
    new_ids = {r["task_id"] for r in new_results if r.get("task_id")}
    all_results = [r for r in all_results if r.get("task_id") not in new_ids]
    all_results.extend(new_results)
    merged_base = {k: v for k, v in existing.items() if k not in SUMMARY_CORE_KEYS}
    for key, value in new_summary.items():
        if key not in SUMMARY_CORE_KEYS:
            merged_base[key] = value
    merged = _summary_payload(
        all_results,
        model=model or new_summary.get("model", "") or existing.get("model", ""),
        base_summary=merged_base,
    )
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)


def _prune_results(
    run_key: str,
    stale_ids: Set[str],
) -> None:
    """Remove stale task results from benchmark_summary and planner logs."""
    summary_path = RUNS_DIR / run_key / "benchmark_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        results = [r for r in summary.get("results", []) if r.get("task_id") not in stale_ids]
        base = {k: v for k, v in summary.items() if k not in SUMMARY_CORE_KEYS}
        summary = _summary_payload(results, model=summary.get("model", ""), base_summary=base)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Remove planner log dirs for stale tasks
    tasks_dest = RUNS_DIR / run_key / "tasks"
    if tasks_dest.exists():
        for tid in stale_ids:
            task_log_dir = tasks_dest / tid
            if task_log_dir.exists():
                shutil.rmtree(str(task_log_dir), ignore_errors=True)


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


def _validate_model(model: str) -> None:
    """Raise ValueError if model name won't resolve to a known provider."""
    from emtom.examples.run_habitat_benchmark import resolve_model_spec

    try:
        resolve_model_spec(model)
    except ValueError:
        from emtom.examples.run_habitat_benchmark import (
            CLAUDE_ALIAS_MODELS,
            MODEL_PROVIDER_MAP,
        )
        known = sorted(set(MODEL_PROVIDER_MAP.keys()) | CLAUDE_ALIAS_MODELS)
        raise ValueError(
            f"Unknown model '{model}'. Known models:\n  " + "\n  ".join(known)
        )


def cmd_add(args: argparse.Namespace) -> None:
    """Add models/modes to the campaign. Creates campaign.json if it doesn't exist."""
    # Validate all models before making any changes
    for model in args.models:
        try:
            _validate_model(model)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

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

    campaign = _ensure_campaign_metadata({
        "created_at": existing.get("created_at", _now_iso()),
        "updated_at": _now_iso(),
        "models": models,
        "modes": modes,
        "competitive_matchups": [list(m) for m in matchups],
        "tasks_dir": str(TASKS_DIR.relative_to(PROJECT_ROOT)),
        "task_counts": task_counts,
        "task_total": sum(task_counts.values()),
        "runs": new_runs,
    })

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


def _collect_results(output_dir: str, run_key: str, model: str = "", append: bool = False) -> None:
    """Copy benchmark summary and planner logs into data/emtom/results/runs/.

    When append=True, merge new results into existing summary instead of overwriting.
    """
    out_path = Path(output_dir)
    dest = RUNS_DIR / run_key
    dest.mkdir(parents=True, exist_ok=True)

    tasks_dest = dest / "tasks"
    tasks_dest.mkdir(exist_ok=True)

    summary_dest = dest / "benchmark_summary.json"

    # Check flat layout first: {output_dir}/results/benchmark_summary.json
    flat_summary = out_path / "results" / "benchmark_summary.json"
    if flat_summary.exists():
        if append:
            with open(flat_summary) as f:
                new_data = json.load(f)
            _merge_results(summary_dest, new_data, model)
        else:
            shutil.copy2(str(flat_summary), str(summary_dest))
        results_dir = flat_summary.parent
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
        return

    # Parallel layout: {output_dir}/{task_stem}/benchmark-Nagents/results/
    # Collect planner logs from each per-task result dir
    for task_wrapper in sorted(out_path.iterdir()):
        if not task_wrapper.is_dir() or task_wrapper.name in ("logs",):
            continue
        for bench_dir in task_wrapper.iterdir():
            if not bench_dir.is_dir() or not bench_dir.name.startswith("benchmark-"):
                continue
            results_dir = bench_dir / "results"
            if not results_dir.exists():
                continue
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

    # Aggregate per-task summaries into one benchmark_summary.json
    agg = _aggregate_parallel_summaries(output_dir, model)
    if agg:
        if append:
            _merge_results(summary_dest, agg, model)
        else:
            with open(summary_dest, "w") as f:
                json.dump(agg, f, indent=2)
    else:
        print(f"  WARNING: no benchmark results found in {output_dir}")


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

    return _summary_payload(all_results, model=model)


def _sync_run(
    run_key: str,
    run_def: Dict[str, Any],
    tasks_dir: Path,
) -> Tuple[Set[str], Set[str]]:
    """Compute new and stale task_ids for a run. Returns (new_ids, stale_ids).

    Also backfills completed_tasks from existing summary if missing.
    """
    category = run_def["category"]
    current = _get_task_ids_by_category(tasks_dir, category)
    current_ids = set(current.keys())

    completed = set(run_def.get("completed_tasks", []))
    if not completed and run_def.get("status") == "complete":
        # Backfill from existing benchmark_summary.json
        completed = set(_backfill_completed_tasks(run_key))
        if completed:
            run_def["completed_tasks"] = sorted(completed)

    new_ids = current_ids - completed
    stale_ids = completed - current_ids
    return new_ids, stale_ids


def cmd_run(args: argparse.Namespace) -> None:
    """Run pending and incremental benchmark runs.

    Pending runs benchmark all tasks. Complete runs detect new/deleted tasks
    and incrementally update results.
    """
    if not CAMPAIGN_FILE.exists():
        print("No campaign found. Run 'campaign add --models ...' first.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    runs = campaign["runs"]
    tasks_dir = PROJECT_ROOT / campaign["tasks_dir"]
    max_workers = args.max_workers

    # Build the set of runs that need work
    to_run: Dict[str, Dict[str, Any]] = {}
    incremental_info: Dict[str, Tuple[Set[str], Set[str]]] = {}  # key -> (new_ids, stale_ids)
    campaign_dirty = False

    # Filter by --only or --models
    model_filter = set(args.models) if args.models else None

    if args.only:
        if args.only not in runs:
            print(f"Run '{args.only}' not found. Available runs:")
            for key in sorted(runs.keys()):
                print(f"  {key} [{runs[key]['status']}]")
            sys.exit(1)
        candidates = {args.only: runs[args.only]}
    elif model_filter:
        candidates = {}
        for key, run_def in runs.items():
            # Solo runs: check "model". Matchup runs: check both "model_a" and "model_b".
            run_models = set()
            if run_def.get("model"):
                run_models.add(run_def["model"])
            if run_def.get("model_a"):
                run_models.add(run_def["model_a"])
            if run_def.get("model_b"):
                run_models.add(run_def["model_b"])
            if run_models & model_filter:
                candidates[key] = run_def
    else:
        candidates = runs

    for key, run_def in candidates.items():
        if run_def["status"] == "pending":
            to_run[key] = run_def
        elif run_def["status"] in ("complete", "failed"):
            new_ids, stale_ids = _sync_run(key, run_def, tasks_dir)
            if "completed_tasks" not in run_def:
                # backfill happened inside _sync_run
                campaign_dirty = True
            if stale_ids:
                print(f"  Pruning {len(stale_ids)} deleted task(s) from {key}")
                _prune_results(key, stale_ids)
                completed = set(run_def.get("completed_tasks", []))
                run_def["completed_tasks"] = sorted(completed - stale_ids)
                campaign_dirty = True
            if new_ids:
                to_run[key] = run_def
                incremental_info[key] = (new_ids, stale_ids)

    # Persist any backfill / prune changes before running benchmarks
    if campaign_dirty:
        campaign["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(CAMPAIGN_FILE, "w") as f:
            json.dump(campaign, f, indent=2)

    if not to_run:
        print("All runs up to date. Use 'campaign status' to see all runs.")
        return

    # Print plan
    print(f"Running {len(to_run)} benchmark(s) (max_workers={max_workers}):")
    for key in sorted(to_run.keys()):
        r = to_run[key]
        label = ""
        if r["type"] == "solo":
            label = f"{r['model']} / {r['mode']} / {r['category']}"
        else:
            label = f"{r['team_0']} vs {r['team_1']} / {r['mode']}"
        if key in incremental_info:
            new_ids, _ = incremental_info[key]
            label += f" (+{len(new_ids)} new tasks)"
        print(f"  {key}: {label}")

    # Execute each run
    for key in sorted(to_run.keys()):
        run_def = to_run[key]
        is_incremental = key in incremental_info
        tmp_dir = None

        try:
            if is_incremental:
                # Create temp dir with symlinks to only new tasks
                new_ids, _ = incremental_info[key]
                category = run_def["category"]
                current = _get_task_ids_by_category(tasks_dir, category)
                tmp_dir = tempfile.mkdtemp(prefix=f"campaign_incr_{key}_")
                for tid in new_ids:
                    src = current.get(tid)
                    if src and src.exists():
                        os.symlink(str(src.resolve()), os.path.join(tmp_dir, src.name))
                benchmark_dir = Path(tmp_dir)
            else:
                benchmark_dir = tasks_dir

            success, output_dir = _run_benchmark(key, run_def, benchmark_dir, max_workers)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Re-read campaign.json before updating so we don't clobber
        # changes made by concurrent 'campaign add' or other processes.
        with open(CAMPAIGN_FILE) as f:
            campaign = json.load(f)

        campaign["runs"][key]["output_dir"] = output_dir
        if success:
            campaign["runs"][key]["status"] = "complete"
            campaign["runs"][key]["completed_at"] = datetime.now(timezone.utc).isoformat()

            model = run_def.get("model", run_def.get("team_0", ""))
            _collect_results(output_dir, key, model=model, append=is_incremental)

            # Update completed_tasks from the (merged) summary
            summary = _read_run_summary(key)
            if summary:
                all_ids = [r["task_id"] for r in summary.get("results", []) if r.get("task_id")]
                campaign["runs"][key]["completed_tasks"] = sorted(set(all_ids))
        else:
            if not is_incremental:
                campaign["runs"][key]["status"] = "failed"
                campaign["runs"][key]["failed_at"] = datetime.now(timezone.utc).isoformat()
            # For incremental failures, keep status="complete" with old results intact

        campaign["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(CAMPAIGN_FILE, "w") as f:
            json.dump(campaign, f, indent=2)

        # Update leaderboard after each run for live visibility
        _generate_report(campaign)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Print campaign status, including new/stale task counts."""
    if not CAMPAIGN_FILE.exists():
        print("No campaign found. Run 'campaign add --models ...' first.")
        archives = _list_archived_campaigns()
        if archives:
            print(f"Archived campaigns: {len(archives)}")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    runs = campaign["runs"]
    tasks_dir = PROJECT_ROOT / campaign["tasks_dir"]
    by_status = {"pending": 0, "complete": 0, "failed": 0, "running": 0}
    for r in runs.values():
        s = r.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1

    task_counts = _count_tasks_by_category(tasks_dir)
    print(f"Campaign: {campaign.get('label', 'Active Campaign')} ({campaign.get('campaign_id', 'active')})")
    print(f"  Runs:     {len(runs)}")
    print(f"  Complete: {by_status['complete']}")
    print(f"  Pending:  {by_status['pending']}")
    print(f"  Failed:   {by_status['failed']}")
    print(f"  Models:   {campaign['models']}")
    print(f"  Modes:    {campaign['modes']}")
    print(f"  Tasks:    {sum(task_counts.values())} ({task_counts})")
    print()

    # Group by type for display
    solo_runs = {k: v for k, v in runs.items() if v.get("type") == "solo"}
    matchup_runs = {k: v for k, v in runs.items() if v.get("type") == "matchup"}

    def _status_label(key: str, r: Dict[str, Any]) -> str:
        status = r["status"].upper()
        suffix = ""
        if r["status"] in ("complete", "failed"):
            new_ids, stale_ids = _sync_run(key, r, tasks_dir)
            parts = []
            if new_ids:
                parts.append(f"+{len(new_ids)} new")
            if stale_ids:
                parts.append(f"-{len(stale_ids)} stale")
            if parts:
                suffix = f" ({', '.join(parts)})"
        completed = r.get("completed_tasks", [])
        count = f" [{len(completed)} tasks]" if completed else ""
        pad = " " * max(0, 10 - len(status))
        return f"[{status}]{pad} {key}{count}{suffix}"

    if solo_runs:
        print("Solo runs (cooperative + mixed):")
        for key in sorted(solo_runs.keys()):
            print(f"  {_status_label(key, solo_runs[key])}")

    if matchup_runs:
        print("\nCompetitive matchups:")
        for key in sorted(matchup_runs.keys()):
            print(f"  {_status_label(key, matchup_runs[key])}")

    archives = _list_archived_campaigns()
    if archives:
        print(f"\nArchived campaigns: {len(archives)}")


def cmd_list(args: argparse.Namespace) -> None:
    """List the active and archived campaigns."""
    active = _load_campaign(CAMPAIGN_FILE)
    if active:
        print("Active:")
        print(
            f"  {active.get('campaign_id', 'active')}: "
            f"{active.get('label', 'Active Campaign')} "
            f"({len(active.get('runs', {}))} runs, {active.get('task_total', 0)} tasks)"
        )
    else:
        print("Active:\n  <none>")

    archives = _list_archived_campaigns()
    print("Archived:")
    if not archives:
        print("  <none>")
        return
    for archive in archives:
        reason = f" — {archive['archive_reason']}" if archive.get("archive_reason") else ""
        print(
            f"  {archive['campaign_id']}: {archive['label']} "
            f"({archive.get('task_total', 0)} tasks, archived {archive.get('archived_at', 'unknown')})"
            f"{reason}"
        )


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


def _category_literal_metrics(
    results: List[Dict[str, Any]],
    category: Optional[str] = None,
) -> Dict[str, Any]:
    scoped = [
        result for result in results
        if category is None or result.get("category") == category
    ]
    return _literal_tom_metrics(scoped)


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
        category_literal = _category_literal_metrics(summary.get("results", []), category)
        solo_results[model_mode]["categories"][category] = {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "pass_rate": summary.get("pass_rate", 0),
            **category_literal,
        }

    # Compute overall pass rates
    for model_mode, data in solo_results.items():
        total = sum(c["total"] for c in data["categories"].values())
        passed = sum(c["passed"] for c in data["categories"].values())
        supported_probe_count = sum(
            c.get("literal_tom_supported_probe_count", 0)
            for c in data["categories"].values()
        )
        passed_probe_count = sum(
            c.get("literal_tom_passed_probe_count", 0)
            for c in data["categories"].values()
        )
        task_count = sum(
            c.get("literal_tom_task_count", 0)
            for c in data["categories"].values()
        )
        literal_score = (
            passed_probe_count / supported_probe_count * 100
            if supported_probe_count > 0
            else None
        )
        data["overall"] = {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "literal_tom_score": round(literal_score, 1) if literal_score is not None else None,
            "literal_tom_task_count": task_count,
            "literal_tom_supported_probe_count": supported_probe_count,
            "literal_tom_passed_probe_count": passed_probe_count,
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
        "generated_at": _now_iso(),
        "campaign_id": campaign.get("campaign_id", "active"),
        "label": campaign.get("label", "Active Campaign"),
        "status": campaign.get("status", "active"),
        "models": campaign["models"],
        "modes": campaign["modes"],
        "solo": solo_results,
        "matchups": matchup_results,
    }

    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2)

    # Print leaderboard
    print(f"\n{'=' * 70}")
    print("LEADERBOARD")
    print(f"{'=' * 70}")

    if solo_results:
        cats = ["cooperative", "mixed"]
        header = f"{'Model':<30} {'Pass':>8} {'LitToM':>8}"
        for cat in cats:
            header += f" {cat[:5]:>8}"
        print(header)
        print("-" * len(header))

        for model_mode in sorted(solo_results.keys()):
            data = solo_results[model_mode]
            overall = data["overall"]
            literal = overall.get("literal_tom_score")
            literal_text = f"{literal:>7.1f}%" if literal is not None else "     -- "
            row = f"{model_mode:<30} {overall['pass_rate']:>7.1f}% {literal_text}"
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

    print(f"\nSaved to: {LEADERBOARD_FILE}")


def cmd_archive(args: argparse.Namespace) -> None:
    """Archive the active campaign and clear active results."""
    if not CAMPAIGN_FILE.exists():
        print("No active campaign found.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    campaign_id = args.campaign_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = _archive_campaign_dir(campaign_id)
    if archive_dir.exists():
        print(f"Archive already exists: {archive_dir}")
        sys.exit(1)

    archive_dir.mkdir(parents=True, exist_ok=False)

    if not LEADERBOARD_FILE.exists():
        _generate_report(campaign)

    archived_campaign = dict(campaign)
    archived_campaign["campaign_id"] = campaign_id
    archived_campaign["label"] = args.label or campaign.get("label") or campaign_id
    archived_campaign["status"] = "archived"
    archived_campaign["archived_at"] = _now_iso()
    archived_campaign["archive_reason"] = args.reason or ""

    with open(archive_dir / "campaign.json", "w") as f:
        json.dump(archived_campaign, f, indent=2)

    if LEADERBOARD_FILE.exists():
        shutil.move(str(LEADERBOARD_FILE), str(archive_dir / "leaderboard.json"))
    if RUNS_DIR.exists():
        shutil.move(str(RUNS_DIR), str(archive_dir / "runs"))
    CAMPAIGN_FILE.unlink(missing_ok=True)

    print(f"Archived active campaign to: {archive_dir}")
    print(f"  Label:  {archived_campaign['label']}")
    if archived_campaign["archive_reason"]:
        print(f"  Reason: {archived_campaign['archive_reason']}")


def cmd_remove(args: argparse.Namespace) -> None:
    """Remove models from the campaign, including their runs and results."""
    if not CAMPAIGN_FILE.exists():
        print("No campaign found.")
        sys.exit(1)

    with open(CAMPAIGN_FILE) as f:
        campaign = json.load(f)

    to_remove = set(args.models)
    current_models = campaign.get("models", [])
    unknown = to_remove - set(current_models)
    if unknown:
        print(f"Models not in campaign: {sorted(unknown)}")
        print(f"Current models: {current_models}")
        sys.exit(1)

    # Find runs to delete
    runs = campaign["runs"]
    keys_to_delete = []
    for key, run_def in runs.items():
        run_models = {run_def.get("model"), run_def.get("model_a"), run_def.get("model_b")} - {None}
        if run_models & to_remove:
            keys_to_delete.append(key)

    # Delete run results from disk
    for key in keys_to_delete:
        results_path = RUNS_DIR / key
        if results_path.exists():
            shutil.rmtree(str(results_path))
        del runs[key]

    # Update models and matchups
    remaining = [m for m in current_models if m not in to_remove]
    campaign["models"] = remaining
    campaign["competitive_matchups"] = [list(m) for m in combinations(remaining, 2)]
    campaign["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(CAMPAIGN_FILE, "w") as f:
        json.dump(campaign, f, indent=2)

    print(f"Removed {sorted(to_remove)}: deleted {len(keys_to_delete)} runs")
    print(f"  Remaining models: {remaining}")
    print(f"  Remaining runs: {len(runs)}")


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

    p_archive = sub.add_parser("archive", help="Archive the active campaign and clear active results")
    p_archive.add_argument("--campaign-id", default=None, help="Archive id (default: timestamp)")
    p_archive.add_argument("--label", default=None, help="Human-readable label for the archive")
    p_archive.add_argument("--reason", default="", help="Why this campaign was archived")

    sub.add_parser("list", help="List active and archived campaigns")

    # remove
    p_rm = sub.add_parser("remove", help="Remove models from the campaign")
    p_rm.add_argument("--models", nargs="+", required=True, help="Models to remove")

    # run
    p_run = sub.add_parser("run", help="Run pending benchmarks")
    p_run.add_argument("--only", default=None, help="Run only this specific run key")
    p_run.add_argument("--models", nargs="+", default=None, help="Only run benchmarks involving these models")
    p_run.add_argument("--max-workers", type=int, default=50, help="Parallel workers per run")

    # status
    sub.add_parser("status", help="Show campaign status")

    # report
    sub.add_parser("report", help="Generate leaderboard from completed runs")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "archive":
        cmd_archive(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
