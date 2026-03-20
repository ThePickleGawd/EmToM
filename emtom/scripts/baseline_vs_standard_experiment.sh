#!/bin/bash
# Baseline vs Standard experiment on newly generated tasks
# Runs standard mode first, then baseline on failures only
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

TASK_DIR="data/emtom/tasks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="outputs/baseline_vs_standard_${TIMESTAMP}"
STANDARD_DIR="${OUTPUT_BASE}/standard"
BASELINE_DIR="${OUTPUT_BASE}/baseline"
EXPERIMENT_TASKS="${OUTPUT_BASE}/experiment_tasks"

mkdir -p "$STANDARD_DIR" "$BASELINE_DIR" "$EXPERIMENT_TASKS"

# Find genuinely new tasks: compare current tasks against pre-run snapshot
# The pre-run task list is stored by the monitor at /tmp/emtom_monitor_state.json
echo "=== Collecting genuinely new tasks from this bulk gen run ==="

python3 - <<'PYEOF' "$TASK_DIR" "$EXPERIMENT_TASKS"
import json, shutil, sys
from pathlib import Path

task_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

# Load the monitor's initial snapshot (set before bulk gen produced tasks)
state_file = Path("/tmp/emtom_monitor_state.json")
if state_file.exists():
    state = json.loads(state_file.read_text())
    seen = set(state.get("seen_tasks", []))
else:
    seen = set()

# Find tasks not in the pre-run snapshot
new_tasks = []
for f in sorted(task_dir.glob("*.json")):
    if f.name not in seen:
        shutil.copy2(f, out_dir / f.name)
        new_tasks.append(f.name)

print(f"Pre-run snapshot: {len(seen)} tasks")
print(f"Current total: {len(list(task_dir.glob('*.json')))}")
print(f"New tasks from this run: {len(new_tasks)}")
for t in new_tasks:
    print(f"  {t}")
PYEOF

TASK_COUNT=$(ls "$EXPERIMENT_TASKS"/*.json 2>/dev/null | wc -l)

if [ "$TASK_COUNT" -eq 0 ]; then
    echo "No new tasks to benchmark yet. Exiting."
    exit 0
fi

echo ""
echo "=== Phase 1: Standard mode benchmark ($TASK_COUNT tasks) ==="

./emtom/run_emtom.sh benchmark \
    --tasks-dir "$EXPERIMENT_TASKS" \
    --run-mode standard \
    --output-dir "$STANDARD_DIR" \
    --max-workers 4 \
    --num-gpus 8 \
    --no-video

echo ""
echo "=== Phase 1 complete. Analyzing results ==="

# Collect failures and build phase 2
python3 - <<'PYEOF' "$STANDARD_DIR" "$EXPERIMENT_TASKS" "$OUTPUT_BASE"
import json, os, shutil
from pathlib import Path
import sys

standard_dir = Path(sys.argv[1])
experiment_tasks = Path(sys.argv[2])
output_base = Path(sys.argv[3])

# Collect results from standard benchmark
passed = []
failed = []

for run_dir in sorted(standard_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    for rf in run_dir.rglob("result.json"):
        try:
            data = json.loads(rf.read_text())
            task_file = data.get("task_file", "")
            task_id = data.get("task_id", run_dir.name)
            success = data.get("success", False) or data.get("functional_success", False)
            entry = {"task_id": task_id, "task_file": task_file, "dir": str(run_dir), "result": data}
            if success:
                passed.append(entry)
            else:
                failed.append(entry)
        except Exception as e:
            print(f"  Warning: could not parse {rf}: {e}")

# Copy failed tasks for phase 2
failed_dir = output_base / "failed_tasks"
failed_dir.mkdir(exist_ok=True)
for ft in failed:
    tf = ft["task_file"]
    if tf and os.path.exists(tf):
        shutil.copy2(tf, failed_dir)
    else:
        # Try matching by task_id prefix
        for src in experiment_tasks.glob("*.json"):
            if ft["task_id"] in src.name:
                shutil.copy2(src, failed_dir)
                break

# Save phase 1 report
report = {
    "total": len(passed) + len(failed),
    "passed": len(passed),
    "failed": len(failed),
    "pass_rate": f"{100 * len(passed) / max(1, len(passed) + len(failed)):.1f}%",
    "passed_tasks": [p["task_id"] for p in passed],
    "failed_tasks": [f["task_id"] for f in failed],
}
(output_base / "phase1_report.json").write_text(json.dumps(report, indent=2))

print(f"\n  Standard mode results:")
print(f"    Passed: {len(passed)}/{len(passed) + len(failed)} ({report['pass_rate']})")
print(f"    Failed: {len(failed)}/{len(passed) + len(failed)}")
PYEOF

FAILED_COUNT=$(ls "$OUTPUT_BASE/failed_tasks"/*.json 2>/dev/null | wc -l)

if [ "$FAILED_COUNT" -eq 0 ]; then
    echo ""
    echo "All tasks passed in standard mode! No baseline comparison needed."
    exit 0
fi

# Phase 2: Baseline mode on failures
echo ""
echo "=== Phase 2: Baseline mode on $FAILED_COUNT failed tasks ==="

./emtom/run_emtom.sh benchmark \
    --tasks-dir "$OUTPUT_BASE/failed_tasks" \
    --run-mode baseline \
    --output-dir "$BASELINE_DIR" \
    --max-workers 4 \
    --num-gpus 8 \
    --no-video

echo ""
echo "=== Generating comparison report ==="

python3 - <<'PYEOF' "$OUTPUT_BASE" "$STANDARD_DIR" "$BASELINE_DIR"
import json
from pathlib import Path
import sys

output_base = Path(sys.argv[1])
standard_dir = Path(sys.argv[2])
baseline_dir = Path(sys.argv[3])

def collect_results(run_dir):
    results = {}
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        for rf in d.rglob("result.json"):
            try:
                data = json.loads(rf.read_text())
                task_id = data.get("task_id", d.name)
                results[task_id] = {
                    "success": data.get("success", False) or data.get("functional_success", False),
                    "score": data.get("score", data.get("functional_score", 0)),
                    "literal_tom": data.get("literal_tom_probe_score"),
                }
            except Exception:
                pass
    return results

phase1 = json.loads((output_base / "phase1_report.json").read_text())
standard = collect_results(standard_dir)
baseline = collect_results(baseline_dir)

baseline_passed = sum(1 for r in baseline.values() if r["success"])
baseline_total = len(baseline)

print()
print("=" * 70)
print("  BASELINE vs STANDARD: ISOLATING ToM AS FAILURE MODE")
print("=" * 70)
print()
print(f"  Standard mode:  {phase1['passed']}/{phase1['total']} passed ({phase1['pass_rate']})")
print(f"  Baseline mode:  {baseline_passed}/{baseline_total} passed on standard-mode failures")
print()

if baseline_total > 0:
    rescue_rate = 100 * baseline_passed / baseline_total
    print(f"  RESCUE RATE: {rescue_rate:.1f}%")
    print(f"  (% of standard failures that pass when info asymmetry is removed)")
    print()

    if rescue_rate > 50:
        verdict = "YES — ToM/information asymmetry IS the key bottleneck"
        recommendation = "Add baseline calibration during task generation to ensure tasks are solvable with full info before testing ToM"
    elif rescue_rate > 20:
        verdict = "PARTIAL — ToM contributes but other factors also matter"
        recommendation = "Consider dual-mode generation: verify tasks pass baseline, then verify ToM adds meaningful difficulty"
    else:
        verdict = "NO — failures persist even with full information"
        recommendation = "Focus on task design quality (agent_necessity, mechanic_utilization) before adding ToM complexity"

    print(f"  VERDICT: {verdict}")
    print(f"  RECOMMENDATION: {recommendation}")
else:
    rescue_rate = 0

# Per-task breakdown
print()
print(f"  {'TASK ID':<55} {'STD':>6} {'BASE':>6} {'LIT_TOM':>8}")
print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*8}")
all_ids = sorted(set(list(standard.keys()) + list(baseline.keys())))
for tid in all_ids:
    short = tid[:55]
    std = standard.get(tid, {})
    base = baseline.get(tid, {})
    std_s = "PASS" if std.get("success") else "FAIL" if tid in standard else "N/A"
    base_s = "PASS" if base.get("success") else "FAIL" if tid in baseline else "N/A"
    lit = base.get("literal_tom")
    lit_s = f"{lit:.2f}" if lit is not None else "N/A"
    print(f"  {short:<55} {std_s:>6} {base_s:>6} {lit_s:>8}")

# Save final report
final = {
    "standard_total": phase1["total"],
    "standard_passed": phase1["passed"],
    "baseline_total": baseline_total,
    "baseline_passed": baseline_passed,
    "rescue_rate": f"{rescue_rate:.1f}%",
    "per_task": {
        tid: {
            "standard": standard.get(tid, {}),
            "baseline": baseline.get(tid, {}),
        }
        for tid in all_ids
    },
}
(output_base / "comparison_report.json").write_text(json.dumps(final, indent=2))
print()
print(f"  Report saved: {output_base}/comparison_report.json")
print("=" * 70)
PYEOF

echo ""
echo "=== Experiment complete ==="
echo "Output: $OUTPUT_BASE"
