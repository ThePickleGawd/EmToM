"""Print a compact summary table of all tasks in a directory.

Usage:
    python -m emtom.utils.task_summary [--stats] [tasks_dir]
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from emtom.evolve.benchmark_wrapper import (
    _migrate_legacy_calibration,
    cal_passed,
    cal_progress,
)

USE_COLOR = sys.stdout.isatty()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


def _style(text: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


def _pct_color(value: float) -> str:
    if value >= 80:
        return GREEN
    if value >= 50:
        return YELLOW
    return RED


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = int(minutes // 60)
    remaining_min = int(minutes % 60)
    return f"{hours}h {remaining_min}m"


def _fmt_steps(value: float) -> str:
    return f"{value:.1f} steps" if value != int(value) else f"{int(value)} steps"


def _parse_tested_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_tasks(tasks_dir: str) -> list:
    """Load all task JSONs, normalizing calibration to current format."""
    tasks_path = Path(tasks_dir)
    if not tasks_path.exists():
        print(f"Directory not found: {tasks_dir}")
        sys.exit(1)

    tasks = []
    for f in sorted(tasks_path.glob("*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        d["calibration"] = _migrate_legacy_calibration(d.get("calibration", []))
        tasks.append(d)
    return tasks


def _model_label(agent_models: dict) -> str:
    """Derive a short label from an agent_models dict."""
    unique = sorted(set(agent_models.values())) if agent_models else ["?"]
    return " vs ".join(unique)


def _matchup_label(entry: dict, task: dict) -> str:
    """Derive a directed matchup label like 'gpt-5.2 vs sonnet' (team_0 vs team_1)."""
    am = entry.get("agent_models", {})
    teams = task.get("teams", {})
    if not am or not teams:
        return _model_label(am)

    team_models = {}
    for team_id in sorted(teams):
        agents = teams[team_id]
        models = set(am.get(a, "?") for a in agents)
        team_models[team_id] = models.pop() if len(models) == 1 else "mixed"

    unique = list(dict.fromkeys(team_models.values()))
    if len(unique) == 1:
        return unique[0]
    return " vs ".join(team_models[t] for t in sorted(team_models))


def _resolve_team_models(entry: dict, task: dict) -> dict:
    """Map team_id -> model name from agent_models + teams."""
    am = entry.get("agent_models", {})
    teams = task.get("teams", {})
    tm = {}
    for team_id in sorted(teams):
        agents = teams[team_id]
        models = set(am.get(a, "?") for a in agents)
        tm[team_id] = models.pop() if len(models) == 1 else "mixed"
    return tm


def aggregate_tasks(tasks_dir: str) -> None:
    tasks = _load_tasks(tasks_dir)
    if not tasks:
        print("No tasks found.")
        return

    by_cat = defaultdict(list)
    for t in tasks:
        by_cat[t.get("category", "?")].append(t)

    # ── Gather per-model stats for each category ──

    # Cooperative: model -> {pass, total, pcts}
    coop = defaultdict(lambda: {"pass": 0, "total": 0, "pcts": []})
    for t in by_cat.get("cooperative", []):
        for entry in t.get("calibration", []):
            am = entry.get("agent_models", {})
            models = set(am.values()) if am else set()
            model = models.pop() if len(models) == 1 else _model_label(am)
            coop[model]["total"] += 1
            coop[model]["pcts"].append(cal_progress(entry))
            if cal_passed(entry):
                coop[model]["pass"] += 1

    # Competitive: h2h matrix + progress
    h2h = defaultdict(lambda: {"wins": 0, "games": 0})
    comp_progress = defaultdict(list)
    n_cross = 0
    for t in by_cat.get("competitive", []):
        teams = t.get("teams", {})
        for entry in t.get("calibration", []):
            am = entry.get("agent_models", {})
            if len(set(am.values())) <= 1 if am else True:
                continue
            n_cross += 1
            tm = _resolve_team_models(entry, t)
            results = entry.get("results", {})
            winner_team = results.get("winner")
            winner_model = None
            if winner_team and teams:
                wa = teams.get(winner_team, [])
                wm = set(am.get(a, "?") for a in wa)
                winner_model = wm.pop() if len(wm) == 1 else None
            if len(tm) == 2:
                t0, t1 = sorted(tm)
                m0, m1 = tm[t0], tm[t1]
                h2h[(m0, m1)]["games"] += 1
                h2h[(m1, m0)]["games"] += 1
                if winner_model == m0:
                    h2h[(m0, m1)]["wins"] += 1
                elif winner_model == m1:
                    h2h[(m1, m0)]["wins"] += 1
            for team_id, team_data in results.get("teams", {}).items():
                prog = team_data.get("progress", 0.0)
                comp_progress[tm.get(team_id, "?")].append(prog)

    # Mixed: main goal pass + per-agent subgoal, both keyed by model
    mixed_main = defaultdict(lambda: {"pass": 0, "total": 0})
    mixed_subgoal = defaultdict(lambda: {"pass": 0, "total": 0})
    for t in by_cat.get("mixed", []):
        for entry in t.get("calibration", []):
            am = entry.get("agent_models", {})
            results = entry.get("results", {})
            run_models = set(am.values()) if am else set()

            # Main goal pass (one per run)
            main_passed = results.get("main_goal", {}).get("passed", False)
            for model in run_models:
                mixed_main[model]["total"] += 1
                if main_passed:
                    mixed_main[model]["pass"] += 1

            # Per-agent subgoal
            for agent_id, agent_res in results.get("agents", {}).items():
                model = am.get(agent_id, "?")
                mixed_subgoal[model]["total"] += 1
                if agent_res.get("subgoal_passed"):
                    mixed_subgoal[model]["pass"] += 1

    all_models = sorted(
        set(coop) | set(m for p in h2h for m in p)
        | set(mixed_subgoal) | set(mixed_main)
    )
    comp_models = sorted(set(m for p in h2h for m in p))
    benchmark_entries = [entry for t in tasks for entry in t.get("calibration", [])]

    # ── Print ──
    def _pct(n, d):
        return f"{n/d*100:.1f}" if d else "—"

    cat_short = {"cooperative": "coop", "competitive": "comp", "mixed": "mixed"}
    cat_counts = ", ".join(
        f"{len(by_cat.get(c, []))} {cat_short.get(c, c)}"
        for c in ["cooperative", "competitive", "mixed"]
        if by_cat.get(c)
    )
    n_coop = len(by_cat.get("cooperative", []))
    n_comp = len(by_cat.get("competitive", []))
    n_mixed = len(by_cat.get("mixed", []))

    W = 60

    print()
    print(_style("═" * W, BOLD, CYAN))
    print(_style(f"{'EMTOM BENCHMARK RESULTS':^{W}}", BOLD, WHITE))
    print(_style("═" * W, BOLD, CYAN))
    print(f"  {_style(str(len(tasks)), BOLD, GREEN)} tasks ({_style(cat_counts, CYAN)})")

    # ── Cooperative ──
    if n_coop:
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style(f'Cooperative (n={n_coop})', BOLD, BLUE)} {_style('—', DIM)} {_style('Task Pass Rate', BOLD)}")
        print(_style("─" * W, CYAN))
        print("  " + _style(f"{'Model':<16} {'Pass/N':>8} {'Rate':>7} {'Avg%':>7}", BOLD))
        print("  " + _style(f"{'─'*16} {'─'*8} {'─'*7} {'─'*7}", DIM))
        for m in sorted(coop, key=lambda m: -coop[m]["pass"]):
            s = coop[m]
            avg = sum(s["pcts"]) / len(s["pcts"]) * 100 if s["pcts"] else 0
            rate = s["pass"] / s["total"] * 100 if s["total"] else 0
            pass_ratio = f"{s['pass']:>3}/{s['total']:<4}"
            pass_rate_str = f"{_pct(s['pass'], s['total']):>6}%"
            avg_str = f"{avg:>6.1f}%"
            print(
                f"  {_style(f'{m:<16}', WHITE)} "
                f"{_style(pass_ratio, BOLD, GREEN if s['pass'] else DIM)} "
                f"{_style(pass_rate_str, _pct_color(rate))} "
                f"{_style(avg_str, _pct_color(avg))}"
            )

    # ── Competitive ──
    if n_comp:
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style(f'Competitive (n={n_comp})', BOLD, MAGENTA)} {_style('—', DIM)} {_style('Head-to-Head Win Rate', BOLD)}")
        print(_style("─" * W, CYAN))
        if len(comp_models) > 1:
            col_w = max(len(m) for m in comp_models) + 1
            col_w = max(col_w, 9)
            print(_style(f"  {'':<{col_w}}" + "".join(f"{m:>{col_w}}" for m in comp_models), BOLD))
            print(_style(f"  {'─'*col_w}" + "".join(f" {'─'*(col_w-1)}" for _ in comp_models), DIM))
            for row_m in comp_models:
                cells = []
                for col_m in comp_models:
                    if row_m == col_m:
                        cells.append(f"{'—':>{col_w}}")
                    else:
                        rec = h2h.get((row_m, col_m))
                        if rec and rec["games"]:
                            pct = rec["wins"] / rec["games"] * 100
                            cells.append(_style(f"{pct:.1f}%".rjust(col_w), _pct_color(pct)))
                        else:
                            cells.append(f"{'—':>{col_w}}")
                print(f"  {_style(f'{row_m:<{col_w}}', WHITE)}" + "".join(cells))
            if comp_progress:
                print()
                print(f"  {_style('Avg progress:', BOLD)} " + ", ".join(
                    f"{_style(m, WHITE)} {_style(f'{sum(v)/len(v)*100:.1f}%', _pct_color(sum(v)/len(v)*100))}"
                    for m, v in sorted(comp_progress.items(),
                                       key=lambda x: -sum(x[1])/len(x[1]))
                ))
        else:
            print(f"  {_style('No cross-model matchups yet.', DIM)}")

    # ── Mixed ──
    if n_mixed:
        mixed_models = sorted(set(mixed_main) | set(mixed_subgoal))
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style(f'Mixed (n={n_mixed})', BOLD, YELLOW)} {_style('—', DIM)} {_style('Main Goal + Private Subgoal Rate', BOLD)}")
        print(_style("─" * W, CYAN))
        if mixed_models and (mixed_main or mixed_subgoal):
            print(_style(f"  {'Model':<16} {'Main Goal':>12} {'Subgoal':>14}", BOLD))
            print(_style(f"  {'─'*16} {'─'*12} {'─'*14}", DIM))
            for m in sorted(mixed_models, key=lambda m: -(mixed_subgoal[m]["pass"] / mixed_subgoal[m]["total"] if mixed_subgoal[m]["total"] else 0)):
                tp = mixed_main[m]
                sg = mixed_subgoal[m]
                tp_str = f"{tp['pass']}/{tp['total']} ({_pct(tp['pass'], tp['total'])}%)" if tp["total"] else "—"
                sg_str = f"{sg['pass']}/{sg['total']} ({_pct(sg['pass'], sg['total'])}%)" if sg["total"] else "—"
                main_rate = tp["pass"] / tp["total"] * 100 if tp["total"] else 0
                sub_rate = sg["pass"] / sg["total"] * 100 if sg["total"] else 0
                print(
                    f"  {_style(f'{m:<16}', WHITE)} "
                    f"{_style(f'{tp_str:>12}', _pct_color(main_rate) if tp['total'] else DIM)} "
                    f"{_style(f'{sg_str:>14}', _pct_color(sub_rate) if sg['total'] else DIM)}"
                )
        else:
            print(f"  {_style('No subgoal data yet (needs re-benchmark).', DIM)}")

    # ── Benchmark Time ──
    step_values = [entry.get("steps", 0) for entry in benchmark_entries if isinstance(entry.get("steps"), int)]
    timestamp_values = sorted(
        dt for dt in (_parse_tested_at(entry.get("tested_at")) for entry in benchmark_entries) if dt is not None
    )
    if step_values or timestamp_values:
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style('Benchmark Time', BOLD, GREEN)}")
        print(_style("─" * W, CYAN))
        if step_values:
            sorted_steps = sorted(step_values)
            avg_steps = sum(sorted_steps) / len(sorted_steps)
            median_steps = sorted_steps[len(sorted_steps) // 2]
            print(f"  {_style('Benchmarked runs', BOLD):<20} {_style(str(len(step_values)), WHITE)}")
            print(f"  {_style('Avg task runtime', BOLD):<20} {_style(_fmt_steps(avg_steps), CYAN)}")
            print(f"  {_style('Median runtime', BOLD):<20} {_style(_fmt_steps(median_steps), CYAN)}")
            print(f"  {_style('Fastest task', BOLD):<20} {_style(_fmt_steps(sorted_steps[0]), GREEN)}")
            print(f"  {_style('Slowest task', BOLD):<20} {_style(_fmt_steps(sorted_steps[-1]), YELLOW)}")
        if len(timestamp_values) >= 2:
            span_seconds = max((timestamp_values[-1] - timestamp_values[0]).total_seconds(), 0.0)
            if span_seconds <= 72 * 3600:
                print(f"  {_style('Logged wall-clock span', BOLD):<20} {_style(_fmt_duration(span_seconds), MAGENTA)}")
            else:
                print(f"  {_style('Logged wall-clock span', BOLD):<20} {_style('mixed benchmark dates', DIM)}")

    # ── ToM Order Distribution ──
    tom_counts = defaultdict(int)
    for t in tasks:
        tom_counts[t.get("tom_level")] += 1
    labeled = {k: v for k, v in tom_counts.items() if k is not None}
    if labeled:
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style('ToM Order Distribution', BOLD)}")
        print(_style("─" * W, CYAN))
        print(_style(f"  {'Order':<10} {'Tasks':>6} {'%':>7}", BOLD))
        print(_style(f"  {'─'*10} {'─'*6} {'─'*7}", DIM))
        for level in sorted(labeled):
            pct = labeled[level] / len(tasks) * 100
            print(
                f"  {_style(f'{level:<10}', WHITE)} "
                f"{_style(f'{labeled[level]:>6}', CYAN)} "
                f"{_style(f'{pct:>6.1f}%', _pct_color(pct))}"
            )
        unlabeled = tom_counts.get(None, 0)
        if unlabeled:
            unlabeled_label = f"{'unlabeled':<10}"
            unlabeled_count = f"{unlabeled:>6}"
            print(f"  {_style(unlabeled_label, DIM)} {_style(unlabeled_count, DIM)}")

    # ── ToM Level Pass Rate (standard mode, per-task) ──
    tom_pass = defaultdict(lambda: {"pass": 0, "tested": 0, "tasks": 0})
    for t in tasks:
        level = t.get("tom_level")
        if level is None:
            continue
        tom_pass[level]["tasks"] += 1
        std_entries = [e for e in t.get("calibration", [])
                       if e.get("run_mode") == "standard"]
        if std_entries:
            tom_pass[level]["tested"] += 1
            if any(cal_passed(e) for e in std_entries):
                tom_pass[level]["pass"] += 1
    if tom_pass:
        print(f"\n{_style('─' * W, CYAN)}")
        print(f"  {_style('ToM Level Pass Rate (standard mode)', BOLD)}")
        print(_style("─" * W, CYAN))
        print(_style(f"  {'Level':<8} {'Tasks':>6} {'Tested':>8} {'Passed':>8} {'Pass Rate':>10}", BOLD))
        print(_style(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*10}", DIM))
        for level in sorted(tom_pass):
            s = tom_pass[level]
            pass_rate = s["pass"] / s["tested"] * 100 if s["tested"] else 0
            task_str = f"{s['tasks']:>6}"
            tested_str = f"{s['tested']:>8}"
            passed_str = f"{s['pass']:>8}"
            pass_rate_str = f"{_pct(s['pass'], s['tested']):>9}%"
            print(
                f"  {_style(f'K={level:<6}', WHITE)} "
                f"{_style(task_str, CYAN)} "
                f"{_style(tested_str, CYAN)} "
                f"{_style(passed_str, GREEN if s['pass'] else DIM)} "
                f"{_style(pass_rate_str, _pct_color(pass_rate) if s['tested'] else DIM)}"
            )

    print()


def _cal_summary(entry: dict, task: dict) -> str:
    """One-line calibration summary for the per-task table."""
    label = _model_label(entry.get("agent_models", {}))
    results = entry.get("results", {})
    category = task.get("category", "")

    if category == "competitive":
        winner = results.get("winner")
        if winner:
            teams = task.get("teams", {})
            am = entry.get("agent_models", {})
            wa = teams.get(winner, [])
            wm = set(am.get(a, "?") for a in wa)
            winner_model = wm.pop() if len(wm) == 1 else winner
            return f"{label}: {winner_model} wins"
        return f"{label}: draw"

    if category == "mixed":
        main = results.get("main_goal", {})
        agents = results.get("agents", {})
        main_str = "main:Y" if main.get("passed") else f"main:{main.get('progress', 0):.0%}"
        agent_parts = []
        for aid in sorted(agents):
            agent_parts.append("Y" if agents[aid].get("subgoal_passed") else "N")
        sub_str = ",".join(agent_parts) if agent_parts else "?"
        return f"{label}: {main_str} sub:[{sub_str}]"

    # Cooperative
    if results.get("passed"):
        return f"{label}: PASS"
    return f"{label}: {results.get('progress', 0):.0%}"


def summarize_tasks(tasks_dir: str) -> None:
    all_tasks = _load_tasks(tasks_dir)

    rows = []
    for d in all_tasks:
        category = d.get("category", "?")
        num_agents = d.get("num_agents", "?")
        title = d.get("title", "Untitled")
        mechanics = d.get("active_mechanics", [])

        cal_parts = [_cal_summary(entry, d) for entry in d.get("calibration", [])]
        cal_str = " | ".join(cal_parts) if cal_parts else "-"

        cat_short = {"cooperative": "coop", "competitive": "comp", "mixed": "mixed"}.get(
            category, category[:5]
        )
        rows.append({
            "title": title,
            "cat": cat_short,
            "agents": num_agents,
            "mechanics": len(mechanics),
            "cal": cal_str,
        })

    if not rows:
        print("No tasks found.")
        return

    # Aggregate stats
    cats = {}
    pass_count = 0
    for r in rows:
        cats[r["cat"]] = cats.get(r["cat"], 0) + 1
        if "PASS" in r["cal"] or "main:Y" in r["cal"] or "wins" in r["cal"]:
            pass_count += 1

    # Print header
    print(f"\n{'#':>3}  {'Cat':5}  {'Ag':>2}  {'Mech':>4}  {'Title':<50}  Calibration")
    print(f"{'─'*3}  {'─'*5}  {'─'*2}  {'─'*4}  {'─'*50}  {'─'*40}")

    for i, r in enumerate(rows, 1):
        title = r["title"][:50]
        print(f"{i:3}  {r['cat']:5}  {r['agents']:>2}  {r['mechanics']:>4}  {title:<50}  {r['cal']}")

    # Footer
    print(f"\n{'─'*120}")
    cat_str = ", ".join(f"{v} {k}" for k, v in sorted(cats.items()))
    print(f"Total: {len(rows)} tasks ({cat_str})  |  {pass_count} passing at least one calibration")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    tasks_dir = args[0] if args else "data/emtom/tasks"

    if "--stats" in flags:
        aggregate_tasks(tasks_dir)
    else:
        summarize_tasks(tasks_dir)


if __name__ == "__main__":
    main()
