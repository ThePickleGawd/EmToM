"""Print a compact summary table of all tasks in a directory.

Usage:
    python -m emtom.utils.task_summary [--stats] [tasks_dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_tasks(tasks_dir: str) -> list:
    """Load all task JSONs, normalizing calibration to list format."""
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
        cal = d.get("calibration", [])
        if isinstance(cal, dict):
            cal = [{**v, "_legacy_key": k} for k, v in cal.items() if isinstance(v, dict)]
            d["calibration"] = cal
        tasks.append(d)
    return tasks


def _model_label(agent_models: dict) -> str:
    """Derive a short label from an agent_models dict."""
    unique = sorted(set(agent_models.values())) if agent_models else ["?"]
    return " vs ".join(unique)


def _matchup_label(entry: dict, task: dict) -> str:
    """Derive a directed matchup label like 'gpt-5.2 vs sonnet' (team_0 vs team_1).

    For same-model runs returns just the model name.
    """
    am = entry.get("agent_models", {})
    teams = task.get("teams", {})
    if not am or not teams:
        return _model_label(am)

    team_models = {}
    for team_id in sorted(teams):
        agents = teams[team_id]
        models = set(am.get(a, "?") for a in agents)
        team_models[team_id] = models.pop() if len(models) == 1 else "mixed"

    unique = list(dict.fromkeys(team_models.values()))  # preserve order, dedup
    if len(unique) == 1:
        return unique[0]
    return " vs ".join(team_models[t] for t in sorted(team_models))


def aggregate_tasks(tasks_dir: str) -> None:
    tasks = _load_tasks(tasks_dir)
    if not tasks:
        print("No tasks found.")
        return

    # ── Flatten: one record per (task, calibration entry) ──
    # Also keep per-task "best run" for task-level stats
    by_cat_tasks = defaultdict(list)   # category -> [task_dict]
    by_cat_runs = defaultdict(list)    # category -> [flat record]

    for t in tasks:
        category = t.get("category", "?")
        by_cat_tasks[category].append(t)
        teams = t.get("teams", {})
        for entry in t.get("calibration", []):
            am = entry.get("agent_models", {})
            matchup = _matchup_label(entry, t)

            # Resolve winner team_id to model name
            winner_model = None
            winner_team = entry.get("winner")
            if winner_team and teams:
                winner_agents = teams.get(winner_team, [])
                winner_models = set(am.get(a, "?") for a in winner_agents)
                winner_model = winner_models.pop() if len(winner_models) == 1 else None

            by_cat_runs[category].append({
                "task": t,
                "entry": entry,
                "matchup": matchup,
                "is_cross_model": len(set(am.values())) > 1 if am else False,
                "passed": entry.get("passed", False),
                "pct": entry.get("percent_complete", 0.0),
                "winner_model": winner_model,
            })

    W = 72
    SEP = "─" * W

    # ── Overview ──
    all_runs = [r for runs in by_cat_runs.values() for r in runs]
    total_pass = sum(1 for r in all_runs if r["passed"])
    avg_pct = sum(r["pct"] for r in all_runs) / len(all_runs) if all_runs else 0
    print(f"\n{'TASK DATASET STATS':^{W}}")
    print(SEP)
    print(f"  Tasks: {len(tasks)}   Calibration runs: {len(all_runs)}")
    cat_counts = ", ".join(
        f"{len(by_cat_tasks.get(c, []))} {c}" for c in ["cooperative", "competitive", "mixed"]
    )
    print(f"  Categories: {cat_counts}")
    print(f"  Overall: {total_pass}/{len(all_runs)} runs passed ({total_pass/len(all_runs)*100:.1f}%)   Avg completion: {avg_pct:.1%}")

    # ═══════════════════════════════════════════════════════
    # COOPERATIVE
    # ═══════════════════════════════════════════════════════
    coop_tasks = by_cat_tasks.get("cooperative", [])
    coop_runs = by_cat_runs.get("cooperative", [])
    if coop_tasks:
        print(f"\n{'═'*W}")
        print(f"  COOPERATIVE ({len(coop_tasks)} tasks, {len(coop_runs)} runs)")
        print(f"{'═'*W}")

        n_pass = sum(1 for r in coop_runs if r["passed"])
        avg = sum(r["pct"] for r in coop_runs) / len(coop_runs) if coop_runs else 0
        avg_steps = sum(r["entry"].get("steps", 0) for r in coop_runs) / len(coop_runs) if coop_runs else 0
        print(f"  Pass rate:      {n_pass}/{len(coop_runs)} ({n_pass/len(coop_runs)*100:.1f}%)")
        print(f"  Avg completion: {avg:.1%}")
        print(f"  Avg steps:      {avg_steps:.0f}")

        # By model
        by_model = defaultdict(list)
        for r in coop_runs:
            by_model[r["matchup"]].append(r)
        if len(by_model) > 1:
            print(f"\n  {'Model':<24} {'Runs':>5} {'Pass':>5} {'Rate':>7} {'Avg%':>7}")
            print(f"  {'─'*24} {'─'*5} {'─'*5} {'─'*7} {'─'*7}")
            for label in sorted(by_model, key=lambda k: -len(by_model[k])):
                recs = by_model[label]
                p = sum(1 for r in recs if r["passed"])
                a = sum(r["pct"] for r in recs) / len(recs)
                print(f"  {label:<24} {len(recs):>5} {p:>5} {p/len(recs)*100:>6.1f}% {a:>6.1%}")

        # By agent count
        by_agents = defaultdict(list)
        for r in coop_runs:
            by_agents[r["task"].get("num_agents", 0)].append(r)
        if len(by_agents) > 1:
            print(f"\n  {'Agents':>8} {'Runs':>5} {'Pass':>5} {'Rate':>7} {'Avg%':>7}")
            print(f"  {'─'*8} {'─'*5} {'─'*5} {'─'*7} {'─'*7}")
            for na in sorted(by_agents):
                recs = by_agents[na]
                p = sum(1 for r in recs if r["passed"])
                a = sum(r["pct"] for r in recs) / len(recs)
                print(f"  {na:>8} {len(recs):>5} {p:>5} {p/len(recs)*100:>6.1f}% {a:>6.1%}")

    # ═══════════════════════════════════════════════════════
    # COMPETITIVE
    # ═══════════════════════════════════════════════════════
    comp_tasks = by_cat_tasks.get("competitive", [])
    comp_runs = by_cat_runs.get("competitive", [])
    if comp_tasks:
        print(f"\n{'═'*W}")
        print(f"  COMPETITIVE ({len(comp_tasks)} tasks, {len(comp_runs)} runs)")
        print(f"{'═'*W}")

        n_decisive = sum(1 for r in comp_runs if r["passed"])
        avg_pct_c = sum(r["pct"] for r in comp_runs) / len(comp_runs) if comp_runs else 0
        avg_steps = sum(r["entry"].get("steps", 0) for r in comp_runs) / len(comp_runs) if comp_runs else 0
        print(f"  Decisive (a team won): {n_decisive}/{len(comp_runs)} ({n_decisive/len(comp_runs)*100:.1f}%)")
        print(f"  Avg completion:        {avg_pct_c:.1%}")
        print(f"  Avg steps:             {avg_steps:.0f}")

        # Group by directed matchup (e.g. "gpt-5.2 vs sonnet", "gpt-5.2")
        by_matchup = defaultdict(list)
        for r in comp_runs:
            by_matchup[r["matchup"]].append(r)

        print(f"\n  {'Matchup':<28} {'Runs':>5} {'Won':>4} {'Draw':>5} {'Avg%':>7}")
        print(f"  {'─'*28} {'─'*5} {'─'*4} {'─'*5} {'─'*7}")
        for matchup in sorted(by_matchup, key=lambda k: -len(by_matchup[k])):
            recs = by_matchup[matchup]
            n_won = sum(1 for r in recs if r["passed"])
            n_draw = len(recs) - n_won
            a = sum(r["pct"] for r in recs) / len(recs)
            print(f"  {matchup:<28} {len(recs):>5} {n_won:>4} {n_draw:>5} {a:>6.1%}")

        # Cross-model details: winner model distribution + team progress
        cross = [r for r in comp_runs if r["is_cross_model"]]
        if cross:
            # Winner by model name
            model_wins = defaultdict(int)
            draws = 0
            for r in cross:
                if r["winner_model"]:
                    model_wins[r["winner_model"]] += 1
                else:
                    draws += 1

            print(f"\n  Cross-model winner breakdown ({len(cross)} runs):")
            for model, count in sorted(model_wins.items(), key=lambda x: -x[1]):
                print(f"    {model:<20} {count} wins")
            if draws:
                print(f"    {'draw/timeout':<20} {draws}")

            # Avg team progress by model
            model_progress = defaultdict(list)
            for r in cross:
                teams = r["task"].get("teams", {})
                am = r["entry"].get("agent_models", {})
                tp = r["entry"].get("team_progress", {})
                for team_id, prog in tp.items():
                    agents = teams.get(team_id, [])
                    models = set(am.get(a, "?") for a in agents)
                    model = models.pop() if len(models) == 1 else "mixed"
                    model_progress[model].append(prog)
            if model_progress:
                parts = [
                    f"{m}: {sum(v)/len(v):.1%}"
                    for m, v in sorted(model_progress.items())
                ]
                print(f"    Avg progress by model: {' | '.join(parts)}")

    # ═══════════════════════════════════════════════════════
    # MIXED
    # ═══════════════════════════════════════════════════════
    mixed_tasks = by_cat_tasks.get("mixed", [])
    mixed_runs = by_cat_runs.get("mixed", [])
    if mixed_tasks:
        print(f"\n{'═'*W}")
        print(f"  MIXED ({len(mixed_tasks)} tasks, {len(mixed_runs)} runs)")
        print(f"{'═'*W}")

        n_pass = sum(1 for r in mixed_runs if r["passed"])
        avg = sum(r["pct"] for r in mixed_runs) / len(mixed_runs) if mixed_runs else 0
        avg_steps = sum(r["entry"].get("steps", 0) for r in mixed_runs) / len(mixed_runs) if mixed_runs else 0
        print(f"  Overall pass:   {n_pass}/{len(mixed_runs)} ({n_pass/len(mixed_runs)*100:.1f}%)")
        print(f"  Avg completion: {avg:.1%}")
        print(f"  Avg steps:      {avg_steps:.0f}")

        # Main goal vs agent subgoal breakdown (from calibration data when available)
        main_runs = [r for r in mixed_runs if "main_goal_success" in r["entry"]]
        if main_runs:
            n_main = sum(1 for r in main_runs if r["entry"]["main_goal_success"])
            print(f"\n  Main goal success: {n_main}/{len(main_runs)} ({n_main/len(main_runs)*100:.1f}%)")

        # Per-agent subgoal breakdown
        subgoal_runs = [r for r in mixed_runs if "agent_subgoal_status" in r["entry"]]
        if subgoal_runs:
            # Aggregate by agent_id
            per_agent = defaultdict(lambda: {"pass": 0, "total": 0})
            for r in subgoal_runs:
                for agent_id, passed in r["entry"]["agent_subgoal_status"].items():
                    per_agent[agent_id]["total"] += 1
                    if passed:
                        per_agent[agent_id]["pass"] += 1

            all_pass = sum(v["pass"] for v in per_agent.values())
            all_total = sum(v["total"] for v in per_agent.values())
            print(f"  Agent subgoal success: {all_pass}/{all_total} ({all_pass/all_total*100:.1f}%)")

            print(f"\n  {'Agent':<12} {'Pass':>5} {'Total':>6} {'Rate':>7}")
            print(f"  {'─'*12} {'─'*5} {'─'*6} {'─'*7}")
            for agent_id in sorted(per_agent):
                s = per_agent[agent_id]
                rate = s["pass"] / s["total"] * 100 if s["total"] else 0
                print(f"  {agent_id:<12} {s['pass']:>5} {s['total']:>6} {rate:>6.1f}%")

        # Count agents with private goals from :goal-owners in problem_pddl
        agents_with_private = 0
        agents_total = 0
        for t in mixed_tasks:
            na = t.get("num_agents", 0)
            agents_total += na
            pddl = t.get("problem_pddl", "")
            if ":goal-owners" in pddl:
                # Count distinct agent_N owners
                import re
                owners = set(re.findall(r"\(agent_\d+", pddl.split(":goal-owners")[1]))
                agents_with_private += len(owners)
        if agents_total:
            print(f"\n  Agents with private goals: {agents_with_private}/{agents_total} "
                  f"(from :goal-owners in PDDL)")

        # By agent count
        by_agents = defaultdict(list)
        for r in mixed_runs:
            by_agents[r["task"].get("num_agents", 0)].append(r)
        if len(by_agents) > 1:
            print(f"\n  {'Agents':>8} {'Runs':>5} {'Pass':>5} {'Rate':>7} {'Avg%':>7}")
            print(f"  {'─'*8} {'─'*5} {'─'*5} {'─'*7} {'─'*7}")
            for na in sorted(by_agents):
                recs = by_agents[na]
                p = sum(1 for r in recs if r["passed"])
                a = sum(r["pct"] for r in recs) / len(recs)
                print(f"  {na:>8} {len(recs):>5} {p:>5} {p/len(recs)*100:>6.1f}% {a:>6.1%}")

    print()


def summarize_tasks(tasks_dir: str) -> None:
    all_tasks = _load_tasks(tasks_dir)

    rows = []
    for d in all_tasks:
        category = d.get("category", "?")
        num_agents = d.get("num_agents", "?")
        title = d.get("title", "Untitled")
        mechanics = d.get("active_mechanics", [])

        cal_parts = []
        for entry in d.get("calibration", []):
            label = _model_label(entry.get("agent_models", {}))
            passed = entry.get("passed", False)
            pct = entry.get("percent_complete", 0.0)
            symbol = "PASS" if passed else f"{pct:.0%}"
            cal_parts.append(f"{label}: {symbol}")

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
        if "PASS" in r["cal"]:
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
