#!/usr/bin/env python3
"""
Entry point for external-agent task generation.

Usage:
    python emtom/task_gen/runner.py --config-name examples/emtom_2_robots +model=gpt-5

Or via shell script:
    ./emtom/run_emtom.sh generate --model gpt-5 --task-gen-agent mini
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from emtom.task_gen.event_log import append_event, maybe_int, write_run_manifest, write_worker_snapshot
from emtom.task_gen.external_agent import ExternalAgentError, ExternalAgentLauncher
from emtom.task_gen.authoring_surface import (
    AUTHORING_ITEMS_NOTICE,
    get_authoring_action_descriptions,
    get_authoring_default_actions,
    get_authoring_mechanics,
    get_authoring_predicates,
)
from emtom.task_gen.prompts import build_external_taskgen_prompt
from emtom.task_gen.seed_selector import (
    SeedSelectionConfig,
    is_task_like_json,
    resolve_seed_tasks_dir,
    select_seed_tasks,
)
from emtom.task_gen.session import default_state


def parse_extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--retry-verification", type=str, default=None)
    parser.add_argument("--target-model", type=str, default=None)
    parser.add_argument("--calibration-model", type=str, default="gpt-5.2")
    parser.add_argument("--target-pass-rate", type=float, default=0.20)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["cooperative", "competitive", "mixed"],
    )
    parser.add_argument("--seed-tasks-dir", type=str, default=None)
    parser.add_argument("--seed-pass-ratio", type=float, default=0.20)
    parser.add_argument("--seed-fail-ratio", type=float, default=0.80)
    parser.add_argument("--sampled-tasks-dir", type=str, default=None)
    parser.add_argument("--judge-threshold", type=float, default=None)
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--test-model", type=str, default=None)
    parser.add_argument("--k-level", type=int, nargs="*", default=None)
    parser.add_argument(
        "--task-gen-agent",
        type=str,
        default="mini",
        choices=["mini", "claude", "codex"],
        help="External agent CLI used for task generation.",
    )
    parser.add_argument(
        "--remove",
        type=str,
        nargs="+",
        default=None,
        help="Skip pipeline components. Choices: pddl, llm-council, simulation, task-evolution, tom, structure, test",
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def parse_runner_args(argv: list[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "num_tasks": 1,
        "model": None,
        "llm_provider": None,
        "output_dir": "data/emtom/tasks",
        "subtasks_min": 2,
        "subtasks_max": 5,
        "agents_min": 2,
        "agents_max": 2,
        "quiet": False,
        "config_name": None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-name":
            if i + 1 >= len(argv):
                raise SystemExit("Error: --config-name requires a value.")
            config["config_name"] = argv[i + 1]
            i += 2
            continue
        if arg.startswith("+") and "=" in arg:
            key, value = arg[1:].split("=", 1)
            if key in {"num_tasks", "subtasks_min", "subtasks_max", "agents_min", "agents_max"}:
                config[key] = int(value)
            elif key == "quiet":
                config[key] = value.lower() == "true"
            elif key in {"model", "llm_provider", "output_dir"}:
                config[key] = value
            i += 1
            continue
        i += 1

    return config


def _infer_task_tom_level(task_data: dict) -> Optional[int]:
    stored = task_data.get("tom_level")
    if isinstance(stored, int) and 0 <= stored <= 3:
        return stored

    try:
        from emtom.task_gen.task_generator import GeneratedTask

        task = GeneratedTask.from_dict(task_data)
        level = task.compute_tom_level(scene_data=None)
        if isinstance(level, int):
            return min(max(level, 0), 3)
    except Exception:
        return None
    return None


def build_workspace_id(task_gen_agent: str, now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-{task_gen_agent}-{uuid.uuid4().hex[:8]}"


def build_generation_run_id(now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-generation-{uuid.uuid4().hex[:8]}"


def _copy_sample(src_path: Path, sampled_tasks_dir: Path, index: int) -> None:
    shutil.copy(src_path, sampled_tasks_dir / f"task_{index}.json")


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _summarize_goal_metadata(task_data: dict) -> tuple[str, str]:
    problem_pddl = task_data.get("problem_pddl")
    if not isinstance(problem_pddl, str) or not problem_pddl.strip():
        return "none", "none"

    try:
        from emtom.pddl.problem_pddl import parse_problem_pddl

        parsed = parse_problem_pddl(problem_pddl)
        goal_summary = _truncate_text(parsed.goal_pddl, limit=260)
        owners = parsed.owners or {}
        if owners:
            owner_counts: dict[str, int] = {}
            for owner in owners.values():
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
            owner_summary = ", ".join(
                f"{owner}:{count}" for owner, count in sorted(owner_counts.items())
            )
        else:
            owner_summary = "none"
        return goal_summary, owner_summary
    except Exception:
        return _truncate_text(problem_pddl, limit=260), "unknown"


def _latest_calibration_pair(task_data: dict, model: str) -> tuple[Optional[dict], Optional[dict]]:
    from emtom.evolve.benchmark_wrapper import find_calibration_entry

    standard = find_calibration_entry(task_data.get("calibration", []), model=model, run_mode="standard")
    baseline = find_calibration_entry(task_data.get("calibration", []), model=model, run_mode="baseline")
    return standard, baseline


def _calibration_outcome_label(entry: Optional[dict]) -> str:
    from emtom.evolve.benchmark_wrapper import cal_passed, cal_progress

    if not entry:
        return "missing"
    progress = cal_progress(entry)
    status = "PASS" if cal_passed(entry) else "FAIL"
    steps = entry.get("steps")
    step_text = f", steps={steps}" if isinstance(steps, int) else ""
    return f"{status} (progress={progress:.0%}{step_text})"


def _calibration_gap_label(standard: Optional[dict], baseline: Optional[dict]) -> str:
    from emtom.evolve.benchmark_wrapper import cal_passed

    if baseline is None and standard is None:
        return "UNTESTED"
    if baseline is None:
        return "NO_BASELINE"
    if standard is None:
        return "NO_STANDARD"
    standard_passed = cal_passed(standard)
    baseline_passed = cal_passed(baseline)
    if baseline_passed and not standard_passed:
        return "GOOD_HARD_GAP"
    if baseline_passed and standard_passed:
        return "TOO_EASY"
    if not baseline_passed:
        return "UNSOLVABLE"
    return "MIXED"


def _calibration_gap_reason(standard: Optional[dict], baseline: Optional[dict]) -> str:
    from emtom.evolve.benchmark_wrapper import cal_passed, cal_progress

    gap_label = _calibration_gap_label(standard, baseline)
    if gap_label == "GOOD_HARD_GAP":
        std_progress = cal_progress(standard)
        base_progress = cal_progress(baseline)
        return (
            "Baseline solved while standard failed. Inspect which hidden fact, observation, or communication bottleneck "
            f"explains the {base_progress - std_progress:.0%} progress gap."
        )
    if gap_label == "TOO_EASY":
        return "Both runs solved it. This pattern is probably too easy for a hard target unless you deepen the information asymmetry."
    if gap_label == "UNSOLVABLE":
        return "Baseline failed too. The physical plan or mechanic stack is probably brittle, not just ToM-demanding."
    if gap_label == "NO_BASELINE":
        return "Only a standard calibration entry is available."
    if gap_label == "NO_STANDARD":
        return "Only a baseline calibration entry is available."
    return "No clear standard-vs-baseline lesson yet."


def _trajectory_digest(entry: Optional[dict], max_turns: int = 4) -> list[str]:
    if not entry:
        return ["- No saved trajectory."]

    trajectory = entry.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return ["- No saved trajectory."]

    lines: list[str] = []
    subtask_events = sum(len(turn.get("subtasks_completed", []) or []) for turn in trajectory if isinstance(turn, dict))
    lines.append(f"- Saved turns: {len(trajectory)}; subtask events: {subtask_events}.")

    for turn in trajectory[:max_turns]:
        if not isinstance(turn, dict):
            continue
        agents = turn.get("agents", {}) if isinstance(turn.get("agents"), dict) else {}
        action_bits = []
        for agent_id in sorted(agents):
            agent_data = agents.get(agent_id) or {}
            action = _truncate_text(agent_data.get("action", "unknown"), limit=80)
            action_bits.append(f"{agent_id}: {action}")
        subtasks = turn.get("subtasks_completed", []) or []
        subtask_text = f" | subtasks: {', '.join(str(s) for s in subtasks[:3])}" if subtasks else ""
        action_text = "; ".join(action_bits) if action_bits else "no agent actions saved"
        lines.append(f"- Turn {turn.get('turn', '?')}: {action_text}{subtask_text}")

    if len(trajectory) > max_turns:
        lines.append("- Additional turns omitted here; inspect raw `calibration[*].trajectory` in the task JSON for the full trace.")

    return lines


def _write_sampled_task_analysis_file(sample_path: Path, task_data: dict, model: str) -> None:
    standard, baseline = _latest_calibration_pair(task_data, model)
    analysis_lines = [
        f"# {sample_path.name} Analysis",
        "",
        f"- Title: {_truncate_text(task_data.get('title') or sample_path.name, limit=140)}",
        f"- Category: {task_data.get('category', 'unknown')}",
        f"- Agents: {task_data.get('num_agents', 'unknown')}",
        f"- Standard: {_calibration_outcome_label(standard)}",
        f"- Baseline: {_calibration_outcome_label(baseline)}",
        f"- Gap label: {_calibration_gap_label(standard, baseline)}",
        f"- Lesson: {_calibration_gap_reason(standard, baseline)}",
        "",
        "## How To Use This Example",
        "- Ask why baseline/full-info succeeded or failed.",
        "- If baseline passed and standard failed, identify the private fact or communication bottleneck that standard could not recover.",
        "- If baseline failed too, treat this as a structural anti-example rather than a ToM-gap example.",
        "",
        "## Standard Trajectory Digest",
        *_trajectory_digest(standard),
        "",
        "## Baseline Trajectory Digest",
        *_trajectory_digest(baseline),
        "",
        "## Raw Trace Location",
        f"- Inspect `{sample_path.name}` and its `calibration[*].trajectory` entries for the full saved trace.",
    ]
    (sample_path.with_suffix("").with_name(sample_path.stem + "_analysis.md")).write_text("\n".join(analysis_lines))


def _write_sampled_tasks_summary(sampled_tasks_dir: Path, model: str = "gpt-5.2") -> None:
    from emtom.evolve.benchmark_wrapper import cal_passed, cal_progress, find_calibration_entry

    sample_files = sorted(sampled_tasks_dir.glob("task_*.json"))
    if not sample_files:
        return

    lines = [
        "# Sampled Task Summary",
        "",
        "Read this file first. It contains the compact fields that matter for seed-task inspiration.",
        "Use sampled tasks for structure only. Do not copy scene IDs, object IDs, agent IDs, or exact wording.",
        "",
        "**Benchmark column**: PASS = baseline solved the task and it was submitted successfully.",
        "FAIL = task was benchmarked but baseline could not solve it. UNTESTED = no benchmark data.",
        "Prefer structural patterns from PASS tasks — they are empirically solvable.",
        "For hard tasks, prioritize examples labeled `GOOD_HARD_GAP`: baseline passes, standard fails, and the gap is likely informational rather than physical.",
        "If you get stuck, inspect more sampled tasks plus their `task_N_analysis.md` files and compare the standard vs baseline traces directly.",
        "",
    ]

    for path in sample_files:
        try:
            with open(path) as f:
                task_data = json.load(f)
        except Exception:
            continue

        title = _truncate_text(task_data.get("title") or path.name, limit=120)
        category = task_data.get("category", "unknown")
        num_agents = task_data.get("num_agents", "unknown")
        mechanics = task_data.get("active_mechanics")
        if not mechanics and isinstance(task_data.get("mechanic_bindings"), list):
            mechanics = [
                binding.get("mechanic_type")
                for binding in task_data["mechanic_bindings"]
                if isinstance(binding, dict) and binding.get("mechanic_type")
            ]
        mechanics_str = ", ".join(str(m) for m in (mechanics or [])) or "none"
        task_text = _truncate_text(task_data.get("task"), limit=240)
        goal_summary, owner_summary = _summarize_goal_metadata(task_data)
        standard_cal, baseline_cal = _latest_calibration_pair(task_data, model)
        gap_label = _calibration_gap_label(standard_cal, baseline_cal)
        gap_reason = _calibration_gap_reason(standard_cal, baseline_cal)

        # Benchmark pass/fail signal from calibration data
        cal = find_calibration_entry(task_data.get("calibration", []), model=model)
        if cal is None:
            benchmark_str = "UNTESTED"
        elif cal_passed(cal):
            progress = cal_progress(cal)
            benchmark_str = f"PASS (progress={progress:.0%})"
        else:
            progress = cal_progress(cal)
            benchmark_str = f"FAIL (progress={progress:.0%})"

        lines.extend(
            [
                f"## {path.name}",
                f"- Title: {title}",
                f"- Category: {category}",
                f"- Agents: {num_agents}",
                f"- Mechanics: {mechanics_str}",
                f"- Task: {task_text}",
                f"- Goal: {goal_summary}",
                f"- Goal owners: {owner_summary}",
                f"- Benchmark: {benchmark_str}",
                f"- Standard: {_calibration_outcome_label(standard_cal)}",
                f"- Baseline: {_calibration_outcome_label(baseline_cal)}",
                f"- Standard vs baseline: {gap_label}",
                f"- Why this matters: {gap_reason}",
                f"- Analysis file: {path.stem}_analysis.md",
                "",
            ]
        )
        _write_sampled_task_analysis_file(path, task_data, model)

    # Add failure analysis section based on sampled tasks
    failed_patterns: list[str] = []
    passed_patterns: list[str] = []
    hard_gap_patterns: list[str] = []
    for path in sample_files:
        try:
            with open(path) as f:
                task_data = json.load(f)
        except Exception:
            continue
        category = task_data.get("category", "unknown")
        mechanics = task_data.get("active_mechanics") or []
        if not mechanics and isinstance(task_data.get("mechanic_bindings"), list):
            mechanics = [b.get("mechanic_type") for b in task_data["mechanic_bindings"] if isinstance(b, dict)]
        pddl = task_data.get("problem_pddl", "")
        has_K = "(K " in pddl
        num_agents = task_data.get("num_agents", 0)
        mech_str = "+".join(sorted(set(str(m) for m in mechanics if m))) or "none"
        label = f"{category}/{num_agents}a/{mech_str}/K={'yes' if has_K else 'no'}"
        standard_cal, baseline_cal = _latest_calibration_pair(task_data, model)
        gap_label = _calibration_gap_label(standard_cal, baseline_cal)

        cal = find_calibration_entry(task_data.get("calibration", []), model=model)
        if cal is None:
            continue

        if cal_passed(cal):
            passed_patterns.append(label)
        else:
            failed_patterns.append(label)
        if gap_label == "GOOD_HARD_GAP":
            hard_gap_patterns.append(label)

    if failed_patterns or passed_patterns:
        lines.extend([
            "---",
            "",
            "# Failure Analysis (learn from these patterns)",
            "",
        ])
        if hard_gap_patterns:
            lines.append("**Patterns with GOOD_HARD_GAP** (baseline passes, standard fails; inspect these first for hard ToM examples):")
            for p in dict.fromkeys(hard_gap_patterns):
                lines.append(f"  - {p}")
            lines.append("")
        if passed_patterns:
            lines.append("**Patterns that PASS** (reuse these structural shapes):")
            for p in dict.fromkeys(passed_patterns):  # dedupe preserving order
                lines.append(f"  - {p}")
            lines.append("")
        if failed_patterns:
            lines.append("**Patterns that FAIL** (avoid or redesign):")
            for p in dict.fromkeys(failed_patterns):
                lines.append(f"  - {p}")
            lines.append("")
        lines.extend([
            "## Common Anti-Patterns (from empirical testing)",
            "- Cross-room object relay (Place on room name instead of furniture) → runtime crash",
            "- Prescriptive secrets ('Tell your partner', 'Ask them') → judge blocks on secret_quality",
            "- Meta-knowledge in secrets ('agent_1 knows X') → judge blocks on secret_quality",
            "- Independent parallel work (no info exchange needed) → judge blocks on task_interdependence",
            "- K=0 tasks (no K() goal in PDDL) → rejected at submit",
            "- 'agent_X cannot enter room' in wrong agent's secret → compiler assigns restriction to wrong agent",
            "",
        ])

    (sampled_tasks_dir / "SUMMARY.md").write_text("\n".join(lines))


def populate_sampled_tasks_dir(
    sampled_tasks_dir: Path,
    selection_config: SeedSelectionConfig,
    sample_count: int = 10,
) -> tuple[Optional[Path], int]:
    selected = select_seed_tasks(selection_config, count=sample_count)
    for i, candidate in enumerate(selected, 1):
        _copy_sample(candidate.path, sampled_tasks_dir, i)
    _write_sampled_tasks_summary(sampled_tasks_dir, model=selection_config.target_model)
    return selection_config.tasks_dir if selected else None, len(selected)


def compute_calibration_stats(tasks_dir: Union[str, Iterable[str]], model: str) -> dict:
    from emtom.evolve.benchmark_wrapper import cal_passed, find_calibration_entry

    stats = {
        "passed": 0,
        "failed": 0,
        "untested": 0,
        "excluded_tom_level_zero": 0,
        "model": model,
        "tom_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        "tom_total": 0,
        "tom_unknown": 0,
        "tom_ratios": {0: None, 1: None, 2: None, 3: None},
    }
    task_dirs = [tasks_dir] if isinstance(tasks_dir, str) else list(tasks_dir)
    task_paths = [Path(path) for path in task_dirs if path]
    if not task_paths:
        stats["total"] = 0
        stats["rate"] = None
        return stats

    seen_keys = set()
    task_files = []
    for tasks_path in task_paths:
        if not tasks_path.exists():
            continue
        for task_file in tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
            except Exception:
                continue
            if not isinstance(task, dict):
                continue
            task_key = task.get("task_id") or str(task_file.resolve())
            if task_key in seen_keys:
                continue
            seen_keys.add(task_key)
            task_files.append((task_file, task))

    if not task_files:
        stats["total"] = 0
        stats["rate"] = None
        return stats

    for task_file, task in task_files:
        try:
            tom_level = _infer_task_tom_level(task)
            if tom_level in (0, 1, 2, 3):
                stats["tom_counts"][tom_level] += 1
                stats["tom_total"] += 1
            else:
                stats["tom_unknown"] += 1

            if isinstance(tom_level, int) and tom_level < 1:
                stats["excluded_tom_level_zero"] += 1
                continue

            cal = find_calibration_entry(task.get("calibration", []), model=model)
            if cal is None:
                stats["untested"] += 1
            elif cal_passed(cal):
                stats["passed"] += 1
            else:
                stats["failed"] += 1
        except Exception:
            continue

    stats["total"] = stats["passed"] + stats["failed"]
    stats["rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else None
    if stats["tom_total"] > 0:
        for level in (0, 1, 2, 3):
            stats["tom_ratios"][level] = stats["tom_counts"][level] / stats["tom_total"]
    return stats


def _load_verification_feedback(path_str: Optional[str]) -> Optional[dict]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            verification_data = json.load(f)
    except Exception:
        return None
    if verification_data.get("is_valid_tom", True):
        return None
    return {
        "required_fixes": verification_data.get("required_fixes", []),
        "criteria": verification_data.get("criteria", {}),
        "overall_reasoning": verification_data.get("overall_reasoning", ""),
    }


def _write_template_file(template_file: Path, agents_max: int) -> None:
    source_template = Path(__file__).parent / "template" / "template.json"
    with open(source_template) as f:
        template = json.load(f)
    template["num_agents"] = agents_max
    default_actions = get_authoring_default_actions(include_find_tools=False)
    template["agent_secrets"] = {
        f"agent_{i}": ["REPLACE_WITH_SECRET_INFO"] for i in range(agents_max)
    }
    template["agent_actions"] = {
        f"agent_{i}": default_actions.copy() for i in range(agents_max)
    }
    with open(template_file, "w") as f:
        json.dump(template, f, indent=2)


def _write_taskgen_shim(working_dir: Path) -> None:
    bin_dir = working_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim_path = bin_dir / "taskgen"
    shim_contents = f"""#!/usr/bin/env bash
set -euo pipefail
exec "{sys.executable}" -m emtom.cli.taskgen --working-dir "{working_dir}" "$@"
"""
    shim_path.write_text(shim_contents)
    shim_path.chmod(0o755)


def _write_bootstrap_files(
    *,
    working_dir: Path,
    prompt_text: str,
    available_items: str,
    available_mechanics: str,
    available_predicates: str,
    action_descriptions: str,
) -> None:
    (working_dir / "taskgen_prompt.md").write_text(prompt_text)
    (working_dir / "available_items.md").write_text(available_items)
    (working_dir / "available_mechanics.md").write_text(available_mechanics)
    (working_dir / "available_predicates.md").write_text(available_predicates)
    (working_dir / "available_actions.md").write_text(action_descriptions)
    (working_dir / "bootstrap_prompt.txt").write_text(prompt_text)


def main() -> None:
    extra_args = getattr(main, "_extra_args", None)
    runner_args = parse_runner_args(sys.argv[1:])

    num_tasks = runner_args["num_tasks"]
    model = runner_args["model"]
    llm_provider = runner_args["llm_provider"]
    output_dir = str(Path(runner_args["output_dir"]).resolve())
    subtasks_min = runner_args["subtasks_min"]
    subtasks_max = runner_args["subtasks_max"]
    agents_min = runner_args["agents_min"]
    agents_max = runner_args["agents_max"]
    quiet = runner_args["quiet"]

    if not model:
        raise SystemExit("Error: model is required.")

    query = extra_args.query if extra_args else None
    retry_verification = extra_args.retry_verification if extra_args else None
    target_model = None
    if extra_args:
        target_model = extra_args.target_model or extra_args.calibration_model
    if not target_model:
        target_model = "gpt-5.2"
    target_pass_rate = extra_args.target_pass_rate if extra_args else 0.20
    category = extra_args.category if extra_args else None
    seed_tasks_dir_arg = extra_args.seed_tasks_dir if extra_args else None
    seed_pass_ratio = extra_args.seed_pass_ratio if extra_args else 0.20
    seed_fail_ratio = extra_args.seed_fail_ratio if extra_args else 0.80
    judge_threshold = extra_args.judge_threshold if extra_args else None
    difficulty = extra_args.difficulty if extra_args else None
    test_model = extra_args.test_model if extra_args else None
    k_levels = extra_args.k_level if extra_args else None
    task_gen_agent = extra_args.task_gen_agent if extra_args else "mini"
    skip_steps = extra_args.remove if extra_args else None
    if skip_steps:
        # Canonical names and legacy aliases
        _legacy_map = {"council": "llm-council", "golden": "simulation"}
        skip_steps = [_legacy_map.get(s, s) for s in skip_steps]
        valid_steps = {"pddl", "llm-council", "simulation", "task-evolution", "tom", "structure", "test"}
        invalid = [s for s in skip_steps if s not in valid_steps]
        if invalid:
            raise SystemExit(
                f"Error: --remove got invalid steps {invalid}. "
                f"Valid choices: {sorted(valid_steps)}"
            )

    if k_levels is not None:
        invalid = [k for k in k_levels if k not in (1, 2, 3)]
        if invalid:
            raise SystemExit(f"Error: --k-level values must be 1, 2, or 3 (got {invalid})")
        k_levels = sorted(set(k_levels))
    if seed_pass_ratio < 0 or seed_fail_ratio < 0:
        raise SystemExit("Error: --seed-pass-ratio and --seed-fail-ratio must be non-negative.")
    if seed_pass_ratio == 0 and seed_fail_ratio == 0:
        raise SystemExit("Error: at least one of --seed-pass-ratio or --seed-fail-ratio must be positive.")

    verification_feedback = _load_verification_feedback(retry_verification)
    project_root = Path(__file__).resolve().parent.parent.parent
    workspace_root = project_root / "tmp" / "task_gen"
    workspace_root.mkdir(parents=True, exist_ok=True)
    instance_id = build_workspace_id(task_gen_agent)
    working_dir = workspace_root / instance_id
    working_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_generation_path(path_value: Any) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()

    generation_run_id = os.environ.get("EMTOM_GENERATION_RUN_ID") or build_generation_run_id()
    generation_run_dir = _resolve_generation_path(
        os.environ.get("EMTOM_GENERATION_RUN_DIR")
        or (project_root / "outputs" / "generations" / generation_run_id)
    )
    generation_worker_id = os.environ.get("EMTOM_GENERATION_WORKER_ID") or "worker-0"
    generation_worker_dir = _resolve_generation_path(
        os.environ.get("EMTOM_GENERATION_WORKER_DIR")
        or (generation_run_dir / "workers" / generation_worker_id)
    )
    generation_mode = os.environ.get("EMTOM_GENERATION_MODE") or "single"
    generation_gpu = maybe_int(os.environ.get("EMTOM_GENERATION_GPU"))
    generation_slot = maybe_int(os.environ.get("EMTOM_GENERATION_SLOT"))
    generation_total_workers = maybe_int(os.environ.get("EMTOM_GENERATION_TOTAL_WORKERS"), 1)
    generation_requested_tasks = maybe_int(os.environ.get("EMTOM_GENERATION_REQUESTED_TASKS"), num_tasks)
    generation_stdout_log = os.environ.get("EMTOM_GENERATION_STDOUT_LOG") or ""
    if generation_stdout_log:
        generation_stdout_log = str(_resolve_generation_path(generation_stdout_log))
    generation_run_dir.mkdir(parents=True, exist_ok=True)
    generation_worker_dir.mkdir(parents=True, exist_ok=True)

    sampled_tasks_dir = working_dir / "sampled_tasks"
    sampled_tasks_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "agent_trajectories").mkdir(parents=True, exist_ok=True)
    (working_dir / "submitted_tasks").mkdir(parents=True, exist_ok=True)
    if k_levels:
        current_k_level = random.choice(k_levels)
    else:
        current_k_level = random.choice([1, 2, 3])

    if not test_model:
        test_model = target_model

    seed_tasks_dir = resolve_seed_tasks_dir(seed_tasks_dir_arg, output_dir)
    calibration_task_dirs = []
    if seed_tasks_dir is not None:
        calibration_task_dirs.append(str(seed_tasks_dir))
    calibration_task_dirs.append(output_dir)
    calibration_stats = compute_calibration_stats(calibration_task_dirs, target_model)
    calibration_stats["target_rate"] = target_pass_rate

    # When task-evolution is removed, skip seed task population and calibration guidance
    _skip_evolution = skip_steps and "task-evolution" in skip_steps
    sampled_tasks_override = extra_args.sampled_tasks_dir if extra_args else None
    if _skip_evolution:
        pass  # No seed tasks — agent generates from scratch
    elif sampled_tasks_override:
        override_path = Path(sampled_tasks_override)
        override_files = [p for p in override_path.glob("*.json") if is_task_like_json(p)]
        selected = sorted(override_files)[:10]
        for i, task_path in enumerate(selected, 1):
            shutil.copy(task_path, sampled_tasks_dir / task_path.name)
            _copy_sample(task_path, sampled_tasks_dir, i)
        _write_sampled_tasks_summary(sampled_tasks_dir, model=target_model)
    elif seed_tasks_dir is not None:
        selection_config = SeedSelectionConfig(
            tasks_dir=seed_tasks_dir,
            target_model=target_model,
            target_pass_rate=target_pass_rate,
            current_pass_rate=calibration_stats["rate"],
            category=category,
            tom_level=current_k_level,
            pass_seed_ratio=seed_pass_ratio,
            fail_seed_ratio=seed_fail_ratio,
        )
        populate_sampled_tasks_dir(
            sampled_tasks_dir,
            selection_config,
            sample_count=10,
        )

    _write_template_file(working_dir / "template.json", agents_max)
    _write_taskgen_shim(working_dir)

    available_items = AUTHORING_ITEMS_NOTICE
    available_mechanics = get_authoring_mechanics()
    available_predicates = get_authoring_predicates()
    action_descriptions = get_authoring_action_descriptions()

    prompt_text = build_external_taskgen_prompt(
        working_dir=str(working_dir),
        task_file=str(working_dir / "working_task.json"),
        category=category or "random",
        num_tasks=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        query=query,
        verification_feedback=verification_feedback,
        calibration_stats=calibration_stats if not _skip_evolution else {},
        difficulty=difficulty if not _skip_evolution else None,
        current_k_level=current_k_level,
        seed_tasks_dir=(str(seed_tasks_dir) if seed_tasks_dir is not None else None) if not _skip_evolution else None,
        seed_pass_ratio=seed_pass_ratio,
        seed_fail_ratio=seed_fail_ratio,
        skip_steps=skip_steps,
    )
    _write_bootstrap_files(
        working_dir=working_dir,
        prompt_text=prompt_text,
        available_items=available_items,
        available_mechanics=available_mechanics,
        available_predicates=available_predicates,
        action_descriptions=action_descriptions,
    )

    state = default_state(
        working_dir=str(working_dir),
        output_dir=output_dir,
        num_tasks_target=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        category=category,
        seed_tasks_dir=str(seed_tasks_dir) if seed_tasks_dir is not None else None,
        seed_pass_ratio=seed_pass_ratio,
        seed_fail_ratio=seed_fail_ratio,
        judge_threshold=judge_threshold,
        difficulty=difficulty,
        test_model=test_model,
        calibration_stats=calibration_stats,
        calibration_tasks_dirs=calibration_task_dirs,
        task_gen_agent=task_gen_agent,
        allowed_k_levels=k_levels,
        skip_steps=skip_steps,
        generation_run_id=generation_run_id,
        generation_run_dir=str(generation_run_dir),
        generation_worker_id=generation_worker_id,
        generation_worker_dir=str(generation_worker_dir),
    )
    state["current_k_level"] = current_k_level
    state["task_gen_model"] = model
    state["task_gen_llm_provider"] = llm_provider
    with open(working_dir / "taskgen_state.json", "w") as f:
        json.dump(state, f, indent=2)
    write_run_manifest(
        generation_run_dir,
        run_id=generation_run_id,
        started_at=datetime.now().isoformat(),
        mode=generation_mode,
        total_workers=generation_total_workers,
        requested_tasks=generation_requested_tasks,
        output_dir=output_dir,
        task_gen_agent=task_gen_agent,
        model=model,
    )
    write_worker_snapshot(
        generation_worker_dir,
        worker_id=generation_worker_id,
        run_id=generation_run_id,
        mode=generation_mode,
        gpu=generation_gpu,
        slot=generation_slot,
        category=category or "random",
        workspace_id=instance_id,
        workspace_path=str(working_dir),
        output_dir=output_dir,
        task_gen_agent=task_gen_agent,
        task_gen_model=model,
        target_tasks=num_tasks,
        submitted_count=0,
        current_task_index=1,
        current_k_level=current_k_level,
        scene_id=None,
        episode_id=None,
        finished=False,
        failed=False,
        fail_reason="",
        status="running",
        agent_trace_path=str(generation_worker_dir / "agent_trace.json"),
        stdout_log_path=generation_stdout_log,
    )
    if generation_stdout_log:
        write_worker_snapshot(
            generation_worker_dir,
            stdout_log_path=generation_stdout_log,
        )
    append_event(
        generation_worker_dir,
        "workspace_initialized",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        workspace=str(working_dir),
        task_gen_agent=task_gen_agent,
        model=model,
        llm_provider=llm_provider,
        category=category,
        num_tasks_target=num_tasks,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
        current_k_level=current_k_level,
        output_dir=output_dir,
    )

    bootstrap_prompt = (working_dir / "bootstrap_prompt.txt").read_text()

    print("=" * 60)
    print("EMTOM External Task Generator")
    print("=" * 60)
    print(f"Task-gen agent: {task_gen_agent}")
    print(f"Model: {model}")
    print(f"Workspace: {working_dir}")
    print(f"Output: {output_dir}")
    print(f"Target tasks: {num_tasks}")
    if query:
        print(f"Query: {query}")
    if not quiet:
        print("Prompt file:", working_dir / "taskgen_prompt.md")
    print("=" * 60)

    launcher = ExternalAgentLauncher(project_root)
    try:
        append_event(
            generation_worker_dir,
            "agent_launch_started",
            run_id=generation_run_id,
            worker_id=generation_worker_id,
            agent_name=task_gen_agent,
            model=model,
        )
        return_code = launcher.run(
            agent_name=task_gen_agent,
            workspace_dir=working_dir,
            bootstrap_prompt=bootstrap_prompt,
            model=model,
            trace_output_path=generation_worker_dir / "agent_trace.json",
        )
    except ExternalAgentError as exc:
        append_event(
            generation_worker_dir,
            "agent_launch_failed",
            run_id=generation_run_id,
            worker_id=generation_worker_id,
            agent_name=task_gen_agent,
            model=model,
            error=str(exc),
        )
        write_worker_snapshot(
            generation_worker_dir,
            status="failed",
            failed=True,
            fail_reason=str(exc),
        )
        raise SystemExit(str(exc)) from exc

    with open(working_dir / "taskgen_state.json") as f:
        final_state = json.load(f)
    final_submitted_tasks = final_state.get("submitted_tasks", [])
    final_status = "failed" if final_state.get("failed", False) else "finished" if final_state.get("finished", False) else "stopped"
    write_worker_snapshot(
        generation_worker_dir,
        status=final_status,
        submitted_count=len(final_submitted_tasks),
        current_task_index=final_state.get("current_task_index"),
        current_k_level=final_state.get("current_k_level"),
        scene_id=final_state.get("scene_id"),
        episode_id=final_state.get("episode_id"),
        finished=final_state.get("finished", False),
        failed=final_state.get("failed", False),
        fail_reason=final_state.get("fail_reason", ""),
        submitted_tasks=final_submitted_tasks,
        workspace_path=str(working_dir),
    )
    append_event(
        generation_worker_dir,
        "generation_finished",
        run_id=generation_run_id,
        worker_id=generation_worker_id,
        agent_name=task_gen_agent,
        model=model,
        return_code=return_code,
        finished=final_state.get("finished", False),
        failed=final_state.get("failed", False),
        fail_reason=final_state.get("fail_reason", ""),
        submitted_tasks=final_state.get("submitted_tasks", []),
        submitted_count=len(final_state.get("submitted_tasks", [])),
    )

    print()
    print("=" * 60)
    print("Generation Result")
    print("=" * 60)
    print(f"Agent exit code: {return_code}")
    print(f"Finished: {final_state.get('finished', False)}")
    print(f"Failed: {final_state.get('failed', False)}")
    print(f"Workspace retained at: {working_dir}")
    for task_path in final_state.get("submitted_tasks", []):
        print(f"  - {task_path}")

    if final_state.get("failed"):
        raise SystemExit(final_state.get("fail_reason") or "Task generation failed.")
    if not final_state.get("finished"):
        # If tasks were submitted despite not finishing, treat as partial success
        if final_submitted_tasks:
            print(f"Warning: Agent exited without calling taskgen finish, but submitted {len(final_submitted_tasks)} task(s).")
        else:
            raise SystemExit("Task generation did not call taskgen finish.")


if __name__ == "__main__":
    extra_args = parse_extra_args()
    main._extra_args = extra_args
    main()
