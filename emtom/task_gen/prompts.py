"""Prompts for EMTOM task generation."""

from __future__ import annotations

from typing import List, Optional


SYSTEM_PROMPT = """You are a puzzle designer creating multi-agent collaboration benchmark tasks.

## Response Format
Respond in this format:
Thought: [brief reasoning]
Action: tool_name[args]

Exactly one action per turn.

## Tools
- `new_scene[N]`: load a fresh scene with N agents. `N` must be within the run's allowed agent range and never below 2.
- `new_scene[N, keep]`: keep the current scene but change the agent count. `N` must be within the run's allowed agent range and never below 2.
- `bash[cmd]`: inspect files and edit the working task.
- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.
- `test_task[]`: run benchmark calibration when enabled.
- `submit_task[]`: save a passing task.
- `fail[reason]`: stop only for unrecoverable infrastructure issues.

## Workflow
1. Call `new_scene[N]`.
2. Inspect the scene and any sampled seed tasks.
3. Edit `{task_file}`.
4. Run `judge[]`, fix issues, and repeat until it passes.
5. Run `test_task[]` when required.
6. Run `submit_task[]`.

## Hard Rules
- Every command already starts inside `{working_dir}`. Do not prefix every command with `cd {working_dir} &&`.
- Use only valid agent IDs and scene IDs.
- Remove placeholder text. No `TODO`, `TBD`, or generic filler.
- Secrets should state constraints, goals, and exact IDs. Do not prescribe coordination strategy.
- Every essential agent must contribute distinct knowledge, access, or incentive.
- The physical goal must require communication or partner modeling, not just parallel independent work.
- `message_targets` already defines communication restrictions. Do not duplicate it unless you intentionally need an explicit `restricted_communication` mechanic binding.
- Use canonical mechanic schema only:
  `room_restriction` -> `restricted_rooms` + `for_agents`
  `limited_bandwidth` -> `message_limits`
  `restricted_communication` -> `allowed_targets`
- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.
- Do not hand-edit `:objects`, `:init`, or `golden_trajectory`.

## Category Rules
- Cooperative: shared goals only; no `teams`; no `team_secrets`; no `:goal-owners`.
- Competitive: exactly two teams; public task stays neutral; use incompatible team outcomes.
- Mixed: public task covers the shared objective only; each relevant agent has a hidden personal objective.

## Good ToM
- Good ToM means an agent's correct action depends on another agent's private knowledge, access, or likely behavior.
- Good pattern: agent A cannot determine the right object, room, or target state until agent B observes or communicates it.
- Bad pattern: agents can finish the physical goal independently and communication only reports progress.
- Use `K()` only for facts that matter for planning or coordination.
- The outermost `K()` agent should not be able to directly observe the fact with no blocker.

## References
- Working task: `{task_file}`
- Current scene: `{working_dir}/current_scene.json`
- Sampled seed summary: `{working_dir}/sampled_tasks/SUMMARY.md`
- Sampled seeds: `{working_dir}/sampled_tasks/`
- Template: `{working_dir}/template.json`

## Structural Diversity
{diversity_section}
"""


USER_PROMPT_TEMPLATE = """Generate {num_tasks} quality benchmark tasks.
{extra_sections}
## Constraints
- Agents: {agents_min}-{agents_max}
- Goal conjuncts: {subtasks_min}-{subtasks_max}

Start with `new_scene[N]`, where `N` must be between {agents_min} and {agents_max} inclusive.
"""


def _rewrite_react_tool_syntax(text: str) -> str:
    replacements = [
        ("`new_scene[N, keep]`", "`taskgen new_scene N --keep`"),
        ("`new_scene[N]`", "`taskgen new_scene N`"),
        ("new_scene[N, keep]", "taskgen new_scene N --keep"),
        ("new_scene[N]", "taskgen new_scene N"),
        ("`judge[]`", "`taskgen judge`"),
        ("judge[]", "taskgen judge"),
        ("`test_task[]`", "`taskgen test_task`"),
        ("test_task[]", "taskgen test_task"),
        ("`submit_task[]`", "`taskgen submit_task`"),
        ("submit_task[]", "taskgen submit_task"),
        ("`fail[reason]`", "`taskgen fail \"reason\"`"),
        ("fail[reason]", "taskgen fail \"reason\""),
        ("Start with `new_scene[N]`.", "Start with `taskgen status` and then `taskgen new_scene N`."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _strip_pddl_from_guidance(guidance: str) -> str:
    guidance = guidance.replace(
        "- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.\n",
        "- `judge[]`: run non-PDDL validation and quality checks.\n",
    )
    guidance = guidance.replace(
        "- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.\n",
        "- PDDL solvability verification is disabled, but you MUST still write `problem_pddl` as the canonical goal format. Author `:goal` and optional `:goal-owners` normally.\n",
    )
    guidance = guidance.replace(
        "- Do not hand-edit `:objects`, `:init`, or `golden_trajectory`.\n",
        "- Do not hand-author `golden_trajectory`.\n",
    )
    guidance = guidance.replace(
        "- Use `K()` only for facts that matter for planning or coordination.\n",
        "",
    )
    guidance = guidance.replace(
        "- The outermost `K()` agent should not be able to directly observe the fact with no blocker.\n",
        "",
    )
    return guidance


def _strip_simulation_from_guidance(guidance: str) -> str:
    return guidance.replace(
        "- `judge[]`: run validation, planner checks, simulation checks when enabled, and quality checks.\n",
        "- `judge[]`: run validation, planner checks when enabled, and quality checks.\n",
    )


def _build_external_category_guidance(category: str, skip_pddl: bool) -> str:
    lines = ["## Category Rules", f"- Requested category: `{category.upper()}`."]
    if category == "cooperative":
        lines.extend(
            [
                "- All goals are shared.",
                "- Do not include `teams` or `team_secrets`.",
            ]
        )
        if not skip_pddl:
            lines.append("- Do not include `:goal-owners`.")
    elif category == "competitive":
        lines.extend(
            [
                "- Use exactly two teams: `team_0` and `team_1`.",
                "- Keep the public `task` neutral.",
                "- Teams must compete over incompatible outcomes, not independent races.",
            ]
        )
        if not skip_pddl:
            lines.append("- `problem_pddl :goal` must be a top-level `(or ...)` with exactly two mutually exclusive branches.")
            lines.append("- Use `:goal-owners` for team-owned goals with entries like `(team_0 (is_on_top bottle_1 table_10))`, not wrapper forms like `(team team_0 ...)`.")
    elif category == "mixed":
        lines.extend(
            [
                "- Public `task` covers only the shared objective.",
                "- Each relevant agent must have a hidden personal objective.",
            ]
        )
        if not skip_pddl:
            lines.append("- Put personal objectives in `:goal-owners` using entries like `(agent_0 (is_open cabinet_10))`, not `(personal agent_0 ...)`.")
    else:
        lines.append("- Pick the category that best fits the scene and obey its invariants.")
    return "\n".join(lines)


def _build_external_spec_guidance(
    *,
    task_file: str,
    working_dir: str,
    skip_pddl: bool,
    skip_evolution: bool,
    skip_test: bool,
) -> str:
    lines = [
        "## Working Files",
        f"- `{task_file}`: edit this task JSON.",
        f"- `{working_dir}/current_scene.json`: current scene after `taskgen new_scene`.",
        f"- `{working_dir}/template.json`: task structure reference.",
        f"- Commands already start in `{working_dir}`. Do not prefix every command with `cd {working_dir} &&`.",
    ]
    if not skip_evolution:
        lines.append(f"- `{working_dir}/sampled_tasks/SUMMARY.md`: compact seed-task fields. Read this first.")
        lines.append(f"- `{working_dir}/sampled_tasks/`: raw seed task JSONs for deeper inspection only when needed.")
    lines.extend(
        [
            "- `available_predicates.md`, `available_mechanics.md`, `available_actions.md`, `available_items.md`: inspect only when needed.",
            "",
            "## Hard Authoring Rules",
            "- Use exact scene IDs and only valid agent IDs returned by `taskgen new_scene`.",
            "- Remove placeholder text.",
            "- Secrets should state constraints, goals, and exact IDs; do not prescribe strategy.",
            "- Every mechanic must materially affect the task.",
            "- Do not hand-author `golden_trajectory`.",
            "- If `message_targets` is present, it already acts as a valid communication restriction.",
            "- Use canonical mechanic schema only:",
            "  `room_restriction` -> `restricted_rooms` + `for_agents`",
            "  `limited_bandwidth` -> `message_limits`",
            "  `restricted_communication` -> `allowed_targets`",
        ]
    )
    if skip_pddl:
        lines.append("- Write the natural-language task, secrets, and mechanics directly.")
    else:
        lines.extend(
            [
                "- Treat `problem_pddl` as machine-owned except for `:goal` and optional `:goal-owners`.",
                "- Do not hand-edit `:objects` or `:init`.",
                "- Use only predicates from `available_predicates.md`.",
            ]
        )
    if skip_test:
        lines.append("- Do not call `taskgen test_task` for this run.")
    return "\n".join(lines)


def _build_external_tom_guidance(skip_pddl: bool) -> str:
    lines = [
        "## Good ToM",
        "- The core task should require an agent's correct action choice to depend on another agent's private knowledge, access, or observation.",
        "- Good pattern: agent A cannot determine the right object, room, or target state until agent B observes or communicates it.",
        "- Bad pattern: agents can finish the physical goal independently and communication only reports what already happened.",
        "- Every essential agent should contribute distinct knowledge, access, or incentive.",
    ]
    if not skip_pddl:
        lines.extend(
            [
                "- Use `K()` only for facts that matter for planning or coordination.",
                "- The outermost `K()` agent should not be able to directly observe the fact with no blocker.",
            ]
        )
    return "\n".join(lines)


def _build_external_empirical_guidance(skip_test: bool) -> str:
    lines = [
        "## Empirical Solvability",
        "- Keep the physical execution short and direct. Prefer tasks that baseline/full-info can finish in roughly 6-10 turns.",
        "- Prefer one clean asymmetry over stacked brittle mechanics. One room/access blocker plus one decisive hidden fact is better than a long chain of dependencies.",
        "- Use actionable targets that runtime tools can find by exact ID. Avoid relying on vague aliases like 'display table' or hidden trigger objects whose exact runtime ID is hard to recover.",
        "- Avoid long cross-house transport chains unless that complexity is the core benchmark point.",
        "- If a task passes `judge` but fails `test_task`, simplify the physical core first before adding more ToM structure.",
    ]
    if not skip_test:
        lines.append("- `taskgen test_task` is the real execution gate: `judge` is not enough if baseline/full-info still cannot complete the task.")
    return "\n".join(lines)


def _build_external_checklist(skip_pddl: bool, skip_test: bool) -> str:
    lines = [
        "## Pre-Submit Checklist",
        "- The physical goal requires communication or partner modeling.",
        "- All referenced agents, objects, furniture, and rooms exist in the current scene or declared items.",
        "- Category fields are valid for the selected category.",
        "- Mechanics and secrets agree about actual constraints, but secrets do not explain the coordination plan.",
        "- No malformed bindings, missing required mechanic fields, or invalid message limits.",
    ]
    if not skip_pddl:
        lines.extend(
            [
                "- `problem_pddl` has a valid `:goal` and, when needed, valid `:goal-owners`.",
                "- `:objects` and `:init` were not hand-edited.",
            ]
        )
    if not skip_test:
        lines.append("- After `taskgen judge` passes, run `taskgen test_task` before submitting.")
    return "\n".join(lines)


def build_external_taskgen_prompt(
    *,
    working_dir: str,
    task_file: str,
    category: str,
    available_items: str,
    available_mechanics: str,
    available_predicates: str,
    action_descriptions: str,
    extra_sections: str,
    num_tasks: int,
    agents_min: int,
    agents_max: int,
    subtasks_min: int,
    subtasks_max: int,
    skip_steps: Optional[List[str]] = None,
) -> str:
    del available_items, available_mechanics, available_predicates, action_descriptions

    skip = set(skip_steps or [])
    skip_pddl = "pddl" in skip
    skip_evolution = "task-evolution" in skip
    skip_test = "test" in skip or skip_evolution

    constraints = USER_PROMPT_TEMPLATE.format(
        num_tasks=num_tasks,
        extra_sections=extra_sections,
        agents_min=agents_min,
        agents_max=agents_max,
        subtasks_min=subtasks_min,
        subtasks_max=subtasks_max,
    )
    constraints = _rewrite_react_tool_syntax(constraints)

    commands = [
        "## Required Commands",
        "- `taskgen status`",
        "- `taskgen new_scene N`",
        "- `taskgen new_scene N --keep`",
        "- `taskgen judge`",
    ]
    if not skip_test:
        commands.append("- `taskgen test_task`")
    commands.extend(
        [
            "- `taskgen submit_task`",
            "- `taskgen finish`",
            '- `taskgen fail "reason"`',
        ]
    )

    workflow = ["1. Run `taskgen status`."]
    if skip_pddl:
        workflow.append(
            f"2. Run `taskgen new_scene N` with `N` between {agents_min} and {agents_max}. Never use `1`. Use only the returned `valid_agent_ids` in mechanic bindings, secrets, message targets, and teams."
        )
        workflow.append(f"3. Edit `{task_file}`. Write the natural-language task, secrets, and mechanics.")
    else:
        workflow.append(
            f"2. Run `taskgen new_scene N` with `N` between {agents_min} and {agents_max}. Never use `1`. Use only the returned `valid_agent_ids` in mechanic bindings, secrets, message targets, and teams."
        )
        workflow.append(
            f"3. Edit `{task_file}`. Author `problem_pddl :goal` first, then make the natural-language fields and mechanics match it."
        )
    if not skip_evolution:
        workflow.insert(
            2,
            f"3. Read `{working_dir}/sampled_tasks/SUMMARY.md` first, then inspect at least 2 sampled seed tasks in `{working_dir}/sampled_tasks/`, including one matching the target category when possible. Reuse only structural patterns that look empirically solvable. Do not copy IDs directly.",
        )
        if skip_pddl:
            workflow[3] = f"4. Edit `{task_file}`. Write the natural-language task, secrets, and mechanics."
        else:
            workflow[3] = (
                f"4. Edit `{task_file}`. Author `problem_pddl :goal` first, then make the natural-language fields and mechanics match it."
            )
    next_step = len(workflow) + 1
    workflow.append(f"{next_step}. Run `taskgen judge`, fix issues, and repeat until it passes.")
    next_step += 1
    if not skip_test:
        workflow.append(f"{next_step}. Run `taskgen test_task`.")
        next_step += 1
    workflow.append(f"{next_step}. Run `taskgen submit_task`.")
    next_step += 1
    workflow.append(f"{next_step}. When you have submitted {num_tasks} tasks, run `taskgen finish`.")

    sections = [
        f"""You are generating multi-agent benchmark tasks in `{working_dir}`.

Use normal shell commands for inspection and file edits.
Use the repo-owned `taskgen` commands for scene loading, judging, testing, submission, and finish/fail.

{constraints}
""",
        _build_external_spec_guidance(
            task_file=task_file,
            working_dir=working_dir,
            skip_pddl=skip_pddl,
            skip_evolution=skip_evolution,
            skip_test=skip_test,
        ),
        "\n".join(commands),
        "## Workflow\n" + "\n".join(workflow),
        _build_external_category_guidance(category, skip_pddl),
        _build_external_tom_guidance(skip_pddl),
        _build_external_empirical_guidance(skip_test),
        _build_external_checklist(skip_pddl, skip_test),
        "## References\n"
        "- `available_predicates.md`: valid predicates and goal syntax.\n"
        "- `available_mechanics.md`: mechanic names and fields.\n"
        "- `available_actions.md`: supported runtime actions.\n"
        "- `available_items.md`: optional task-added items.\n"
        "- Avoid repeating the same mechanic stack or target pattern across tasks in this run.",
    ]

    skip_notice = ""
    if skip:
        skip_notice = (
            "\n\n## Removed Pipeline Components\n"
            f"The following pipeline components have been removed for this run via `--remove`: **{', '.join(sorted(skip))}**.\n"
            "Do not attempt to run or rely on these components.\n"
        )
        if skip_pddl:
            skip_notice += "- `pddl`: do not write or reference `problem_pddl`.\n"
        if "tom" in skip:
            skip_notice += "- `tom`: do not worry about tom_level computation.\n"
        if "simulation" in skip:
            skip_notice += "- `simulation`: simulator verification is skipped inside `taskgen judge`.\n"
        if "llm-council" in skip:
            skip_notice += "- `llm-council`: quality checks are auto-passed.\n"
        if skip_evolution:
            skip_notice += "- `task-evolution`: no seed tasks or calibration data are available.\n"
        if "test" in skip:
            skip_notice += "- `test`: skip `taskgen test_task` and go directly to submission.\n"

    prompt = "\n\n".join(section.strip() for section in sections if section.strip())
    return f"{prompt}{skip_notice}"
