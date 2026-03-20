# Benchmark Architecture

This file is the source of truth for the benchmark architecture. Keep it short and keep it current.

## Goal

EMTOM is a benchmark for embodied Theory of Mind. A task is good if success requires agents to act under asymmetric information, communicate, and reason about what other agents know.

Runtime benchmark scoring separates:
- `functional_success`: physical and owned task completion under asymmetric information.
- `literal_tom_probe`: end-of-episode probes derived from `K()` goals that measure whether agents can explicitly report the predicate and entities they believe are known, or abstain with `unknown`.

Task generation should optimize for functional ToM, not just literal ToM:
- good tasks make success depend on adapting actions to partner-specific private information, access, incentives, or communication limits
- weak tasks only hide a fact and ask one agent to relay it

## Pipeline

1. Explore a scene and discover useful mechanics.
2. Generate a task grounded in the current scene and mechanics.
3. Verify the task statically and with runtime checks.
4. Judge whether the task genuinely requires ToM reasoning.
5. Benchmark agents on the final task.

## Campaigns

- Keep exactly one active benchmark campaign in `data/emtom/results/`.
- When benchmark semantics change enough to invalidate comparability, archive the active campaign into `data/emtom/results/archives/<campaign_id>/` before starting a new one.
- The visualizer should expose both the active campaign and archived campaigns, but never merge their leaderboards.
- Campaign reporting must keep `pass_rate` and `literal_tom_score` separate. They answer different questions and should be shown side by side, not collapsed into one metric.

## Benchmark Modes

- `standard`: task secrets are private and agents only observe normal benchmark channels.
- `baseline`: all task secrets are shared with all agents, and agents may read other agents' completed Thought+Action trajectories through a runtime benchmark tool.
- `full_info`: all task secrets are shared with all agents, and agents may read other agents' completed Observation+Thought+Action trajectories through a runtime benchmark tool.

## Code Ownership

- `emtom/pddl/`: goal language, runtime goal projection, epistemic compilation, and solvability checks.
- `emtom/task_gen/`: task generation, validation, calibration, and submission gates.
- `emtom/runner/`: execution in the environment.
- `emtom/cli/`: stable command surfaces for operators and agents.
- `README.md`: brief setup and command entrypoints.
- `docs/*.md`: conceptual design and architecture. This is the single source of truth.

## Design Rules

- Keep one clear implementation path.
- Prefer direct data flow over hidden coupling.
- Treat verification as a hard gate, not a warning system.
- Keep `problem_pddl` as the single authored source of epistemic structure.
- Keep benchmark mechanics authored once in `mechanic_bindings`; derive planner-only init facts like `is_restricted` from those bindings instead of duplicating them in `problem_pddl`.
- Keep the public `task` high-level and non-leaking; use exact scene IDs in `agent_secrets` and `team_secrets` for goal-critical targets so private grounding remains precise.
- Runtime task success ignores `K()` and uses the projected non-epistemic goal only.
- End-of-episode literal ToM probes are derived deterministically from `K()` formulas and reported separately from task success.
- Benchmark `percent_complete` should track the same success-relevant functional scope; mixed tasks may expose separate all-goal progress for diagnostics.
- Golden trajectories are physical-only and do not include epistemic-only communication steps.
- Update this file whenever the benchmark structure or invariants change.
