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
2. Select sampled task examples from the existing task pool for a target benchmark model, using an explicit logical pass/fail mix over that model's calibrated results.
3. Generate a task grounded in the current scene and mechanics, starting from the blank task template and using the sampled examples only as inspiration.
4. Verify the task statically and with runtime checks.
5. Judge whether the task genuinely requires ToM reasoning.
6. Benchmark agents on the final task in both `standard` and `baseline`, using `standard` for calibration and `baseline` as the full-info solvability check.

Task generation runs through an external SWE-agent CLI (`mini`, `claude`, or `codex`) inside a repo-local workspace under `tmp/task_gen/`. The agent executable may come from the operator environment, but each task-generation run gets its own sandbox environment in `tmp/task_gen/<run_id>/.venv` so parallel runs stay isolated. The repo provides the prompt, sampled seed context, and a stable `taskgen` command surface for `new_scene`, `judge`, `verify_golden_trajectory`, `test_task`, `submit_task`, and `finish`.

`tmp/task_gen/` is workspace-only. All generation logs and visualizer-facing artifacts live under `outputs/generations/<run_id>/`, including the run manifest, per-worker status snapshots, normalized EMTOM event logs, backend-native agent traces such as `agent_trace.json`, and any bulk-launcher stdout logs. The visualizer reads those files live in dev through filesystem-backed endpoints; it does not rely on a generated generation-data snapshot.

There is no separate evolution pipeline. Difficulty shaping happens inside normal task generation:
- the sampled-task selector uses target-model calibration data and an explicit pass/fail mix, defaulting to 80% failed examples and 20% passed examples
- `new_scene` always creates `working_task.json` from the blank template with the requested number of agents
- sampled examples are inspiration only; they are not loaded directly into the authored task

## Campaigns

- Keep exactly one active benchmark campaign in `data/emtom/results/`.
- When benchmark semantics change enough to invalidate comparability, archive the active campaign into `data/emtom/results/archives/<campaign_id>/` before starting a new one.
- The visualizer should expose both the active campaign and archived campaigns, but never merge their leaderboards.
- Campaign reporting must keep `pass_rate` and `literal_tom_score` separate. They answer different questions and should be shown side by side, not collapsed into one metric.

## Benchmark Modes

- `standard`: task secrets are private and agents only observe normal benchmark channels.
- `baseline`: all task secrets are shared with all agents, and agents may read other agents' completed Thought+Action trajectories through a runtime benchmark tool.
- `full_info`: all task secrets are shared with all agents, and agents may read other agents' completed Observation+Thought+Action trajectories through a runtime benchmark tool.
- All benchmark modes must still run with partial observability and per-agent asymmetric world graphs. Baseline/full_info change secret and trace access, not raw world-state visibility.
- Under partial observability, each agent's world graph must be private: only that agent's own observations may add or update entities. Communication may inform planning, but it must not directly mutate the recipient's world graph.

## Task Generation Gates

- `verify_golden_trajectory` remains the canonical deterministic solvability gate. It proves the authored task spec is functionally solvable under the planner/runtime semantics.
- Judge-time ToM evidence must come from the strict Fast Downward proof path. Structural or syntactic fallback metadata is not valid submission evidence.
- `test_task` now runs both `standard` and `baseline` in parallel.
- Dataset difficulty calibration uses the `standard` result only, with a target pass rate of 20% by default for the current target model.
- Calibration and sampled-task selection ignore `tom_level = 0` tasks. New submissions with `tom_level < 1` must be rejected.
- The `test_task` acceptance gate should use the current calibrated pass/fail counts and accept only the next `standard` outcome that moves the dataset closer to the target pass rate.
- `baseline` does not replace the planner/golden-trajectory check; it is an additional empirical check that the task becomes solvable when private information is removed.
- Submitted benchmark tasks must stay grounded in a real dataset `scene_id` and `episode_id`. Synthetic fallback scenes are allowed for lightweight authoring environments, but they must be rejected before submission and benchmark runs.

## Code Ownership

- `emtom/pddl/`: goal language, runtime goal projection, epistemic compilation, and solvability checks.
- `emtom/task_gen/`: seed selection, task generation, validation, calibration, and submission gates.
- `emtom/runner/`: execution in the environment.
- `emtom/cli/`: stable command surfaces for operators and agents.
- `README.md`: brief setup and command entrypoints.
- `docs/*.md`: conceptual design and architecture. This is the single source of truth.

## Design Rules

- Keep one clear implementation path.
- Prefer direct data flow over hidden coupling.
- Treat verification as a hard gate, not a warning system.
- Task authoring currently supports only these mechanics: `room_restriction`, `limited_bandwidth`, `restricted_communication`, `remote_control`, `state_mirroring`, and `inverse_state`.
- Task-added items are temporarily hidden from authoring. Do not rely on `items`, `locked_containers`, or `UseItem` in newly generated benchmark tasks.
- Keep `problem_pddl` as the single authored source of epistemic structure and goals.
- Generate `problem_pddl :objects` and `:init` deterministically from the loaded scene snapshot and mechanic bindings instead of hand-authoring scene state.
- Keep benchmark mechanics authored once in `mechanic_bindings`; derive all planner-only mechanic init facts from those bindings instead of duplicating them in `problem_pddl`.
- Keep the public `task` high-level and non-leaking; use exact scene IDs in `agent_secrets` and `team_secrets` only for goal-critical facts that the agent actually knows or observed.
- `agent_secrets` should contain positive private facts, constraints, and private objectives only. Do not add ignorance lines like 'you do not know ...', self-intro boilerplate, or epistemic coaching like 'By the end, you must be confident ...'.
- When an object's identity or location is the hidden fact, do not name its exact runtime object ID in the public `task` or in any secret for an agent who does not already know that fact. Reserve exact IDs for the agents who actually know or observed that fact, plus `problem_pddl`.
- Runtime task success ignores `K()` and uses the projected non-epistemic goal only.
- `verify-pddl`, deterministic planning, and golden trajectory verification all solve the same projected non-epistemic functional goal.
- Mechanic predicates such as `is_inverse`, `controls*`, `mirrors*`, `requires_item`, `unlocks`, `is_restricted`, and communication wiring are init-only support facts. They must never appear in `pddl_goal`, including inside `K()`.
- The deterministic planner is the canonical mechanic implementation path. Runtime handlers and compiled planner facts must agree on mechanic semantics.
- End-of-episode literal ToM probes are derived deterministically from `K()` formulas and reported separately from task success.
- Benchmark `percent_complete` should track the same success-relevant functional scope; mixed tasks may expose separate all-goal progress for diagnostics.
- Golden trajectories are physical-only and do not include epistemic-only communication steps.
- Update this file whenever the benchmark structure or invariants change.
