# Benchmark Architecture

This file is the source of truth for the benchmark architecture. Keep it short and keep it current.

## Goal

EMTOM is a benchmark for embodied Theory of Mind. A task is good if success requires agents to act under asymmetric information, communicate, and reason about what other agents know.

## Pipeline

1. Explore a scene and discover useful mechanics.
2. Generate a task grounded in the current scene and mechanics.
3. Verify the task statically and with runtime checks.
4. Judge whether the task genuinely requires ToM reasoning.
5. Benchmark agents on the final task.

## Code Ownership

- `emtom/pddl/`: goal language, epistemic compilation, belief tracking, and solvability checks.
- `emtom/task_gen/`: task generation, validation, calibration, and submission gates.
- `emtom/runner/`: execution in the environment.
- `emtom/cli/`: stable command surfaces for operators and agents.
- `README.md`: brief setup and command entrypoints.
- `docs/*.md`: conceptual design and architecture. This is the single source of truth.

## Design Rules

- Keep one clear implementation path.
- Prefer direct data flow over hidden coupling.
- Treat verification as a hard gate, not a warning system.
- Treat higher-order knowledge conservatively. If the system cannot prove a `K()` goal, it should reject it.
- Update this file whenever the benchmark structure or invariants change.
