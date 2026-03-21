# EMTOM

EMTOM is an embodied Theory of Mind benchmark.

Use `./emtom/run_emtom.sh` for exploration, task generation, verification, judging, and benchmarking.

## Setup

- Install dependencies with [INSTALLATION.md](INSTALLATION.md).
- Read the benchmark architecture in [docs/benchmark-architecture.md](docs/benchmark-architecture.md).

## Quick Start

```bash
./emtom/run_emtom.sh explore --steps 30 --model gpt-5
./emtom/run_emtom.sh generate --task-gen-agent mini --model gpt-5.2 --target-model gpt-5.2 --seed-tasks-dir data/emtom/tasks
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --llm openai_chat --model gpt-5
./emtom/run_emtom.sh benchmark
```

Generation now launches an external SWE-agent CLI (`mini`, `claude`, or `codex`) inside a repo-local workspace under `tmp/task_gen/`. The repo owns the task-generation prompt plus the stable `taskgen` command surface (`new_scene`, `judge`, `verify_golden_trajectory`, `test_task`, `submit_task`, `finish`), while seed selection and calibration stay in the same pipeline.

## Task Generation

Task generation uses one loop:

1. `./emtom/run_emtom.sh generate ...` creates a repo-local workspace under `tmp/task_gen/`.
2. The runner picks a target model and target pass rate, reads calibration from the task pool, and chooses a required K-level for the current task.
3. The runner populates `sampled_tasks/` from `--seed-tasks-dir` using the logical sampled-task mix:
   by default `80% fail / 20% pass` on `--target-model`.
4. `taskgen new_scene N` creates a fresh `working_task.json` from the blank template and fills in the exact requested number of agents.
5. The external agent inspects `sampled_tasks/` for inspiration, then edits `working_task.json` and iterates through `taskgen judge`, `taskgen verify_golden_trajectory`, and `taskgen test_task`.
6. `taskgen test_task` runs `standard` and `baseline`, then writes calibration back into the task JSON.
7. `taskgen submit_task` saves the task, then advances the loop by picking the next K-level. Sampled-task selection is inspiration only; the authored task itself always starts blank.
8. After the requested number of tasks have been submitted, the agent runs `taskgen finish`.

Important generation flags:

- `--target-model`: model whose calibration defines pass/fail seed buckets.
- `--target-pass-rate`: desired dataset pass rate for that target model.
- `--seed-tasks-dir`: task pool used to populate `sampled_tasks/`.
- `--seed-pass-ratio` and `--seed-fail-ratio`: logical pass/fail sampled-task mix. Defaults are `0.20` and `0.80`.

## Bulk Generation

`./emtom/run_emtom.sh generate` remains the main entry point.

For now, parallel generation is launched through the helper wrapper below:

```bash
./emtom/bulk_generate.sh --num-tasks 8 --task-gen-agent mini --model gpt-5.2
```

- `./emtom/bulk_generate.sh --num-tasks N` means `N` total submitted tasks for the whole bulk run.
- `--per-gpu N` controls concurrency. The launcher starts up to `num_gpus * per_gpu` fixed workers.
- The bulk wrapper divides `N` across those workers up front, and each worker runs `./emtom/run_emtom.sh generate --num-tasks <assigned_count>` in its own taskgen workspace under `tmp/task_gen/`.
- Workers are not respawned. If a worker fails early, the bulk run ends short of the requested total.
- Run-level logs and live visualizer data are written under `outputs/generations/<run_id>/`.

Useful bulk commands:

```bash
./emtom/bulk_generate.sh --dry-run --num-tasks 8 --task-gen-agent mini --model gpt-5.2
./emtom/bulk_generate.sh --per-gpu 1 --num-tasks 8 --task-gen-agent mini --model gpt-5.2
```

## Environment

Create a `.env` file in the repo root with the provider credentials you need, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS Bedrock credentials. If you use `--task-gen-agent mini`, install `mini-swe-agent` in the main operator environment first so the `mini` executable is available on `PATH`. Each task-generation run executes its shell commands inside its own isolated repo-local virtualenv under `tmp/task_gen/<run_id>/.venv`.
