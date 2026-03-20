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

## Environment

Create a `.env` file in the repo root with the provider credentials you need, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS Bedrock credentials. If you use `--task-gen-agent mini`, install `mini-swe-agent` in the main operator environment first so the `mini` executable is available on `PATH`. Each task-generation run executes its shell commands inside its own isolated repo-local virtualenv under `tmp/task_gen/<run_id>/.venv`.
