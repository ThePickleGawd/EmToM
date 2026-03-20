# EMTOM

EMTOM is an embodied Theory of Mind benchmark.

Use `./emtom/run_emtom.sh` for exploration, task generation, verification, judging, and benchmarking.

## Setup

- Install dependencies with [INSTALLATION.md](INSTALLATION.md).
- Read the benchmark architecture in [docs/benchmark-architecture.md](docs/benchmark-architecture.md).

## Quick Start

```bash
./emtom/run_emtom.sh explore --steps 30 --model gpt-5
./emtom/run_emtom.sh generate --model gpt-5.2 --target-model gpt-5.2 --seed-tasks-dir data/emtom/tasks
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --llm openai_chat --model gpt-5
./emtom/run_emtom.sh benchmark
```

Generation now uses the same loop for normal task creation and difficulty shaping: the generator selects seed tasks from the task pool for a target model, `new_scene` re-samples that seed context, and calibration stays in the standard pipeline.

## Environment

Create a `.env` file in the repo root with the provider credentials you need, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS Bedrock credentials.
