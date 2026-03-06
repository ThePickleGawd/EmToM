# EMTOM

EMTOM is an embodied Theory of Mind benchmark.

Use `./emtom/run_emtom.sh` for exploration, task generation, verification, judging, and benchmarking.

## Setup

- Install dependencies with [INSTALLATION.md](INSTALLATION.md).
- Read the benchmark architecture in [docs/benchmark-architecture.md](docs/benchmark-architecture.md).

## Quick Start

```bash
./emtom/run_emtom.sh explore --steps 30 --model gpt-5
./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5.2
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --llm openai_chat --model gpt-5
./emtom/run_emtom.sh benchmark
```

## Environment

Create a `.env` file in the repo root with the provider credentials you need, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or AWS Bedrock credentials.
