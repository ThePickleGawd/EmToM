# EMTOM: Embodied Theory of Mind Benchmark

EMTOM tests whether AI agents can reason about others' mental states in embodied environments. Tasks are designed to require agents to track what others know, communicate discoveries, and coordinate based on asymmetric information.

## Installation

See [INSTALLATION.md](INSTALLATION.md) for conda setup.

## Quick Start

```bash
# Generate a task using OpenAI
./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5.2

# Generate a task using Claude (AWS Bedrock)
./emtom/run_emtom.sh generate --llm bedrock_claude --model sonnet

# Run benchmark on generated tasks
./emtom/run_emtom.sh benchmark

# Evaluate a task for Theory of Mind requirements
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --llm openai_chat --model gpt-5
```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# For OpenAI
OPENAI_API_KEY=your-openai-key

# For AWS Bedrock Claude
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

## Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. EXPLORATION    │────▶│  2. TASK GENERATION │────▶│    3. BENCHMARK     │
│                     │     │                     │     │                     │
│  LLM explores env   │     │  Agent creates &    │     │  Multi-agent eval   │
│  discovers mechanics│     │  validates tasks    │     │  on generated tasks │
│                     │     │                     │     │                     │
│  Output: trajectory │     │  Output: task.json  │     │  Output: metrics    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │    ToM JUDGE        │
                            │                     │
                            │  Validates tasks    │
                            │  require ToM        │
                            └─────────────────────┘
```

---

## Commands

### 1. Exploration

Discover how mechanics work through LLM-guided exploration.

```bash
./emtom/run_emtom.sh explore --steps 30
./emtom/run_emtom.sh explore --num-agents 3 --agent-type human
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--steps N` | Number of exploration steps | 20 |
| `--num-agents N` | Number of agents (2-5) | 2 |
| `--agent-type TYPE` | `robot` or `human` | robot |

---

### 2. Task Generation

Create benchmark tasks using an LLM agent that iteratively designs and tests tasks.

```bash
./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5.2
./emtom/run_emtom.sh generate --llm bedrock_claude --model sonnet --num-tasks 5
./emtom/run_emtom.sh generate --llm openai_chat --model gpt-5 --query "A task using the radio"
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--llm PROVIDER` | LLM provider (required) | - |
| `--model MODEL` | Model name (required) | - |
| `--num-tasks N` | Number of tasks to generate | 1 |
| `--subtasks N` | Steps per task | 3 |
| `--max-iterations N` | Max agent iterations | 100 |
| `--query "TEXT"` | Seed query to guide generation | - |
| `--retry-verification FILE` | Retry with ToM judge suggestions | - |

**Supported Models:**

| Provider | Models |
|----------|--------|
| `openai_chat` | `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.2` |
| `bedrock_claude` | `sonnet`, `haiku`, `opus` |

**How it works:**
1. Loads a random scene from the PARTNR dataset
2. LLM agent designs a task using available objects and mechanics
3. Task is tested in simulation to verify it's completable
4. ToM judge validates the task requires Theory of Mind reasoning
5. If validation fails, agent retries with suggestions

**Agent Tools:**
| Tool | Purpose |
|------|---------|
| `bash` | Read files, edit `working_task.json` |
| `test_task` | Run task in simulation, get completion metrics |
| `verify_golden_trajectory` | Verify the golden solution works |
| `judge_tom` | Evaluate task for ToM requirements |
| `submit_task` | Save validated task to output |

---

### 3. ToM Judge

Evaluate whether a task genuinely requires Theory of Mind reasoning.

```bash
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --llm openai_chat --model gpt-5
./emtom/run_emtom.sh judge --task my_task.json --llm bedrock_claude --model sonnet --no-auto-retry
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--task FILE` | Task file to evaluate (required) | - |
| `--llm PROVIDER` | LLM provider (required) | - |
| `--model MODEL` | Model name (required) | - |
| `--threshold N` | Score threshold for passing | 0.7 |
| `--no-auto-retry` | Don't auto-retry on failure | false |

**Evaluation Criteria:**
| Criterion | Description |
|-----------|-------------|
| Information Asymmetry | Agents have different knowledge/access |
| Interdependence | Agents must rely on each other |
| Mental State Reasoning | Agents must track others' beliefs |
| Coordination Requirement | Success requires joint action |

Tasks must score above the threshold on all criteria to pass. A score of -1.0 on mental state reasoning triggers an automatic fail.

---

### 4. Benchmark

Run multi-agent evaluation on generated tasks.

```bash
./emtom/run_emtom.sh benchmark
./emtom/run_emtom.sh benchmark --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh benchmark --max-sim-steps 1000 --num-agents 4
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--task FILE` | Specific task file | most recent |
| `--max-sim-steps N` | Max simulation steps | 200000 |
| `--max-llm-calls N` | Max LLM calls per agent | 20 |
| `--num-agents N` | Number of agents (2-5) | 2 |
| `--agent-type TYPE` | `robot` or `human` | robot |

---

### 5. Human Test Mode

Interactive testing with manual command input.

```bash
./emtom/run_emtom.sh test --task data/emtom/tasks/my_task.json
./emtom/run_emtom.sh test --mechanics inverse_state remote_control
./emtom/run_emtom.sh test --llm-agents agent_1  # Make agent_1 LLM-controlled
```

**Available Actions:**
- `Navigate[target]`, `Open[target]`, `Close[target]`
- `Pick[target]`, `Place[target]`, `Use[target]`
- `Inspect[target]`, `Communicate[message]`

**Commands:** `status`, `mechanics`, `history`, `skip`, `quit`, `help`

---

## Task Structure

Generated tasks are saved to `data/emtom/tasks/` as JSON:

```json
{
  "task_id": "task_001",
  "task_name": "Hidden Radio Message",
  "task_description": "Agent 0 must find and relay a radio message to Agent 1",
  "public_goal": "Coordinate to complete the objective",
  "subtasks": [
    {
      "description": "Agent 0 finds the radio",
      "success_criteria": {...}
    }
  ],
  "mechanic_bindings": [
    {
      "mechanic_type": "inverse_state",
      "target": "drawer_1"
    }
  ],
  "agent_secrets": {
    "agent_0": ["The drawer opens when you try to close it"],
    "agent_1": []
  },
  "golden_trajectory": [...]
}
```

---

## Mechanics

| Mechanic | Effect |
|----------|--------|
| `inverse_state` | Actions have opposite effects (open→close) |
| `remote_control` | Actions affect a different object |
| `counting_state` | Object needs N interactions to respond |

---

## Directory Structure

```
emtom/
├── run_emtom.sh           # Main entry point
├── task_gen/              # Task generation
│   ├── runner.py          # Generation entry point
│   ├── agent.py           # ReAct agent with tools
│   ├── tom_judge.py       # ToM validation
│   └── judge_cli.py       # Standalone judge CLI
├── exploration/           # Environment exploration
├── examples/              # Runner scripts
│   ├── run_habitat_exploration.py
│   ├── run_habitat_benchmark.py
│   └── run_human_test.py
├── mechanics/             # Mechanic handlers
└── actions/               # Custom EMTOM actions

habitat_llm/
├── llm/                   # LLM implementations
│   ├── openai_chat.py     # OpenAI provider
│   └── bedrock_claude.py  # AWS Bedrock Claude
└── conf/                  # Hydra configs
```

---

## Outputs

All outputs are saved to `outputs/emtom/` with timestamps:

- `YYYY-MM-DD_HH-MM-SS-exploration/` - Exploration trajectories and videos
- `YYYY-MM-DD_HH-MM-SS-generate/` - Generation logs
- `YYYY-MM-DD_HH-MM-SS-benchmark/` - Benchmark results
- `YYYY-MM-DD_HH-MM-SS-judge/` - ToM verification results

Generated tasks are saved to `data/emtom/tasks/`.
