# EMTOM: Embodied Theory of Mind Benchmark

EMTOM tests whether AI agents can reason about others' mental states in embodied environments. Tasks are designed to require agents to track what others know, communicate discoveries, and coordinate based on asymmetric information.

## Installation

See [INSTALLATION.md](INSTALLATION.md) for conda setup.

## Quick Start

```bash
# Explore the environment using GPT-5
./emtom/run_emtom.sh explore --steps 30 --model gpt-5

# Explore using Claude Sonnet (AWS Bedrock)
./emtom/run_emtom.sh explore --steps 30 --model sonnet

# Generate a task using OpenAI
./emtom/run_emtom.sh generate --model gpt-5.2

# Generate a task using Claude (AWS Bedrock)
./emtom/run_emtom.sh generate --model sonnet

# Run benchmark on generated tasks
./emtom/run_emtom.sh benchmark

# Evaluate a task for Theory of Mind requirements
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --model gpt-5
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
./emtom/run_emtom.sh explore --steps 30 --model gpt-5
./emtom/run_emtom.sh explore --num-agents 3 --model sonnet
./emtom/run_emtom.sh explore --steps 50 --model opus --agent-type human
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--steps N` | Number of exploration steps | 20 |
| `--model MODEL` | LLM model (see table below) | gpt-5.2 |
| `--num-agents N` | Number of agents (2-5) | 2 |
| `--agent-type TYPE` | `robot` or `human` | robot |

**Supported Models:** Same as Task Generation (see table below).

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

| Provider | Model Aliases | Full Model ID |
|----------|---------------|---------------|
| `openai_chat` | `gpt-5`, `gpt5` | gpt-5 |
| `openai_chat` | `gpt-5-mini`, `gpt5-mini` | gpt-5-mini |
| `openai_chat` | `gpt-5.1`, `gpt5.1` | gpt-5.1 |
| `openai_chat` | `gpt-5.2`, `gpt5.2` | gpt-5.2 |
| `bedrock_claude` | `sonnet`, `sonnet-4.5` | Claude Sonnet 4.5 |
| `bedrock_claude` | `haiku`, `haiku-4.5` | Claude Haiku 4.5 |
| `bedrock_claude` | `opus`, `opus-4.5` | Claude Opus 4.5 |
| `bedrock_claude` | `qwen3-80b`, `qwen3-next` | Qwen3 Next 80B A3B |
| `bedrock_claude` | `qwen3-vl`, `qwen3-vl-235b` | Qwen3 VL 235B A22B |
| `bedrock_claude` | `kimi-k2`, `kimi-thinking` | Kimi K2 Thinking |
| `bedrock_claude` | `ministral-8b` | Ministral 3 8B |
| `bedrock_claude` | `ministral-14b` | Ministral 3 14B |
| `bedrock_claude` | `mistral-large` | Mistral Large 3 675B |

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

Evaluate whether a task genuinely requires Theory of Mind reasoning. The judge is grounded with real system capabilities (actions, mechanics, items, predicates) to provide actionable improvement suggestions.

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

**Supported Models:** Same as Task Generation (see table above).

**Evaluation Criteria:**
| Criterion | Description | Auto-Fail |
|-----------|-------------|-----------|
| Information Asymmetry | Agents have different knowledge/access | No |
| Interdependence | Agents must rely on each other | No |
| Mental State Reasoning | Agents must track others' beliefs/knowledge | Yes (-1.0) |
| Coordination Requirement | Success requires joint action | No |

Tasks must score ≥0.7 overall and ≥0.5 on each criterion to pass. Mental state reasoning can trigger an **automatic fail** (-1.0) if agents have identical information or one agent can complete the task alone.

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

EMTOM uses a layered mechanic system that transforms actions and state to create Theory of Mind challenges. All mechanics are stateless handlers that operate on `EMTOMGameState`.

### State Transform Mechanics
Modify how actions affect object states.

| Mechanic | Description |
|----------|-------------|
| `inverse_state` | Actions have opposite effects (open→close, turn_on→turn_off). Objects marked as "inverse" behave contrary to the agent's intent. |
| `conditional_unlock` | Targets remain locked until a trigger object is interacted with. Enables prerequisite-based puzzles. |

### Hidden Mapping Mechanics
Create non-obvious connections between objects.

| Mechanic | Description |
|----------|-------------|
| `remote_control` | Actions on a trigger object affect a different target object. The mapping is hidden from agents. |
| `state_mirroring` | Paired objects mirror each other's state changes. Opening one container opens its paired counterpart. |

### Belief Tracking Mechanics (Theory of Mind)
Require agents to track what others know vs. reality.

| Mechanic | Description |
|----------|-------------|
| `location_change` | Objects are moved while certain agents are absent. Creates false belief scenarios (Sally-Anne test pattern). Absent agents believe objects are in original locations. |
| `container_swap` | Contents of containers are swapped while agents are away. Agents who didn't witness the swap have incorrect beliefs about container contents. |
| `state_change_unseen` | Object properties change (locked→unlocked, on→off) while agents aren't observing. Creates belief-reality gaps about object states. |

### Information Asymmetry Mechanics
Control when and what information agents receive.

| Mechanic | Description |
|----------|-------------|
| `delayed_information` | Information is revealed to specific agents only after N steps. Creates temporal knowledge gaps that require reasoning about what others know at different times. |

### Communication Constraint Mechanics
Limit or distort inter-agent communication.

| Mechanic | Description |
|----------|-------------|
| `limited_bandwidth` | Agents have a maximum number of messages they can send. Forces strategic communication decisions. |
| `delayed_messages` | Messages take N steps to be delivered. Recipients must reason about sender's state at send time vs. current time. |
| `noisy_channel` | Messages may be corrupted (character substitution) or dropped entirely based on noise parameters. Agents must handle unreliable communication. |

### Coordination Mechanics
Require explicit multi-agent coordination.

| Mechanic | Description |
|----------|-------------|
| `hidden_agenda` | Agents have secret goals that may conflict with the public objective or other agents' goals. Tests ability to infer others' motivations from behavior. |
| `simultaneous_action` | Certain actions require multiple agents to act together within a time window. Agents must coordinate timing without explicit synchronization. |

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
