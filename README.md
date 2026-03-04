# EMTOM: Embodied Theory of Mind Benchmark

EMTOM tests whether AI agents can reason about others' mental states in embodied environments. Tasks are designed to require agents to track what others know, communicate discoveries, and coordinate based on asymmetric information.

## Installation

See [INSTALLATION.md](INSTALLATION.md) for conda setup.

## Quick Start

```bash
# Explore the environment using GPT-5
./emtom/run_emtom.sh explore --steps 30 --model gpt-5

# Explore using Claude Sonnet (Anthropic API or AWS Bedrock)
./emtom/run_emtom.sh explore --steps 30 --model sonnet

# Generate a task using OpenAI
./emtom/run_emtom.sh generate --model gpt-5.2

# Generate a task using Claude
./emtom/run_emtom.sh generate --model sonnet

# Run benchmark on generated tasks
./emtom/run_emtom.sh benchmark

# Evaluate a task for Theory of Mind requirements
./emtom/run_emtom.sh judge --task data/emtom/tasks/my_task.json --model gpt-5

# Verify a task by executing its golden trajectory in simulator
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json

# Fast static preflight (no Habitat/GPU)
./emtom/run_emtom.sh verify-static --task data/emtom/tasks/my_task.json
```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# For OpenAI
OPENAI_API_KEY=your-openai-key

# For Anthropic Claude (preferred for sonnet/haiku/opus when set)
ANTHROPIC_API_KEY=your-anthropic-key

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
./emtom/run_emtom.sh generate --llm anthropic_claude --model sonnet --num-tasks 5
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
| `--seed-task FILE` | Use existing task JSON as seed instead of blank template | - |
| `--retry-verification FILE` | Retry with ToM judge suggestions | - |

**Supported Models:**

*OpenAI (provider: `openai_chat`, requires `OPENAI_API_KEY`):*

| Model | Aliases | Model ID |
|-------|---------|----------|
| GPT-5 | `gpt-5`, `gpt5` | `gpt-5` |
| GPT-5 Mini | `gpt-5-mini`, `gpt5-mini` | `gpt-5-mini` |
| GPT-5.1 | `gpt-5.1`, `gpt5.1` | `gpt-5.1` |
| GPT-5.2 | `gpt-5.2`, `gpt5.2` | `gpt-5.2` |

*Anthropic API (provider: `anthropic_claude`, requires `ANTHROPIC_API_KEY`):*

| Model | Aliases | Model ID |
|-------|---------|----------|
| Claude Sonnet 4.5 | `sonnet`, `sonnet-4.5`, `sonnet4.5` | `claude-sonnet-4-5-20250929` |
| Claude Haiku 4.5 | `haiku`, `haiku-4.5`, `haiku4.5` | `claude-haiku-4-5-20251001` |
| Claude Opus 4.5 | `opus`, `opus-4.5`, `opus4.5` | `claude-opus-4-5-20251101` |

*AWS Bedrock (provider: `bedrock_claude`, requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`):*

When using alias models (`sonnet`, `haiku`, `opus`) with `./emtom/run_emtom.sh --model ...`,
the provider auto-selects `anthropic_claude` if `ANTHROPIC_API_KEY` is set, otherwise `bedrock_claude`.

| Model | Aliases | Model ID |
|-------|---------|----------|
| Claude Sonnet 4.5 | `sonnet`, `sonnet-4.5`, `sonnet4.5` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Claude Haiku 4.5 | `haiku`, `haiku-4.5`, `haiku4.5` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude Opus 4.5 | `opus`, `opus-4.5`, `opus4.5` | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| **Qwen (Alibaba)** | | |
| Qwen3 Next 80B A3B | `qwen3-80b`, `qwen3-next`, `qwen3-next-80b` | `qwen.qwen3-next-80b-a3b` |
| Qwen3 VL 235B A22B | `qwen3-vl`, `qwen3-vl-235b` | `qwen.qwen3-vl-235b-a22b` |
| **Kimi (Moonshot)** | | |
| Kimi K2 Thinking | `kimi-k2`, `kimi-thinking`, `kimi-k2-thinking` | `moonshot.kimi-k2-thinking` |
| **Mistral** | | |
| Ministral 3 8B | `ministral-8b`, `ministral-3-8b` | `mistral.ministral-3-8b-instruct` |
| Ministral 3 14B | `ministral-14b`, `ministral-3-14b` | `mistral.ministral-3-14b-instruct` |
| Mistral Large 3 675B | `mistral-large`, `mistral-large-3` | `mistral.mistral-large-3-675b-instruct` |

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
./emtom/run_emtom.sh judge --task my_task.json --llm anthropic_claude --model sonnet --no-auto-retry
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

### 4. Golden Trajectory Verification (Runtime)

Execute a task's `golden_trajectory` step-by-step in simulator and check success conditions at the end.

Use this when debugging mechanics, state-tracking, or simulator behavior. This is the recommended path for agent-driven triage with Claude/Codex.

```bash
# Verify one task end-to-end in Habitat
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json

# Save structured verification output
./emtom/run_emtom.sh verify --task data/emtom/tasks/my_task.json \
  --report-file /tmp/my_task_verify.json
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--task FILE` | Task file to execute (required) | - |
| `--report-file FILE` | Write JSON verification report | autogenerated `/tmp/...` |
| `--output-dir DIR` | Working dir for Hydra verify outputs | `/tmp` |
| `--agent-type TYPE` | `robot` or `human` config selection | `robot` |

The report includes:
- `valid` pass/fail
- executed per-step action results
- failure reason (if any)
- fallback error + log path if simulator exits before producing a result

---

### 5. Static Task Verification (No Habitat)

Run fast structural checks without launching simulator:
- action syntax and allowed actions
- trajectory structure and per-agent consistency
- unsupported predicates (e.g., XOR/custom unsupported predicate schemas)

```bash
# Verify one task quickly
./emtom/run_emtom.sh verify-static --task data/emtom/tasks/my_task.json

# Verify an entire directory and save JSON report
./emtom/run_emtom.sh verify-static --tasks-dir data/emtom/tasks \
  --report-file /tmp/task_static_verify.json
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--task FILE` | Single task file | - |
| `--tasks-dir DIR` | Verify all task JSON in directory | `data/emtom/tasks` |
| `--scene-data FILE` | Optional scene objects for stricter ID checks | - |
| `--strict-object-ids` | Fail on unknown object IDs in actions | off |
| `--report-file FILE` | Write JSON verification report | - |

---

### 6. Benchmark

Run multi-agent evaluation on generated tasks. By default, runs **all tasks** in `data/emtom/tasks/`.

```bash
# Run all tasks with default model (gpt-5.2)
./emtom/run_emtom.sh benchmark

# Run all tasks with Claude Sonnet
./emtom/run_emtom.sh benchmark --model sonnet

# Run a single specific task
./emtom/run_emtom.sh benchmark --task data/emtom/tasks/my_task.json

# Run all tasks without video (faster)
./emtom/run_emtom.sh benchmark --model sonnet --no-video

# Run only competitive tasks
./emtom/run_emtom.sh benchmark --category competitive

# Pit models by team in competitive tasks
./emtom/run_emtom.sh benchmark --team-model-map team_0=sonnet,team_1=gpt-5
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--task FILE` | Run single task file only | all tasks |
| `--model MODEL` | LLM model for agents (see table below) | gpt-5.2 |
| `--max-sim-steps N` | Max simulation steps | 200000 |
| `--max-llm-calls N` | Max LLM calls per agent | 5x golden trajectory |
| `--category TYPE` | Filter benchmark tasks by category (`cooperative`, `competitive`, `mixed`) | all categories |
| `--agent-type TYPE` | `robot` or `human` | robot |
| `--team-model-map MAP` | Team-to-model mapping for competitive tasks (`team_0=sonnet,team_1=gpt-5`) | none |
| `--no-video` | Disable video recording (faster) | false |

**Supported Models:** Same as Task Generation (see table above).

**Output:** Results are saved to `outputs/emtom/<timestamp>-benchmark/benchmark_summary.json` with per-task pass/fail status, overall pass rate, and per-task `team_model_mapping`/`agent_model_mapping`.

**Communication Metrics:** After each benchmark run, communication is automatically scored:
- **Secret leakage score** (0-1): Did agents dump their private secrets into the group chat?
- **Efficiency score** (0-1): Were messages strategic and concise vs redundant/wasteful? (LLM-judged via gpt-5.2)
- **Overall communication score**: Weighted combination of leakage and efficiency

These metrics appear in the benchmark output JSON under `communication_metrics`.

---

### Seed Task Workflow

Use `--seed-task` to iterate on existing tasks (e.g., elevating Theory of Mind depth):

```bash
# Generate a basic task
./emtom/run_emtom.sh generate --model gpt-5.2 --category cooperative

# Re-generate with higher ToM using the basic task as seed
./emtom/run_emtom.sh generate --model gpt-5.2 \
  --seed-task data/emtom/tasks/task_abc123.json \
  --query "Elevate to level 2 theory of mind with false beliefs"
```

---

### Task Authoring (CLI)

Create tasks manually using the step-by-step CLI workflow. This is how an LLM agent (or human) authors tasks outside of the fully automated `generate` pipeline.

#### Workflow

```
new-scene → edit task JSON → validate-task → verify-pddl → judge → verify → test-task → submit-task
```

#### Step 1: Load a Scene

```bash
# Load a random scene with 2 agents
./emtom/run_emtom.sh new-scene --agents 2 --output-dir /tmp/my_task

# Load with 3 agents
./emtom/run_emtom.sh new-scene --agents 3 --output-dir /tmp/my_task
```

This writes `current_scene.json` to the output directory containing rooms, furniture, objects, and agent spawn positions. All task authoring must reference entities from this scene file.

#### Step 2: Create the Task JSON

Copy the template and fill it in:

```bash
cp emtom/task_gen/template/template.json /tmp/my_task/working_task.json
```

Edit the task JSON to set:
- `category`: `cooperative`, `competitive`, or `mixed`
- `task`: Natural language description (no object IDs, no assigned roles)
- `num_agents`: Must match the scene
- `scene_id`, `episode_id`, `agent_spawns`: Copy from `current_scene.json`
- `pddl_goal`: PDDL formula using scene entity IDs (see Predicates below)
- `pddl_ordering`: Dependencies between goal conjuncts (required when >1 conjunct)
- `mechanic_bindings`: 1-2 mechanics (max 3) — `active_mechanics` is auto-derived
- `agent_secrets`: Per-agent private info (team membership, message limits, hints about mechanics)
- `agent_actions`: Per-agent allowed actions
- `items`: Inventory items placed inside containers
- `locked_containers`: Container → key item mappings

`golden_trajectory` is auto-generated by `verify` — leave it empty.

#### Step 3: Validate Structure

```bash
./emtom/run_emtom.sh validate-task --task /tmp/my_task/working_task.json \
  --scene-data /tmp/my_task/current_scene.json
```

Checks JSON structure, required fields, entity IDs against scene data, and mechanic binding validity.

#### Step 4: Verify PDDL Solvability

```bash
./emtom/run_emtom.sh verify-pddl --task /tmp/my_task/working_task.json
```

Parses the `pddl_goal`, checks all predicates are valid, verifies entity references, and computes the ToM depth (K-nesting level). Fix any errors before proceeding.

#### Step 5: Judge Quality

```bash
./emtom/run_emtom.sh judge --task /tmp/my_task/working_task.json
```

Multi-model council (Claude Opus + GPT-5.2) evaluates task quality on criteria including information asymmetry, interdependence, mental state reasoning, mechanic utilization, and PDDL solvability. Must score ≥0.65 overall and ≥0.5 per criterion.

#### Step 6: Verify Golden Trajectory

```bash
./emtom/run_emtom.sh verify --task /tmp/my_task/working_task.json
```

Deterministically regenerates the golden trajectory from the task spec and executes it in the Habitat simulator. Confirms the task is physically completable. Requires GPU.

#### Step 7: Calibrate Difficulty

```bash
./emtom/run_emtom.sh test-task --task /tmp/my_task/working_task.json
```

Runs LLM agents against the task to measure pass rate. Target: ~10% for hard tasks. Use `--test-model` to specify the test model.

#### Step 8: Submit

```bash
./emtom/run_emtom.sh submit-task --task /tmp/my_task/working_task.json \
  --output-dir data/emtom/tasks
```

Copies the finalized task to the task directory.

#### PDDL Predicates

| Predicate | Args | Description |
|-----------|------|-------------|
| `is_on_top` | `(obj, furniture)` | Object is on top of furniture |
| `is_inside` | `(obj, container)` | Object is inside container |
| `is_in_room` | `(entity, room)` | Entity is in room |
| `is_open` | `(furniture)` | Furniture is open |
| `is_closed` | `(furniture)` | Furniture is closed |
| `is_unlocked` | `(furniture)` | Furniture is unlocked |
| `has_item` | `(agent, item)` | Agent has inventory item |
| `K` | `(agent, predicate)` | Agent knows the predicate is true (epistemic) |

Epistemic nesting: `(K agent_0 (K agent_1 (is_inside key_1 cabinet_5)))` = K=2 (agent_0 knows that agent_1 knows).

---

### 7. Human Test Mode

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

### 8. Evolutionary Difficulty Pipeline

Automatically generates increasingly difficult tasks by iterating through a ladder of models. Each tier benchmarks the previous tier's tasks against a stronger model, then generates harder tasks informed by what failed and what passed.

```bash
# Full pipeline with default 6-model ladder
./emtom/run_emtom.sh evolve

# Custom model ladder, smaller run
./emtom/run_emtom.sh evolve \
  --model-ladder "ministral-3-8b,gpt-5-mini,sonnet,gpt-5.2" \
  --tasks-per-round 10 --seed-pool-size 20

# Resume a previous run
./emtom/run_emtom.sh evolve --resume outputs/evolve/2025-06-15_14-30-00
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--model-ladder` | Comma-separated models from weakest to strongest | `ministral-3-8b,haiku,gpt-5-mini,sonnet,gpt-5.1,gpt-5.2` |
| `--generator-model` | Model used to generate tasks | `gpt-5.2` |
| `--tasks-per-round` | Tasks generated per tier | `20` |
| `--seed-pool-size` | Number of initial seed tasks (tier 0) | `30` |
| `--max-workers` | Max parallel generation/benchmark processes | `50` |
| `--icl-total-examples` | In-context learning examples per generation | `10` |
| `--icl-failure-ratio` | Fraction of ICL examples that are failures | `0.9` |
| `--judge-threshold` | Judge quality threshold for generation (fixed across tiers) | `0.7` |
| `--target-pass-rate` | Skip harder generation when pass rate is at/below this % | `30.0` |
| `--seed-query` | Optional query for seed task generation | none |
| `--output-dir` | Base output directory | `outputs/evolve` |
| `--resume DIR` | Resume from existing output directory | - |

**How it works:**

```
Tier 0: Seed Pool
  Generate N simple tasks (subtasks: 2-4)

Tier 1: ministral-3-8b
  Benchmark seed tasks → sample failures → generate harder tasks (subtasks: 2-5)

Tier 2: haiku
  Benchmark tier 1 tasks → sample failures → generate harder tasks (subtasks: 3-7)

  ...progressively harder through the model ladder...

Tier N: gpt-5.2 (final)
  Benchmark tier N-1 tasks → final evaluation only (no generation)
```

Each tier:
1. **Benchmarks** the previous tier's tasks against the current model (parallelized, one process per task)
2. **Samples** ICL examples: mostly failures (90%) with a few passes (10%), annotated with `_benchmark_result`
3. **Generates** harder tasks guided by what failed, with subtask complexity ramping per tier (parallelized, one process per task)
4. **Checkpoints** progress to `state.json` for resumability

If a model's pass rate is already at/below the target (default 30%), the tier reuses the same tasks instead of generating harder ones.
When pass rate is above target, generation size is scaled (instead of always full `--tasks-per-round`) so evolution only generates as many harder tasks as needed.

**Subtask complexity per tier:**
| Tier | Subtasks Min-Max |
|------|-----------------|
| 0 (seed) | 2-4 |
| 1 | 2-5 |
| 2 | 3-7 |
| 3 | 3-10 |
| 4 | 4-12 |
| 5 | 5-15 |
| 6+ | 5-20 |

**Parallelization:** Both generation and benchmarking run in parallel (controlled by `--max-workers`). Each process gets its own log file under `<output_dir>/logs/`. Generation spawns up to `num_tasks` concurrent processes, with retries capped at 3x the target count.

**Output structure:**
```
outputs/evolve/<timestamp>/
├── config.json              # Run configuration
├── state.json               # Checkpoint for resumption
├── tier_0_seed/
│   └── tasks/               # Seed task JSONs
│       └── logs/            # Per-process generation logs
├── tier_1_ministral-3-8b/
│   ├── benchmark/           # Per-task benchmark results
│   │   └── logs/            # Per-process benchmark logs
│   ├── sampled_tasks/       # Annotated ICL examples (failed_1_45pct.json, passed_1.json)
│   ├── tasks/               # Generated harder tasks
│   ├── tier_metrics.json    # Pass rate summary
│   └── ...
├── tier_2_haiku/
│   └── ...
├── report.json              # Full results across all tiers
└── report.md                # Human-readable summary table
```

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
  "tom_level": 2,
  "tom_reasoning": "Agent 0 must reason about Agent 1's false belief about the drawer",
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
├── rl/                    # RL training module (see [emtom/rl/README.md](emtom/rl/README.md))
├── evolve/                # Evolution pipeline (run via: run_emtom.sh evolve)
├── task_gen/              # Task generation
│   ├── runner.py          # Generation entry point
│   ├── agent.py           # ReAct agent with tools
│   ├── tom_judge.py       # ToM validation
│   └── judge_cli.py       # Standalone judge CLI
├── evolve/                # Evolutionary difficulty pipeline
│   ├── orchestrator.py    # Main loop: seed → tier 1 → ... → tier N
│   ├── config.py          # EvolutionConfig dataclass
│   ├── benchmark_wrapper.py  # Parallel benchmark runner + result parser
│   ├── icl_sampler.py     # Prepare annotated ICL examples from benchmark results
│   └── report.py          # Generate report.json and report.md
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
│   ├── anthropic_claude.py # Anthropic Claude API
│   └── bedrock_claude.py   # AWS Bedrock Claude
└── conf/                  # Hydra configs
```

---

## Outputs

All outputs are saved to `outputs/emtom/` with timestamps:

- `YYYY-MM-DD_HH-MM-SS-exploration/` - Exploration trajectories and videos
- `YYYY-MM-DD_HH-MM-SS-generate/` - Generation logs
- `YYYY-MM-DD_HH-MM-SS-benchmark/` - Benchmark results
- `YYYY-MM-DD_HH-MM-SS-judge/` - ToM verification results
- `evolve/YYYY-MM-DD_HH-MM-SS/` - Evolution pipeline runs (tiers, reports)

Generated tasks are saved to `data/emtom/tasks/`.
