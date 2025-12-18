# EMTOM: Embodied Theory of Mind Benchmark

EMTOM tests whether AI agents can reason about others' mental states by introducing "surprise mechanics" - unexpected behaviors that require agents to update beliefs and communicate discoveries.

## Install

See [Installation.md](INSTALLATION.md) for conda setup.

## Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. EXPLORATION    │────▶│  2. TASK GENERATION │────▶│    3. BENCHMARK     │
│                     │     │                     │     │                     │
│  LLM explores env   │     │  Agent creates &    │     │  Multi-agent eval   │
│  discovers mechanics│     │  tests tasks        │     │  on generated tasks │
│                     │     │                     │     │                     │
│  Output: trajectory │     │  Output: task.json  │     │  Output: metrics    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

**Quick Start:**
```bash
./emtom/run_emtom.sh exploration      # Step 1
./emtom/run_emtom.sh generate # Step 2
./emtom/run_emtom.sh benchmark        # Step 3
```

---

## 1. Exploration

**Goal:** Discover how mechanics work through LLM-guided exploration.

**How it works:**
1. Agent explores Habitat environment using ReAct prompting
2. Mechanics transform actions (e.g., "open" becomes "close")
3. Agent detects surprises and logs discoveries
4. Output: trajectory JSON + video

**Key Components:**
- `HabitatExplorer` - Main exploration loop
- `CuriosityModel` - LLM selects actions based on curiosity
- `GameStateManager` - Applies mechanics to actions

**Mechanics Available:**
| Mechanic | Effect |
|----------|--------|
| `inverse_state` | Actions have opposite effects |
| `remote_control` | Actions affect a different object |
| `counting_state` | Object needs N interactions to respond |

**Output Format** (`data/emtom/trajectories/`):
```json
{
  "steps": [...],
  "surprises": [...],
  "mechanic_bindings": {"inverse_state": {"targets": ["drawer_1"]}}
}
```

---

## 2. Task Generation (Agentic)

**Goal:** Create quality benchmark tasks by iteratively testing them.

**How it works:**

The agent edits a task file, tests it in the benchmark, and iterates until quality is good.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  bash            │────▶│  test_task       │────▶│  submit_task     │
│                  │     │                  │     │                  │
│  Read trajectory │     │  Run benchmark   │     │  Save if good    │
│  Edit task.json  │     │  Get metrics     │     │  Move to next    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │
        │◀───────────────────────┘
        │     (iterate if bad)
```

**Tools (only 3):**
| Tool | Purpose |
|------|---------|
| `bash` | Read trajectories, edit `working_task.json` |
| `test_task` | Run benchmark, return {steps, done, episode_over} |
| `submit_task` | Save task to output directory |

**Quality Criteria:**
| Metric | Good | Bad |
|--------|------|-----|
| Steps | 10-50 | <10 (too easy) or >100 (too hard) |
| done | True | False (agents couldn't complete) |
| episode_over | False | True (environment crashed) |

**Task Structure:**
```json
{
  "task_id": "task_001",
  "public_goal": "Open the drawer",
  "mechanic_bindings": [{"mechanic_type": "inverse_state", "target": "drawer_1"}],
  "agent_secrets": {"agent_0": ["The drawer closes when you open it"], "agent_1": []}
}
```

---

## 3. Benchmark

**Goal:** Evaluate multi-agent performance on generated tasks.

**How it works:**
1. Load task and initialize Habitat environment
2. Agents execute using ReAct planning
3. Mechanics apply to actions (same as exploration)
4. Measure success based on goal completion

**Key Components:**
- `EMTOMBaseRunner` - Runs tasks with mechanics
- `GameStateManager` - Tracks state, applies mechanics
- Goal checking via `object_properties` overlay

**Action Execution Order:**
```
1. Check mechanics (block/transform?)
2. Execute in Habitat (physical action)
3. If Habitat fails (too far) → return failure, no state change
4. If Habitat succeeds → apply mechanic state changes
```

**Metrics:**
- Success rate (goals completed)
- Steps to completion
- Communication effectiveness

---

## Directory Structure

```
emtom/
├── exploration/           # Step 1
│   ├── habitat_explorer.py
│   └── curiosity.py
├── task_gen/              # Step 2
│   └── agentic/
│       ├── agent.py       # 3-tool ReAct agent
│       └── runner.py
├── runner/                # Step 3
│   └── base.py            # EMTOMBaseRunner
├── mechanics/
│   └── handlers.py        # inverse_state, remote_control, etc.
└── state/
    └── manager.py         # GameStateManager
```

---

## Configuration

```bash
# Exploration
./emtom/run_emtom.sh exploration --steps 50 --num-agents 2

# Task Generation
./emtom/run_emtom.sh generate --num-tasks 10

# Benchmark
./emtom/run_emtom.sh benchmark --max-sim-steps 2000
```
