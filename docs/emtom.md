# EMTOM: Embodied Theory of Mind Benchmark

EMTOM (Embodied Theory of Mind) is a benchmark framework for testing theory of mind reasoning in multi-agent embodied AI systems. It works by introducing "unexpected behaviors" (mechanics) that induce surprise and require agents to model each other's mental states.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Core Concepts](#core-concepts)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Extending EMTOM](#extending-emtom)
- [API Reference](#api-reference)

---

## Overview

EMTOM tests whether AI agents can:
1. **Detect unexpected behaviors** - Notice when actions don't produce expected results
2. **Update mental models** - Adjust beliefs based on observations
3. **Reason about others' beliefs** - Understand what other agents know/don't know
4. **Communicate effectively** - Share relevant information with teammates

The benchmark uses a three-stage pipeline:
1. **Exploration** - LLM-guided discovery of mechanics in Habitat environments
2. **Task Generation** - LLM-based creation of collaborative challenges from discoveries
3. **Evaluation** - Multi-agent benchmark with theory of mind requirements

---

## Quick Start

### Running the Full Pipeline

```bash
# Run everything: exploration -> task generation -> benchmark
./emtom/run_emtom.sh all

# Or run individual stages
./emtom/run_emtom.sh exploration --steps 50
./emtom/run_emtom.sh generate
./emtom/run_emtom.sh benchmark --max-sim-steps 2000
```

### Running Exploration Only

```bash
# LLM-guided exploration with video output
./emtom/run_emtom.sh exploration --steps 30
```

### Running Benchmark Only

```bash
# Run benchmark evaluation
./emtom/run_emtom.sh benchmark --max-llm-calls 20
```

---

## Architecture

```
EMTOM Pipeline
==============

┌─────────────────────────────────────────────────────────────────┐
│                    1. EXPLORATION PHASE                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Habitat    │───▶│  Curiosity   │───▶│    Surprise      │   │
│  │ Environment  │    │    Model     │    │   Detection      │   │
│  └──────────────┘    │   (LLM)      │    └──────────────────┘   │
│         │            └──────────────┘             │              │
│         │                   │                     │              │
│         ▼                   ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Mechanics System                       │   │
│  │  • inverse_state: Actions have opposite effects          │   │
│  │  • remote_control: Actions affect different objects      │   │
│  │  • counting_state: Objects require multiple interactions │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  Trajectory Log  │                         │
│                    │  (JSON + Video)  │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. TASK GENERATION PHASE                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Trajectory  │───▶│  Trajectory  │───▶│ Task Generator   │   │
│  │   Analyzer   │    │   Analysis   │    │     (LLM)        │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                   │              │
│                                                   ▼              │
│                                          ┌──────────────────┐   │
│                                          │  Generated Tasks │   │
│                                          │     (JSON)       │   │
│                                          └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. BENCHMARK PHASE                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Habitat    │◀──▶│   Agents     │◀──▶│   Evaluation     │   │
│  │ Environment  │    │  (Planner)   │    │    Measures      │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                              │                     │             │
│                              ▼                     ▼             │
│                    ┌─────────────────────────────────────┐      │
│                    │         Benchmark Results           │      │
│                    │  • Success rate by task category    │      │
│                    │  • ToM vs non-ToM performance       │      │
│                    │  • Communication effectiveness      │      │
│                    └─────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Exploration

**Purpose**: Discover how mechanics work in the environment through LLM-guided exploration.

**Key Components**:
- `HabitatExplorer` (`emtom/exploration/habitat_explorer.py`): Main exploration loop
- `CuriosityModel` (`emtom/exploration/curiosity.py`): LLM-based action selection
- `TrajectoryLogger` (`emtom/exploration/trajectory_logger.py`): Records exploration data

**Flow**:
1. Agent explores the Habitat environment using ReACT prompting
2. Mechanics transform actions to produce surprising outcomes
3. When agent observes something unexpected, it includes "Surprise:" in its reasoning
4. All actions, observations, and surprises are logged to trajectory files

**Output**: JSON trajectory files + video recordings in `data/emtom/trajectories/`

**ReACT Format** (matches benchmark):
```
Thought: [Agent's reasoning, may include "Surprise:" for discoveries]
Agent_0_Action: ActionName[target]
Assigned!
Agent_0_Observation: [Result of action]
```

### Stage 2: Task Generation

**Purpose**: Create collaborative challenges that leverage discovered mechanics.

**Key Components**:
- `TrajectoryAnalyzer` (`emtom/task_gen/trajectory_analyzer.py`): Extracts patterns from trajectories
- `TaskGenerator` (`emtom/task_gen/task_generator.py`): LLM-based task creation

**Flow**:
1. Analyze trajectory files to extract surprises and mechanics
2. LLM generates tasks that require understanding the discovered mechanics
3. Tasks include knowledge asymmetry (one agent knows mechanics, other doesn't)
4. Success/failure conditions are grounded in real scene objects

**Output**: JSON task files in `data/emtom/tasks/`

### Stage 3: Benchmark Evaluation

**Purpose**: Evaluate multi-agent performance on generated tasks.

**Key Components**:
- `HabitatBenchmarkRunner` (`emtom/benchmark/habitat_runner.py`): Runs tasks in Habitat
- `BenchmarkEvaluator` (`emtom/benchmark/evaluator.py`): Computes metrics
- Habitat measures (`habitat_llm/agent/env/measures.py`): Task verification

**Flow**:
1. Load task definitions and initialize Habitat environment
2. Agents execute tasks using ReACT-based planning
3. Measures track proposition satisfaction and constraints
4. Results aggregated into success rates by category

**Metrics**:
- Overall success rate
- Success by task category (coordination, knowledge_asymmetry, etc.)
- ToM-required vs non-ToM task performance
- Communication effectiveness

---

## Core Concepts

### Mechanics

Mechanics are the "unexpected behaviors" that create theory of mind challenges. They transform normal action outcomes into surprising results.

**Base Classes** (`emtom/core/mechanic.py`):
- `Mechanic`: Abstract base for all mechanics
- `SceneAwareMechanic`: Discovers applicable objects at runtime
- `Effect`: Represents state changes from actions
- `ActionResult`: Contains effects, observations, and surprise triggers

**Available Mechanics** (`emtom/mechanics/`):

| Mechanic | Category | Description |
|----------|----------|-------------|
| `inverse_state` | INVERSE | Actions have opposite effects (open → closes) |
| `remote_control` | HIDDEN_MAPPING | Actions affect different objects |
| `counting_state` | CONDITIONAL | Objects require N interactions to change |

**Mechanic Categories** (`MechanicCategory`):
- `INVERSE`: Actions produce opposite effects
- `HIDDEN_MAPPING`: Actions affect unexpected targets
- `CONDITIONAL`: Effects depend on hidden state
- `TIME_DELAYED`: Effects happen after a delay
- `PER_AGENT`: Different agents observe different effects

### Custom Actions

EMTOM adds custom actions beyond partnr's built-in tools (`emtom/actions/custom_actions.py`):

| Action | Description |
|--------|-------------|
| `Hide` | Hide an object from other agents |
| `Inspect` | Examine object properties closely |
| `WriteMessage` | Leave a message for other agents |

**Adding New Actions**:
```python
from emtom.actions.registry import register_action
from emtom.actions.custom_actions import EMTOMAction, ActionResult

@register_action("MyAction")
class MyAction(EMTOMAction):
    name = "MyAction"
    description = "Does something interesting"

    def execute(self, agent_id, target, env_interface, world_state) -> ActionResult:
        # Implementation
        return ActionResult(
            success=True,
            observation="You did the thing!",
        )
```

### Task Structure

Generated tasks have this structure (`emtom/task_gen/task_generator.py`):

```python
@dataclass
class GeneratedTask:
    task_id: str
    title: str
    category: TaskCategory  # COORDINATION, KNOWLEDGE_ASYMMETRY, etc.
    description: str

    initial_world_state: Dict[str, Any]
    required_mechanics: List[str]

    num_agents: int
    agent_roles: Dict[str, str]       # {"agent_0": "Expert", "agent_1": "Novice"}
    agent_knowledge: Dict[str, List[str]]  # What each agent knows

    subtasks: List[Subtask]
    success_condition: SuccessCondition
    failure_conditions: List[FailureCondition]

    difficulty: int
    theory_of_mind_required: bool
    communication_required: bool
```

### Task Verification

Tasks are verified through Habitat's measure system (`habitat_llm/agent/env/measures.py`):

**Key Measures**:
1. `AutoEvalPropositionTracker`: Tracks proposition satisfaction over time
2. `TaskConstraintValidation`: Validates temporal/argument constraints
3. `TaskPercentComplete`: Calculates completion ratio (0.0-1.0)
4. `TaskStateSuccess`: Binary success (1.0 if 100% complete)
5. `TaskExplanation`: Natural language failure explanation

**Proposition Types** (`predicate_wrappers.py`):
- `is_on_top`: Object on receptacle
- `is_inside`: Object inside receptacle
- `is_in_room`: Object in specific room
- `is_next_to`: Entity proximity
- `is_powered_on/off`: Power state
- `is_clean/dirty`: Cleanliness state

---

## Directory Structure

```
emtom/
├── __init__.py                 # Package exports
├── config.py                   # Configuration dataclasses
├── run_emtom.sh               # Main pipeline script
│
├── core/                       # Core abstractions
│   ├── mechanic.py            # Mechanic, Effect, ActionResult
│   └── object_selector.py     # Scene object selection utilities
│
├── mechanics/                  # Mechanic implementations
│   ├── registry.py            # @register_mechanic decorator
│   ├── inverse_state.py       # Actions have opposite effects
│   ├── remote_control.py      # Actions affect different objects
│   └── counting_state.py      # Objects need N interactions
│
├── actions/                    # Custom EMTOM actions
│   ├── registry.py            # @register_action decorator
│   └── custom_actions.py      # Hide, Inspect, WriteMessage
│
├── exploration/                # Exploration phase
│   ├── curiosity.py           # LLM-guided action selection
│   ├── habitat_explorer.py    # Main exploration loop
│   ├── surprise_detector.py   # Surprise detection (legacy)
│   └── trajectory_logger.py   # Trajectory recording
│
├── task_gen/                   # Task generation phase
│   ├── task_generator.py      # LLM-based task creation
│   └── trajectory_analyzer.py # Extract patterns from trajectories
│
├── benchmark/                  # Benchmark phase
│   ├── habitat_runner.py      # Run tasks in Habitat
│   ├── task_runner.py         # Task execution logic
│   └── evaluator.py           # Metrics computation
│
├── tools/                      # Tool integrations
│   └── emtom_tools.py         # EMTOM tool wrappers
│
└── examples/                   # Entry point scripts
    ├── run_habitat_exploration.py
    ├── run_habitat_benchmark.py
    └── generate_tasks.py
```

**Configuration Files** (`habitat_llm/conf/`):
```
habitat_llm/conf/
├── instruct/
│   └── emtom_exploration.yaml  # Exploration ReACT prompt
└── examples/
    ├── emtom_two_robots.yaml   # Two robot benchmark config
    └── emtom_two_humans.yaml   # Two human benchmark config
```

**Data Directories**:
```
data/emtom/
├── trajectories/              # Exploration outputs (JSON + video)
└── tasks/                     # Generated tasks (JSON)

outputs/emtom/                 # Timestamped run outputs
├── YYYY-MM-DD_HH-MM-SS-exploration/
│   └── results/
│       ├── trajectory_*.json
│       └── videos/
└── YYYY-MM-DD_HH-MM-SS-benchmark/
    └── results/
```

---

## Configuration

### Exploration Config

The exploration prompt is defined in `habitat_llm/conf/instruct/emtom_exploration.yaml`:

```yaml
prompt: |-
    - Overview:
    You are exploring a simulated home environment...

    {tool_descriptions}

    IMPORTANT:
    - To interact with objects physically, you MUST first navigate close to them.
    - When you observe something UNEXPECTED, include "Surprise:" in your Thought.

    [Examples...]

    Task: {input}
    Thought:

stopword: "Assigned!"
end_expression: "Final Thought:"
```

**Placeholders**:
- `{id}`: Agent ID (0, 1, etc.)
- `{tool_descriptions}`: Available tools from agent config
- `{input}`: Current task/world state

### Benchmark Config

Example config in `habitat_llm/conf/examples/emtom_two_robots.yaml`:

```yaml
defaults:
  - planner_multi_agent_demo_config
  - _self_

# Override for EMTOM
habitat:
  environment:
    max_episode_steps: 20000

evaluation:
  agents:
    agent_0:
      planner:
        plan_config:
          replanning_threshold: 20
    agent_1:
      planner:
        plan_config:
          replanning_threshold: 20
```

### Command Line Overrides

Use Hydra syntax for runtime overrides:

```bash
# Change exploration steps
python emtom/examples/run_habitat_exploration.py +exploration_steps=100

# Change LLM model
python emtom/examples/run_habitat_benchmark.py llm.model=gpt-4

# Disable video saving
python emtom/examples/run_habitat_exploration.py evaluation.save_video=false
```

---

## Extending EMTOM

### Adding a New Mechanic

1. Create a new file in `emtom/mechanics/`:

```python
# emtom/mechanics/my_mechanic.py
from emtom.core.mechanic import SceneAwareMechanic, ActionResult, Effect, MechanicCategory
from emtom.mechanics.registry import register_mechanic

@register_mechanic("my_mechanic")
class MyMechanic(SceneAwareMechanic):
    name = "my_mechanic"
    category = MechanicCategory.CONDITIONAL
    description = "My custom mechanic behavior"

    def bind_to_scene(self, world_state) -> bool:
        # Discover applicable objects
        return self.bind_to_entities_with_state(world_state, max_targets=2)

    def applies_to(self, action_name, target, world_state) -> bool:
        return target in self._bound_targets

    def transform_effect(self, action_name, actor_id, target, intended_effect, world_state) -> ActionResult:
        # Transform the action outcome
        return ActionResult(
            success=True,
            effects=[...],
            observations={actor_id: "Something unexpected happened!"},
            surprise_triggers={actor_id: "Expected X but got Y"},
        )
```

2. Import in `emtom/mechanics/__init__.py`:
```python
from emtom.mechanics.my_mechanic import MyMechanic
```

3. Add to exploration setup in `run_habitat_exploration.py`:
```python
from emtom.mechanics import MyMechanic
mechanics = [..., MyMechanic()]
```

### Adding a New Custom Action

1. Add to `emtom/actions/custom_actions.py`:

```python
@register_action("MyAction")
class MyAction(EMTOMAction):
    name = "MyAction"
    description = "Does something for Theory of Mind testing"

    def execute(self, agent_id, target, env_interface, world_state) -> ActionResult:
        return ActionResult(
            success=True,
            observation=f"You performed MyAction on {target}",
            effect="my_effect",
            other_observations={"agent_1": "You notice agent_0 doing something"},
        )

    def get_available_targets(self, env_interface, world_state) -> List[str]:
        return [e["name"] for e in world_state.get("entities", [])][:10]
```

The action is automatically registered and available in exploration.

### Adding a New Task Category

1. Add to `TaskCategory` enum in `emtom/task_gen/task_generator.py`:
```python
class TaskCategory(Enum):
    COORDINATION = "coordination"
    # ... existing categories
    MY_CATEGORY = "my_category"  # Add new category
```

2. Update the task generation prompt to include examples of the new category.

---

## API Reference

### Core Classes

#### `Mechanic` (Abstract Base)
```python
class Mechanic(ABC):
    name: str
    category: MechanicCategory
    description: str

    def applies_to(self, action_name: str, target: str, world_state) -> bool: ...
    def transform_effect(self, action_name, actor_id, target, intended_effect, world_state) -> ActionResult: ...
    def reset(self) -> None: ...
    def get_hidden_state_for_debug(self) -> Dict[str, Any]: ...
```

#### `SceneAwareMechanic`
```python
class SceneAwareMechanic(Mechanic):
    required_affordance: Optional[str]

    def bind_to_scene(self, world_state) -> bool: ...
    def bind_to_entities_with_state(self, world_state, state_names=None, max_targets=1) -> bool: ...

    @property
    def is_bound(self) -> bool: ...
    @property
    def bound_targets(self) -> List[str]: ...
```

#### `CuriosityModel`
```python
class CuriosityModel:
    def __init__(self, llm_client, instruct_config=None, llm_config=None): ...
    def select_action(self, agent_id, world_description, available_actions, exploration_history=None) -> ActionChoice: ...
    def add_observation(self, agent_id, observation: str): ...
    def reset(self): ...
    def set_tool_descriptions(self, tool_descriptions: str): ...
```

#### `HabitatExplorer`
```python
class HabitatExplorer:
    def __init__(self, env_interface, mechanics, curiosity_model, surprise_detector, agent=None, config=None): ...
    def run(self, metadata=None) -> Dict[str, Any]: ...
    def stop(self): ...
```

#### `TaskGenerator`
```python
class TaskGenerator:
    def __init__(self, llm_client=None): ...
    def generate_tasks(self, trajectory, analysis, num_agents=2, max_tasks=5) -> List[GeneratedTask]: ...
```

### Registry Functions

```python
# Mechanics
from emtom.mechanics.registry import register_mechanic, MechanicRegistry

@register_mechanic("name")
class MyMechanic(Mechanic): ...

MechanicRegistry.list_all() -> List[str]
MechanicRegistry.get("name") -> Type[Mechanic]
MechanicRegistry.instantiate("name", **params) -> Mechanic

# Actions
from emtom.actions.registry import register_action, ActionRegistry

@register_action("Name")
class MyAction(EMTOMAction): ...

ActionRegistry.list_all() -> List[str]
ActionRegistry.instantiate_all() -> Dict[str, EMTOMAction]
```

---

## Troubleshooting

### Common Issues

**"LLM not set in FindObjectTool"**
```python
# Ensure LLM is passed to agent tools
agent.pass_llm_to_tools(llm_client)
```

**"Failed to parse Action from LLM response"**
- The LLM didn't follow ReACT format
- Check the prompt in `emtom_exploration.yaml`
- Ensure examples show correct format: `Agent_{id}_Action: ActionName[target]`

**"No surprises found in trajectory"**
- Mechanics may not have bound to any objects
- Check mechanic binding logs for "bound to: [...]"
- Ensure scene has objects with appropriate states (is_open, is_on, etc.)

**Video not saving**
- Ensure `evaluation.save_video=true` in config
- Check output directory permissions
- Look for video files in `outputs/emtom/*/results/videos/`

### Debug Tips

1. **Check mechanic bindings**:
   ```python
   for m in mechanics:
       print(m.get_hidden_state_for_debug())
   ```

2. **View exploration logs**:
   ```bash
   cat data/emtom/trajectories/trajectory_*.json | python -m json.tool
   ```

3. **Run with verbose output**:
   ```bash
   HYDRA_FULL_ERROR=1 python emtom/examples/run_habitat_exploration.py
   ```
