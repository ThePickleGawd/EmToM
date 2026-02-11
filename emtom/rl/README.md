# RL Training Module

GRPO-based reinforcement learning for EmToM multi-agent tasks. Wraps the existing benchmark stack (BenchmarkRunner, LLMPlanner, Habitat) into a PettingZoo-style RL environment.

## Quick Start

```bash
# Dry run: verify the full pipeline works (1 task x 2 episodes x 3 turns)
CUDA_VISIBLE_DEVICES=0 python emtom/rl/dry_run.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task-file data/emtom/tasks/some_task.json

# Full training (sequential)
python emtom/rl/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task-dir data/emtom/tasks \
  --epochs 3 \
  --group-size 4

# Training with DR-GRPO advantage method
python emtom/rl/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task-dir data/emtom/tasks \
  --advantage-method drgrpo

# Parallel training across multiple GPUs
python emtom/rl/train.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task-dir data/emtom/tasks \
  --parallel-workers 8
```

## Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model MODEL` | HuggingFace model name or path (required) | - |
| `--task-dir DIR` | Directory with task JSONs | `data/emtom/tasks` |
| `--output-dir DIR` | Output directory for checkpoints/stats | `outputs/rl_training` |
| `--epochs N` | Number of training epochs | 3 |
| `--group-size N` | GRPO group size (episodes per task) | 4 |
| `--max-turns N` | Max turns per episode | 20 |
| `--lr FLOAT` | Learning rate | 1e-5 |
| `--clip-epsilon FLOAT` | PPO clip epsilon | 0.2 |
| `--kl-coeff FLOAT` | KL penalty coefficient | 0.01 |
| `--temperature FLOAT` | Sampling temperature | 0.7 |
| `--advantage-method METHOD` | `grpo`, `drgrpo`, or `rloo` | `grpo` |
| `--parallel-workers N` | Parallel workers (1 = sequential) | 1 |
| `--config-name NAME` | Hydra config name | `examples/emtom_2_robots` |
| `--save-every N` | Save checkpoint every N epochs | 1 |
| `--tasks-per-epoch N` | Limit tasks per epoch | all |
| `--device DEVICE` | Device for model | `cuda:0` |

## Advantage Methods

| Method | Formula | When to Use |
|--------|---------|-------------|
| `grpo` | `(R - mean) / std` | Standard GRPO. Good default. |
| `drgrpo` | `(R - mean) / \|mean\|` | DR-GRPO. More stable when reward variance is low. Falls back to std when mean ~0. |
| `rloo` | `R - mean(R_{-j})` | RLOO (leave-one-out). Lower variance baseline. |

## Architecture

```
emtom/rl/
├── env.py          # EmtomMultiAgentEnv — PettingZoo-style wrapper over Habitat
├── reward.py       # RewardShaper — cooperative/competitive/mixed reward functions
├── grpo.py         # Trajectory, TurnData, advantage functions, clipped surrogate loss
├── train.py        # Main training loop (sequential or parallel)
├── parallel.py     # Subprocess + GPU round-robin episode collection
├── worker.py       # Single-episode subprocess (spawned by parallel.py)
└── dry_run.py      # End-to-end smoke test
```

### How it Works

1. **Environment** (`env.py`): Wraps BenchmarkRunner + LLMPlanner into `reset()`/`step()` API. Observations are the exact ReAct prompts. Actions are raw LLM completions parsed into tool calls.

2. **Reward Shaping** (`reward.py`): Per-step rewards based on task category:
   - **Cooperative**: shared `delta(percent_complete)` + terminal bonus
   - **Competitive**: `+1` winner / `-1` loser at episode end
   - **Mixed**: shared main goal progress + individual subgoal bonuses

3. **Episode Collection**: Each episode runs the HF model through the env for up to `max_turns`, collecting `TurnData` (prompt tokens, completion tokens, per-token logprobs, step reward).

4. **GRPO Update**: For each task, run `group_size` episodes. Compute group-relative advantages over episode rewards. Update policy with clipped surrogate loss (PPO-style).

5. **Parallel Mode**: When `--parallel-workers > 1`, episode collection is distributed across GPUs via subprocess spawning (same pattern as `emtom/evolve/benchmark_wrapper.py`). Each worker loads the model independently and writes trajectory JSON.

## Tests

```bash
pytest tests/rl/ -v
```

45 unit tests covering reward shaping, advantage computation (all 3 methods), loss/gradient flow, action parsing, and episode tracking. Integration tests (requiring Habitat + GPU) are marked with `@pytest.mark.integration` and skipped by default.
