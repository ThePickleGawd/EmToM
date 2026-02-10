"""
RL training module for EmToM.

Wraps the existing EmToM benchmark stack (BenchmarkRunner, LLMPlanner,
GameStateManager, EnvironmentInterface) into a multi-agent RL environment
with GRPO-based training.
"""

from emtom.rl.env import EmtomMultiAgentEnv, MultiAgentEnv
from emtom.rl.reward import RewardShaper
from emtom.rl.grpo import (
    Trajectory,
    TurnData,
    compute_grpo_advantages,
    compute_drgrpo_advantages,
    compute_rloo_advantages,
    compute_advantages,
    grpo_loss,
)
from emtom.rl.parallel import run_episodes_parallel

__all__ = [
    "EmtomMultiAgentEnv",
    "MultiAgentEnv",
    "RewardShaper",
    "Trajectory",
    "TurnData",
    "compute_grpo_advantages",
    "compute_drgrpo_advantages",
    "compute_rloo_advantages",
    "compute_advantages",
    "grpo_loss",
    "run_episodes_parallel",
]
