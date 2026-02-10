"""
GRPO (Group Relative Policy Optimization) for multi-agent RL.

Episode-level GRPO: run N full episodes per task, compute group-relative
advantages over episode rewards, update policy with clipped surrogate loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TurnData:
    """Data from one agent's action in one turn."""

    agent_id: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    logprobs: List[float]  # Per-token logprobs from policy at generation time
    step_reward: float = 0.0

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_token_ids)


@dataclass
class Trajectory:
    """One full episode rollout."""

    turns: List[TurnData] = field(default_factory=list)
    episode_reward: float = 0.0
    task_id: str = ""

    @property
    def total_completion_tokens(self) -> int:
        return sum(td.num_completion_tokens for td in self.turns)


def compute_grpo_advantages(
    trajectories: List[Trajectory],
    eps: float = 1e-8,
) -> List[float]:
    """
    Episode-level GRPO: A_j = (R_j - mean(R)) / std(R).

    All trajectories should be from the same task (same group).

    Args:
        trajectories: N trajectories from the same task.
        eps: Small constant to avoid division by zero.

    Returns:
        List of N advantage values, one per trajectory.
    """
    if len(trajectories) <= 1:
        return [0.0] * len(trajectories)

    rewards = np.array([t.episode_reward for t in trajectories])
    mean_r = rewards.mean()
    std_r = rewards.std() + eps
    return ((rewards - mean_r) / std_r).tolist()


def compute_drgrpo_advantages(
    trajectories: List[Trajectory],
    eps: float = 1e-8,
) -> List[float]:
    """
    DR-GRPO: A_j = (R_j - mean(R)) / |mean(R)|.

    More stable than GRPO when reward variance is low, since it normalizes
    by the absolute mean rather than std. Falls back to std normalization
    when mean is near zero.

    Args:
        trajectories: N trajectories from the same task.
        eps: Small constant to avoid division by zero.

    Returns:
        List of N advantage values, one per trajectory.
    """
    if len(trajectories) <= 1:
        return [0.0] * len(trajectories)

    rewards = np.array([t.episode_reward for t in trajectories])
    mean_r = rewards.mean()

    # Fall back to std normalization when mean ≈ 0
    if abs(mean_r) < eps:
        std_r = rewards.std() + eps
        return ((rewards - mean_r) / std_r).tolist()

    return ((rewards - mean_r) / abs(mean_r)).tolist()


def compute_rloo_advantages(
    trajectories: List[Trajectory],
) -> List[float]:
    """
    RLOO (Reinforce Leave-One-Out): A_j = R_j - mean(R_{-j}).

    Uses a leave-one-out baseline for each trajectory, which provides
    lower variance than the GRPO group baseline.

    Args:
        trajectories: N trajectories from the same task.

    Returns:
        List of N advantage values, one per trajectory.
    """
    if len(trajectories) <= 1:
        return [0.0] * len(trajectories)

    rewards = np.array([t.episode_reward for t in trajectories])
    total = rewards.sum()
    n = len(rewards)
    advantages = []
    for j in range(n):
        baseline = (total - rewards[j]) / (n - 1)
        advantages.append(float(rewards[j] - baseline))
    return advantages


def compute_advantages(
    trajectories: List[Trajectory],
    method: str = "grpo",
    **kwargs,
) -> List[float]:
    """
    Dispatcher for advantage computation methods.

    Args:
        trajectories: N trajectories from the same task.
        method: One of "grpo", "drgrpo", "rloo".
        **kwargs: Passed to the underlying function.

    Returns:
        List of N advantage values, one per trajectory.
    """
    if method == "grpo":
        return compute_grpo_advantages(trajectories, **kwargs)
    elif method == "drgrpo":
        return compute_drgrpo_advantages(trajectories, **kwargs)
    elif method == "rloo":
        return compute_rloo_advantages(trajectories, **kwargs)
    else:
        raise ValueError(f"Unknown advantage method: {method!r}. Choose from: grpo, drgrpo, rloo")


def grpo_loss(
    model: torch.nn.Module,
    trajectories: List[Trajectory],
    advantages: List[float],
    tokenizer,
    clip_epsilon: float = 0.2,
    kl_coeff: float = 0.01,
    ref_model: Optional[torch.nn.Module] = None,
    max_seq_len: int = 4096,
    device: str = "cuda",
) -> torch.Tensor:
    """
    GRPO policy gradient loss.

    For each trajectory j with advantage A_j:
      For each turn's completion tokens:
        ratio = exp(new_logprob - old_logprob)
        clipped_ratio = clip(ratio, 1-eps, 1+eps)
        loss -= min(ratio * A_j, clipped_ratio * A_j)
        loss += kl_coeff * KL(new || ref)

    Args:
        model: HuggingFace model for computing new logprobs.
        trajectories: List of trajectories with old logprobs.
        advantages: Per-trajectory advantage values.
        tokenizer: HuggingFace tokenizer.
        clip_epsilon: PPO-style clipping parameter.
        kl_coeff: KL penalty coefficient.
        ref_model: Reference model for KL penalty (None = skip KL).
        max_seq_len: Maximum sequence length for model input.
        device: Device for tensors.

    Returns:
        Scalar loss tensor.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_tokens = 0

    for traj, advantage in zip(trajectories, advantages):
        if abs(advantage) < 1e-10:
            continue

        advantage_t = torch.tensor(advantage, device=device, dtype=torch.float32)

        for turn in traj.turns:
            if not turn.completion_token_ids:
                continue

            # Build full input: prompt + completion
            full_ids = turn.prompt_token_ids + turn.completion_token_ids
            # Truncate from the left (keep most recent context)
            if len(full_ids) > max_seq_len:
                excess = len(full_ids) - max_seq_len
                full_ids = full_ids[excess:]
                # Adjust prompt length
                prompt_len = max(0, len(turn.prompt_token_ids) - excess)
            else:
                prompt_len = len(turn.prompt_token_ids)

            completion_len = len(full_ids) - prompt_len
            if completion_len <= 0:
                continue

            input_ids = torch.tensor([full_ids], device=device)

            # Get new logprobs from model
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Logprobs for completion tokens
            # logits[t] predicts token[t+1], so for completion starting at prompt_len:
            # logits[prompt_len-1 : prompt_len-1+completion_len] predicts completion tokens
            completion_logits = logits[0, prompt_len - 1 : prompt_len - 1 + completion_len]
            completion_targets = input_ids[0, prompt_len : prompt_len + completion_len]

            new_logprobs = F.log_softmax(completion_logits, dim=-1)
            new_token_logprobs = new_logprobs.gather(
                1, completion_targets.unsqueeze(1)
            ).squeeze(1)

            # Old logprobs (from vLLM generation)
            old_logprobs_list = turn.logprobs[:completion_len]
            # Pad if needed (shouldn't happen but defensive)
            while len(old_logprobs_list) < completion_len:
                old_logprobs_list.append(0.0)
            old_token_logprobs = torch.tensor(
                old_logprobs_list[:completion_len], device=device, dtype=torch.float32
            )

            # PPO-style clipped surrogate
            ratio = torch.exp(new_token_logprobs - old_token_logprobs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            surrogate = torch.min(ratio * advantage_t, clipped_ratio * advantage_t)
            policy_loss = -surrogate.sum()

            # Optional KL penalty against reference model
            kl_loss = torch.tensor(0.0, device=device)
            if ref_model is not None and kl_coeff > 0:
                with torch.no_grad():
                    ref_outputs = ref_model(input_ids=input_ids)
                ref_logits = ref_outputs.logits[0, prompt_len - 1 : prompt_len - 1 + completion_len]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_token_logprobs = ref_logprobs.gather(
                    1, completion_targets.unsqueeze(1)
                ).squeeze(1)
                # KL(new || ref) ≈ sum of (new_logprob - ref_logprob) * exp(new_logprob)
                kl_div = (new_token_logprobs.exp() * (new_token_logprobs - ref_token_logprobs)).sum()
                kl_loss = kl_coeff * kl_div

            total_loss = total_loss + policy_loss + kl_loss
            total_tokens += completion_len

    if total_tokens > 0:
        total_loss = total_loss / total_tokens

    return total_loss
