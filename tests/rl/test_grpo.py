"""Unit tests for GRPO — pure math, no Habitat needed."""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from emtom.rl.grpo import (
    Trajectory,
    TurnData,
    compute_grpo_advantages,
    compute_drgrpo_advantages,
    compute_rloo_advantages,
    compute_advantages,
    grpo_loss,
)


class TestTurnData:
    def test_num_tokens(self):
        td = TurnData(
            agent_id="agent_0",
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[4, 5, 6, 7],
            logprobs=[-0.5, -0.3, -0.2, -0.1],
        )
        assert td.num_completion_tokens == 4

    def test_empty_completion(self):
        td = TurnData(
            agent_id="agent_0",
            prompt_token_ids=[1, 2],
            completion_token_ids=[],
            logprobs=[],
        )
        assert td.num_completion_tokens == 0


class TestTrajectory:
    def test_total_tokens(self):
        t1 = TurnData("agent_0", [1], [2, 3], [-0.1, -0.2])
        t2 = TurnData("agent_1", [1], [4, 5, 6], [-0.1, -0.2, -0.3])
        traj = Trajectory(turns=[t1, t2], episode_reward=1.5, task_id="task_1")
        assert traj.total_completion_tokens == 5


class TestGRPOAdvantages:
    def test_single_trajectory(self):
        traj = Trajectory(episode_reward=1.0, task_id="t1")
        adv = compute_grpo_advantages([traj])
        assert adv == [0.0]

    def test_zero_variance(self):
        trajs = [Trajectory(episode_reward=1.0) for _ in range(4)]
        adv = compute_grpo_advantages(trajs)
        for a in adv:
            assert abs(a) < 1e-6

    def test_basic_advantages(self):
        trajs = [
            Trajectory(episode_reward=0.0),
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=2.0),
            Trajectory(episode_reward=3.0),
        ]
        adv = compute_grpo_advantages(trajs)

        # Mean = 1.5, std ≈ 1.118
        assert adv[0] < adv[1] < adv[2] < adv[3]
        assert abs(np.mean(adv)) < 1e-6  # Mean advantage should be ~0

    def test_negative_rewards(self):
        trajs = [
            Trajectory(episode_reward=-2.0),
            Trajectory(episode_reward=-1.0),
            Trajectory(episode_reward=0.0),
            Trajectory(episode_reward=1.0),
        ]
        adv = compute_grpo_advantages(trajs)
        assert adv[0] < adv[3]  # Worst < best

    def test_empty(self):
        assert compute_grpo_advantages([]) == []


class TestGRPOLoss:
    @pytest.fixture
    def simple_model(self):
        """Create a tiny model for testing loss computation."""
        from torch import nn

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.head = nn.Linear(hidden, vocab_size)

            def forward(self, input_ids, **kwargs):
                h = self.embed(input_ids)
                logits = self.head(h)
                return MagicMock(logits=logits)

        model = TinyLM()
        model.to("cpu")
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kwargs: list(range(10))
        return tokenizer

    def test_zero_advantage_zero_loss(self, simple_model, mock_tokenizer):
        """With zero advantage, loss should be approximately zero."""
        traj = Trajectory(
            turns=[
                TurnData(
                    agent_id="agent_0",
                    prompt_token_ids=list(range(5)),
                    completion_token_ids=list(range(5, 10)),
                    logprobs=[-0.5] * 5,
                )
            ],
            episode_reward=1.0,
        )

        loss = grpo_loss(
            model=simple_model,
            trajectories=[traj],
            advantages=[0.0],
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        assert abs(loss.item()) < 1e-6

    def test_positive_advantage_loss_sign(self, simple_model, mock_tokenizer):
        """With positive advantage, surrogate should encourage these actions."""
        traj = Trajectory(
            turns=[
                TurnData(
                    agent_id="agent_0",
                    prompt_token_ids=list(range(5)),
                    completion_token_ids=list(range(5, 10)),
                    logprobs=[-0.5] * 5,
                )
            ],
            episode_reward=2.0,
        )

        loss = grpo_loss(
            model=simple_model,
            trajectories=[traj],
            advantages=[1.0],
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        # Loss should be a finite number
        assert torch.isfinite(loss)

    def test_gradient_flow(self, simple_model, mock_tokenizer):
        """Verify gradients flow back to model parameters."""
        traj = Trajectory(
            turns=[
                TurnData(
                    agent_id="agent_0",
                    prompt_token_ids=list(range(5)),
                    completion_token_ids=list(range(5, 10)),
                    logprobs=[-0.5] * 5,
                )
            ],
            episode_reward=2.0,
        )

        loss = grpo_loss(
            model=simple_model,
            trajectories=[traj],
            advantages=[1.5],
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in simple_model.parameters())
        assert has_grad, "No gradients flowed to model parameters"

    def test_multiple_trajectories(self, simple_model, mock_tokenizer):
        """Test with multiple trajectories and mixed advantages."""
        trajs = []
        for i in range(3):
            trajs.append(Trajectory(
                turns=[
                    TurnData(
                        agent_id="agent_0",
                        prompt_token_ids=list(range(5)),
                        completion_token_ids=list(range(5, 10)),
                        logprobs=[-0.5] * 5,
                    )
                ],
                episode_reward=float(i),
            ))

        advantages = compute_grpo_advantages(trajs)
        loss = grpo_loss(
            model=simple_model,
            trajectories=trajs,
            advantages=advantages,
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        assert torch.isfinite(loss)


class TestDRGRPOAdvantages:
    def test_single_trajectory(self):
        traj = Trajectory(episode_reward=1.0, task_id="t1")
        adv = compute_drgrpo_advantages([traj])
        assert adv == [0.0]

    def test_basic_values(self):
        """DR-GRPO: A_j = (R_j - mean) / |mean|."""
        trajs = [
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=2.0),
            Trajectory(episode_reward=3.0),
            Trajectory(episode_reward=4.0),
        ]
        adv = compute_drgrpo_advantages(trajs)
        # mean = 2.5, |mean| = 2.5
        # A_0 = (1 - 2.5) / 2.5 = -0.6
        # A_1 = (2 - 2.5) / 2.5 = -0.2
        # A_2 = (3 - 2.5) / 2.5 = 0.2
        # A_3 = (4 - 2.5) / 2.5 = 0.6
        assert abs(adv[0] - (-0.6)) < 1e-6
        assert abs(adv[1] - (-0.2)) < 1e-6
        assert abs(adv[2] - 0.2) < 1e-6
        assert abs(adv[3] - 0.6) < 1e-6

    def test_mean_zero_fallback(self):
        """When mean ≈ 0, falls back to std normalization."""
        trajs = [
            Trajectory(episode_reward=-1.0),
            Trajectory(episode_reward=1.0),
        ]
        adv = compute_drgrpo_advantages(trajs)
        # mean = 0, falls back to std normalization
        assert adv[0] < 0
        assert adv[1] > 0

    def test_negative_mean(self):
        """With negative mean, |mean| is used for normalization."""
        trajs = [
            Trajectory(episode_reward=-4.0),
            Trajectory(episode_reward=-2.0),
        ]
        adv = compute_drgrpo_advantages(trajs)
        # mean = -3, |mean| = 3
        # A_0 = (-4 - (-3)) / 3 = -1/3
        # A_1 = (-2 - (-3)) / 3 = 1/3
        assert abs(adv[0] - (-1.0 / 3.0)) < 1e-6
        assert abs(adv[1] - (1.0 / 3.0)) < 1e-6

    def test_empty(self):
        assert compute_drgrpo_advantages([]) == []


class TestRLOOAdvantages:
    def test_single_trajectory(self):
        traj = Trajectory(episode_reward=1.0, task_id="t1")
        adv = compute_rloo_advantages([traj])
        assert adv == [0.0]

    def test_basic_values(self):
        """RLOO: A_j = R_j - mean(R_{-j})."""
        trajs = [
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=2.0),
            Trajectory(episode_reward=3.0),
        ]
        adv = compute_rloo_advantages(trajs)
        # A_0 = 1 - mean(2, 3) = 1 - 2.5 = -1.5
        # A_1 = 2 - mean(1, 3) = 2 - 2.0 = 0.0
        # A_2 = 3 - mean(1, 2) = 3 - 1.5 = 1.5
        assert abs(adv[0] - (-1.5)) < 1e-6
        assert abs(adv[1] - 0.0) < 1e-6
        assert abs(adv[2] - 1.5) < 1e-6

    def test_ordering(self):
        """Higher rewards get higher advantages."""
        trajs = [
            Trajectory(episode_reward=0.0),
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=2.0),
            Trajectory(episode_reward=3.0),
        ]
        adv = compute_rloo_advantages(trajs)
        assert adv[0] < adv[1] < adv[2] < adv[3]

    def test_empty(self):
        assert compute_rloo_advantages([]) == []


class TestComputeAdvantagesDispatcher:
    def test_routes_grpo(self):
        trajs = [
            Trajectory(episode_reward=0.0),
            Trajectory(episode_reward=2.0),
        ]
        adv_direct = compute_grpo_advantages(trajs)
        adv_dispatch = compute_advantages(trajs, method="grpo")
        assert adv_direct == adv_dispatch

    def test_routes_drgrpo(self):
        trajs = [
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=3.0),
        ]
        adv_direct = compute_drgrpo_advantages(trajs)
        adv_dispatch = compute_advantages(trajs, method="drgrpo")
        assert adv_direct == adv_dispatch

    def test_routes_rloo(self):
        trajs = [
            Trajectory(episode_reward=1.0),
            Trajectory(episode_reward=2.0),
            Trajectory(episode_reward=3.0),
        ]
        adv_direct = compute_rloo_advantages(trajs)
        adv_dispatch = compute_advantages(trajs, method="rloo")
        assert adv_direct == adv_dispatch

    def test_unknown_method_raises(self):
        trajs = [Trajectory(episode_reward=1.0)]
        with pytest.raises(ValueError, match="Unknown advantage method"):
            compute_advantages(trajs, method="unknown")
