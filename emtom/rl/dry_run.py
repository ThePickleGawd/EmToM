#!/usr/bin/env python
"""
Dry-run script: tests the full RL training pipeline end-to-end.

Usage:
    CUDA_VISIBLE_DEVICES=0 python emtom/rl/dry_run.py --model Qwen/Qwen2.5-0.5B-Instruct

Runs 1 task x 2 episodes x 3 turns to verify data collection + gradient flow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def dry_run(model_name: str, task_file: str, device: str = "cuda:0"):
    """Minimal end-to-end test of the training pipeline."""

    print("=" * 60)
    print("RL Training Dry Run")
    print("=" * 60)

    # Step 1: Load the model for training
    print("\n[1/6] Loading HuggingFace model for training...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {model_name} ({param_count:.0f}M params) on {device}")

    # Step 2: Load task
    print(f"\n[2/6] Loading task from {task_file}...")
    from emtom.task_gen.task_generator import GeneratedTask

    with open(task_file) as f:
        task_data = json.load(f)
    task = GeneratedTask.from_dict(task_data)
    print(f"  Task: {task.title}")
    print(f"  Category: {task.category}")
    print(f"  Agents: {sorted(task.agent_actions.keys())}")
    print(f"  Episode: {task.episode_id}")

    # Step 3: Initialize Habitat + RL env
    print("\n[3/6] Initializing Habitat environment...")
    env = _init_habitat_env(task)
    print("  Habitat initialized successfully!")

    # Step 4: Run episodes and collect trajectories
    print("\n[4/6] Running 2 episodes (3 turns each)...")
    from emtom.rl.grpo import Trajectory, TurnData, compute_grpo_advantages, grpo_loss
    from emtom.rl.train import generate_batch

    trajectories = []
    for ep_i in range(2):
        print(f"\n  --- Episode {ep_i + 1} ---")
        obs, info = env.reset(options={"task": task})

        turns = []
        total_reward = 0.0

        for turn in range(3):
            agent_ids = list(env.agents)
            prompts = [obs[aid] for aid in agent_ids]

            for aid, p in zip(agent_ids, prompts):
                print(f"    Turn {turn+1} | {aid} prompt: {len(p)} chars")

            # Generate with HF model
            model.eval()
            completions = generate_batch(
                model,
                prompts,
                tokenizer,
                temperature=0.7,
                max_tokens=128,
                stop=["Assigned!"],
                device=device,
            )

            actions = {}
            for agent_id, prompt, completion in zip(agent_ids, prompts, completions):
                text = completion["text"]
                if "Assigned!" not in text:
                    text = text.rstrip() + "\nAssigned!"
                actions[agent_id] = text
                print(f"    {agent_id}: {text[:100].replace(chr(10), ' | ')}...")

                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                turns.append(TurnData(
                    agent_id=agent_id,
                    prompt_token_ids=prompt_ids,
                    completion_token_ids=completion["token_ids"],
                    logprobs=completion["logprobs"],
                    step_reward=0.0,
                ))

            obs, rewards, terms, truncs, infos = env.step(actions)

            for td in turns[-len(agent_ids):]:
                td.step_reward = rewards.get(td.agent_id, 0.0)
                total_reward += td.step_reward

            print(f"    Rewards: {rewards}")

            if all(terms.values()) or all(truncs.values()):
                print("    Episode terminated!")
                break

        trajectories.append(Trajectory(
            turns=turns,
            episode_reward=total_reward,
            task_id=task.task_id,
        ))
        print(f"  Episode reward: {total_reward:.4f}")

    # Step 5: GRPO update
    print("\n[5/6] Computing GRPO loss and gradients...")
    advantages = compute_grpo_advantages(trajectories)
    print(f"  Advantages: {advantages}")
    print(f"  Rewards: {[t.episode_reward for t in trajectories]}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    loss = grpo_loss(
        model=model,
        trajectories=trajectories,
        advantages=advantages,
        tokenizer=tokenizer,
        clip_epsilon=0.2,
        kl_coeff=0.0,
        max_seq_len=2048,
        device=device,
    )
    print(f"  Loss: {loss.item():.6f}")

    if loss.requires_grad:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"  Grad norm: {grad_norm:.4f}")
        print("  Gradient update successful!")
    else:
        print("  Warning: Loss has no grad — no update performed")

    print("\n[6/6] Summary")
    for i, traj in enumerate(trajectories):
        print(f"  Episode {i+1}: {len(traj.turns)} turns, reward={traj.episode_reward:.4f}, "
              f"tokens={traj.total_completion_tokens}")

    print("\n" + "=" * 60)
    print("DRY RUN COMPLETE — All systems operational!")
    print("=" * 60)


def _init_habitat_env(task):
    """Initialize Habitat env for a single task."""
    from hydra import compose, initialize_config_dir

    from habitat_llm.utils import fix_config, setup_config
    from habitat_llm.agent.env import (
        EnvironmentInterface,
        register_actions,
        register_measures,
        register_sensors,
    )
    from habitat_llm.agent.env.dataset import CollaborationDatasetV0

    from emtom.rl.env import EmtomMultiAgentEnv

    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../habitat_llm/conf"
    ))

    n_agents = task.num_agents
    config_name = f"examples/emtom_{n_agents}_robots"

    from omegaconf import open_dict
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=config_path, version_base=None):
        config = compose(config_name=config_name)

    # Emulate Hydra runtime context (required by fix_config for interpolation resolution)
    HydraConfig().cfg = config
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "./outputs/rl_dry_run"

    fix_config(config)
    config = setup_config(config, seed=47668090)

    register_sensors(config)
    register_actions(config)
    register_measures(config)

    dataset = CollaborationDatasetV0(config.habitat.dataset)
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        print(f"  Warning: World graph init: {e}")

    env = EmtomMultiAgentEnv(
        config=config,
        env_interface=env_interface,
        task_pool=[task],
        max_turns=20,
    )

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.task_file is None:
        task_dir = Path("data/emtom/tasks")
        for f in sorted(task_dir.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            if d.get("num_agents") == 2:
                args.task_file = str(f)
                break

    if args.task_file is None:
        print("No 2-agent task found in data/emtom/tasks/")
        sys.exit(1)

    dry_run(args.model, args.task_file, args.device)
