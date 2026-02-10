"""
GRPO training loop for EmToM multi-agent RL.

Uses HuggingFace model.generate() for inference (with logprobs extraction)
and the same model for gradient updates.
Episode-level GRPO: N rollouts per task, group-relative advantages.

For production: swap generate_batch() with vLLM when Python 3.10+ is available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from emtom.rl.grpo import Trajectory, TurnData, compute_advantages, grpo_loss


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load HuggingFace model and tokenizer for training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[train] Loading model: {model_name}")
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
    print(f"[train] Model loaded: {model.config.hidden_size}d, {param_count:.0f}M params on {device}")
    return model, tokenizer


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    prompts: List[str],
    tokenizer,
    temperature: float = 0.7,
    max_tokens: int = 256,
    stop: Optional[List[str]] = None,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Generate completions for a batch of prompts using HuggingFace model.generate().

    Extracts per-token logprobs from the generation output.

    Returns list of dicts with keys: text, token_ids, logprobs
    """
    stop = stop or ["Assigned!"]

    # Encode stop tokens for stopping criteria
    stop_token_ids = []
    for s in stop:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids)

    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=4096 - max_tokens).to(device)
        prompt_len = input_ids.shape[1]

        # Generate with sampling
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = output.sequences[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Truncate at stop sequence
        for s in stop:
            idx = generated_text.find(s)
            if idx >= 0:
                generated_text = generated_text[:idx]
                # Re-encode to get correct token count
                truncated_ids = tokenizer.encode(generated_text, add_special_tokens=False)
                generated_ids = torch.tensor(truncated_ids, device=device)
                break

        # Extract logprobs from scores
        logprobs = []
        scores = output.scores  # Tuple of (vocab_size,) tensors, one per generated token
        n_tokens = min(len(generated_ids), len(scores))
        for i in range(n_tokens):
            log_probs = F.log_softmax(scores[i][0], dim=-1)
            token_id = generated_ids[i].item() if isinstance(generated_ids, torch.Tensor) else generated_ids[i]
            logprobs.append(log_probs[token_id].item())

        token_ids = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)

        results.append({
            "text": generated_text,
            "token_ids": token_ids[:n_tokens],
            "logprobs": logprobs,
        })

    return results


def run_episode(
    env,
    model: torch.nn.Module,
    tokenizer,
    max_turns: int = 20,
    temperature: float = 0.7,
    device: str = "cuda",
) -> Trajectory:
    """
    Run one full episode, collecting training data.

    Each turn:
      1. Get prompts for all agents
      2. Generate via HF model (sequential per agent) with logprobs
      3. Step env with generated actions
      4. Store TurnData per agent
    """
    obs, info = env.reset()
    turns: List[TurnData] = []
    total_reward = 0.0
    task_id = info[env.agents[0]].get("task_id", "unknown")

    for t in range(max_turns):
        agent_ids = list(env.agents)
        prompts = [obs[aid] for aid in agent_ids]

        # Generate (sequential per agent, but could batch with padding)
        completions = generate_batch(
            model,
            prompts,
            tokenizer,
            temperature=temperature,
            stop=["Assigned!"],
            device=device,
        )

        # Build actions dict and turn data
        actions = {}
        turn_entries = []
        for agent_id, prompt, completion in zip(agent_ids, prompts, completions):
            text = completion["text"]
            if "Assigned!" not in text:
                text = text.rstrip() + "\nAssigned!"
            actions[agent_id] = text

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            td = TurnData(
                agent_id=agent_id,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion["token_ids"],
                logprobs=completion["logprobs"],
                step_reward=0.0,
            )
            turn_entries.append(td)

        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Update step rewards
        for td in turn_entries:
            td.step_reward = rewards.get(td.agent_id, 0.0)
            total_reward += td.step_reward
        turns.extend(turn_entries)

        if all(terms.get(aid, False) for aid in agent_ids) or all(
            truncs.get(aid, False) for aid in agent_ids
        ):
            break

    return Trajectory(
        turns=turns,
        episode_reward=total_reward,
        task_id=task_id,
    )


def train(
    model_name: str,
    task_dir: str = "data/emtom/tasks",
    output_dir: str = "outputs/rl_training",
    num_epochs: int = 3,
    group_size: int = 4,
    max_turns: int = 20,
    lr: float = 1e-5,
    clip_epsilon: float = 0.2,
    kl_coeff: float = 0.01,
    temperature: float = 0.7,
    max_seq_len: int = 4096,
    device: str = "cuda",
    config_name: str = "examples/emtom_2_robots",
    save_every: int = 1,
    tasks_per_epoch: Optional[int] = None,
    advantage_method: str = "grpo",
    parallel_workers: int = 1,
):
    """
    Main GRPO training loop.

    1. Load HuggingFace model (generation + gradient updates)
    2. Load tasks
    3. For each epoch:
       a. Sample batch of tasks
       b. For each task, run N episodes (GRPO group)
       c. Compute advantages
       d. Gradient update
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    ref_model = None  # Could load a frozen copy for KL

    # Load tasks
    tasks = _load_tasks(task_dir)
    if not tasks:
        raise ValueError(f"No tasks found in {task_dir}")
    print(f"[train] Loaded {len(tasks)} tasks")

    # Initialize Habitat env (only needed for sequential mode)
    env = None
    if parallel_workers <= 1:
        env = _init_env(config_name, tasks, max_turns, device)

    print(f"[train] Advantage method: {advantage_method}, parallel_workers: {parallel_workers}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    stats_log = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_tasks = tasks[:tasks_per_epoch] if tasks_per_epoch else tasks
        np.random.shuffle(epoch_tasks)

        epoch_rewards = []
        epoch_losses = []

        if parallel_workers > 1:
            # Parallel episode collection across all tasks
            from emtom.rl.parallel import run_episodes_parallel

            episode_dir = os.path.join(output_dir, f"episodes_epoch{epoch+1}")
            trajectory_groups = run_episodes_parallel(
                task_pool=epoch_tasks,
                model_name=model_name,
                group_size=group_size,
                max_workers=parallel_workers,
                max_turns=max_turns,
                output_dir=episode_dir,
                temperature=temperature,
                config_name=config_name,
            )

            for task_idx, (task, trajectories) in enumerate(zip(epoch_tasks, trajectory_groups)):
                if not trajectories:
                    print(f"\n[Epoch {epoch+1}] Task {task_idx+1}: {task.title} — no trajectories collected")
                    continue

                print(f"\n[Epoch {epoch+1}/{num_epochs}] Task {task_idx+1}/{len(epoch_tasks)}: {task.title}")
                for ep_i, traj in enumerate(trajectories):
                    print(f"  Episode {ep_i+1}: reward={traj.episode_reward:.3f}, turns={len(traj.turns)}")

                advantages = compute_advantages(trajectories, method=advantage_method)
                rewards = [t.episode_reward for t in trajectories]
                epoch_rewards.extend(rewards)
                print(f"  Advantages: {[f'{a:.2f}' for a in advantages]}")
                print(f"  Rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")

                optimizer.zero_grad()
                loss = grpo_loss(
                    model=model,
                    trajectories=trajectories,
                    advantages=advantages,
                    tokenizer=tokenizer,
                    clip_epsilon=clip_epsilon,
                    kl_coeff=kl_coeff,
                    ref_model=ref_model,
                    max_seq_len=max_seq_len,
                    device=device,
                )

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                print(f"  Loss: {loss_val:.4f}")
        else:
            # Sequential episode collection
            for task_idx, task in enumerate(epoch_tasks):
                print(f"\n[Epoch {epoch+1}/{num_epochs}] Task {task_idx+1}/{len(epoch_tasks)}: {task.title}")

                trajectories = []
                for ep_i in range(group_size):
                    model.eval()
                    traj = run_episode(
                        env,
                        model,
                        tokenizer,
                        max_turns=max_turns,
                        temperature=temperature,
                        device=device,
                    )
                    model.train()
                    traj.task_id = task.task_id
                    trajectories.append(traj)
                    n_agents = max(len(env.agents), 1)
                    print(f"  Episode {ep_i+1}/{group_size}: reward={traj.episode_reward:.3f}, "
                          f"turns={len(traj.turns)//n_agents}")

                advantages = compute_advantages(trajectories, method=advantage_method)
                rewards = [t.episode_reward for t in trajectories]
                epoch_rewards.extend(rewards)
                print(f"  Advantages: {[f'{a:.2f}' for a in advantages]}")
                print(f"  Rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")

                optimizer.zero_grad()
                loss = grpo_loss(
                    model=model,
                    trajectories=trajectories,
                    advantages=advantages,
                    tokenizer=tokenizer,
                    clip_epsilon=clip_epsilon,
                    kl_coeff=kl_coeff,
                    ref_model=ref_model,
                    max_seq_len=max_seq_len,
                    device=device,
                )

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                print(f"  Loss: {loss_val:.4f}")

        # Epoch stats
        elapsed = time.time() - epoch_start
        stats = {
            "epoch": epoch + 1,
            "mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "std_reward": float(np.std(epoch_rewards)) if epoch_rewards else 0.0,
            "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "elapsed_s": elapsed,
            "num_tasks": len(epoch_tasks),
        }
        stats_log.append(stats)
        print(f"\n[Epoch {epoch+1}] reward={stats['mean_reward']:.3f}+/-{stats['std_reward']:.3f}, "
              f"loss={stats['mean_loss']:.4f}, time={elapsed:.0f}s")

        # Save checkpoint
        if save_every and (epoch + 1) % save_every == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Saved checkpoint to {ckpt_dir}")

    # Save final stats
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(stats_log, f, indent=2)

    print(f"\n[train] Done! {num_epochs} epochs, {len(stats_log)} logged.")
    return stats_log


def _load_tasks(task_dir: str) -> list:
    """Load GeneratedTask objects from a task directory."""
    from emtom.task_gen.task_generator import GeneratedTask

    tasks = []
    task_path = Path(task_dir)

    if not task_path.exists():
        return tasks

    for json_file in sorted(task_path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if "tasks" in data:
                for t in data["tasks"]:
                    tasks.append(GeneratedTask.from_dict(t))
            else:
                tasks.append(GeneratedTask.from_dict(data))
        except Exception as e:
            print(f"[train] Warning: Could not load {json_file}: {e}")

    return tasks


def _init_env(
    config_name: str,
    tasks: list,
    max_turns: int,
    device: str,
):
    """Initialize Habitat environment and EmtomMultiAgentEnv."""
    from hydra import compose, initialize_config_dir

    from habitat_llm.agent.env import (
        EnvironmentInterface,
        register_actions,
        register_measures,
        register_sensors,
    )
    from habitat_llm.agent.env.dataset import CollaborationDatasetV0
    from habitat_llm.utils import fix_config, setup_config

    from emtom.rl.env import EmtomMultiAgentEnv

    # Load Hydra config
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../habitat_llm/conf",
    )
    config_path = os.path.abspath(config_path)

    from omegaconf import open_dict
    from hydra.core.hydra_config import HydraConfig

    with initialize_config_dir(config_dir=config_path, version_base=None):
        config = compose(config_name=config_name)

    # Emulate Hydra runtime context (required by fix_config for interpolation resolution)
    HydraConfig().cfg = config
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "./outputs/rl_training"

    fix_config(config)
    config = setup_config(config, seed=47668090)

    # Register Habitat components
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Create dataset and env
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    try:
        env_interface.initialize_perception_and_world_graph()
    except Exception as e:
        print(f"[train] Warning: World graph init: {e}")

    # Create RL env
    env = EmtomMultiAgentEnv(
        config=config,
        env_interface=env_interface,
        task_pool=tasks,
        max_turns=max_turns,
    )

    print(f"[train] Environment initialized with {len(tasks)} tasks")
    return env


def main():
    parser = argparse.ArgumentParser(description="GRPO training for EmToM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--task-dir", type=str, default="data/emtom/tasks", help="Directory with task JSONs")
    parser.add_argument("--output-dir", type=str, default="outputs/rl_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size (episodes per task)")
    parser.add_argument("--max-turns", type=int, default=20, help="Max turns per episode")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--kl-coeff", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--config-name", type=str, default="examples/emtom_2_robots", help="Hydra config name")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--tasks-per-epoch", type=int, default=None, help="Limit tasks per epoch")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model")
    parser.add_argument("--advantage-method", type=str, default="grpo",
                        choices=["grpo", "drgrpo", "rloo"], help="Advantage computation method")
    parser.add_argument("--parallel-workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    args = parser.parse_args()

    train(
        model_name=args.model,
        task_dir=args.task_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        group_size=args.group_size,
        max_turns=args.max_turns,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        temperature=args.temperature,
        max_seq_len=args.max_seq_len,
        config_name=args.config_name,
        save_every=args.save_every,
        tasks_per_epoch=args.tasks_per_epoch,
        device=args.device,
        advantage_method=args.advantage_method,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    main()
