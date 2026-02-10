#!/usr/bin/env python
"""
Single-episode worker — spawned by parallel.py.

Loads a task, initializes Habitat, runs one episode with an HF model,
and writes the trajectory to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from emtom.rl.grpo import Trajectory, TurnData
from emtom.rl.parallel import trajectory_to_dict


def run_worker(
    task_file: str,
    model_name: str,
    episode_id: int,
    output_file: str,
    config_name: str = "examples/emtom_2_robots",
    max_turns: int = 20,
    temperature: float = 0.7,
):
    """Run a single episode and write trajectory JSON."""
    print(f"[worker] Episode {episode_id}, task={task_file}, model={model_name}")

    # Load task
    from emtom.task_gen.task_generator import GeneratedTask

    with open(task_file) as f:
        task_data = json.load(f)
    task = GeneratedTask.from_dict(task_data)

    # Init Habitat env (same pattern as dry_run._init_habitat_env)
    env = _init_habitat_env(task, config_name)

    # Load HF model on cuda:0 (parent set CUDA_VISIBLE_DEVICES)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from emtom.rl.train import load_model_and_tokenizer, run_episode

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    model.eval()

    # Run one episode
    traj = run_episode(
        env,
        model,
        tokenizer,
        max_turns=max_turns,
        temperature=temperature,
        device=device,
    )
    traj.task_id = task.task_id

    # Write trajectory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(trajectory_to_dict(traj), f, indent=2)

    print(f"[worker] Done. Reward={traj.episode_reward:.4f}, "
          f"turns={len(traj.turns)}, output={output_file}")


def _init_habitat_env(task, config_name):
    """Initialize Habitat env for a single task."""
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from hydra.core.hydra_config import HydraConfig

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
    actual_config_name = f"examples/emtom_{n_agents}_robots"

    with initialize_config_dir(config_dir=config_path, version_base=None):
        config = compose(config_name=actual_config_name)

    HydraConfig().cfg = config
    with open_dict(config):
        config.hydra = {}
        config.hydra.runtime = {}
        config.hydra.runtime.output_dir = "./outputs/rl_worker"

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
        print(f"[worker] Warning: World graph init: {e}")

    env = EmtomMultiAgentEnv(
        config=config,
        env_interface=env_interface,
        task_pool=[task],
        max_turns=20,
    )

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL single-episode worker")
    parser.add_argument("--task-file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="examples/emtom_2_robots")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_worker(
        task_file=args.task_file,
        model_name=args.model,
        episode_id=args.episode_id,
        output_file=args.output_file,
        config_name=args.config_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )
