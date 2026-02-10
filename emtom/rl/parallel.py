"""
Parallel episode collection for RL training.

Follows the subprocess + GPU round-robin pattern from
emtom/evolve/benchmark_wrapper.py, adapted for RL episode rollouts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from emtom.rl.grpo import Trajectory, TurnData


def _detect_gpu_ids() -> List[int]:
    """Detect available CUDA GPU IDs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().split("\n")]
    except Exception:
        pass
    return [0]


def trajectory_to_dict(traj: Trajectory) -> Dict[str, Any]:
    """Serialize a Trajectory to a JSON-compatible dict."""
    return {
        "episode_reward": traj.episode_reward,
        "task_id": traj.task_id,
        "turns": [
            {
                "agent_id": td.agent_id,
                "prompt_token_ids": td.prompt_token_ids,
                "completion_token_ids": td.completion_token_ids,
                "logprobs": td.logprobs,
                "step_reward": td.step_reward,
            }
            for td in traj.turns
        ],
    }


def trajectory_from_dict(d: Dict[str, Any]) -> Trajectory:
    """Deserialize a Trajectory from a dict."""
    turns = [
        TurnData(
            agent_id=td["agent_id"],
            prompt_token_ids=td["prompt_token_ids"],
            completion_token_ids=td["completion_token_ids"],
            logprobs=td["logprobs"],
            step_reward=td.get("step_reward", 0.0),
        )
        for td in d.get("turns", [])
    ]
    return Trajectory(
        turns=turns,
        episode_reward=d.get("episode_reward", 0.0),
        task_id=d.get("task_id", ""),
    )


def run_episodes_parallel(
    task_pool: List[Any],
    model_name: str,
    group_size: int = 4,
    max_workers: int = 4,
    max_turns: int = 20,
    output_dir: str = "outputs/rl_episodes",
    temperature: float = 0.7,
    config_name: str = "examples/emtom_2_robots",
) -> List[List[Trajectory]]:
    """
    Run episodes in parallel — one subprocess per (task, episode) pair.

    For each task, spawns `group_size` worker processes. Each worker loads
    the model on a single GPU (round-robin assigned) and runs one episode.

    Args:
        task_pool: List of GeneratedTask objects (or anything with .task_id and to_dict()).
        model_name: HuggingFace model name or path.
        group_size: Number of episodes per task.
        max_workers: Maximum concurrent worker processes.
        max_turns: Max turns per episode.
        output_dir: Base directory for task/episode files.
        temperature: Sampling temperature for generation.
        config_name: Hydra config name for Habitat.

    Returns:
        List of lists: trajectory_groups[task_idx] = [traj_0, traj_1, ...].
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_dir = out_path / "logs"
    log_dir.mkdir(exist_ok=True)

    gpu_ids = _detect_gpu_ids()
    worker_script = os.path.join(os.path.dirname(__file__), "worker.py")

    # Prepare all jobs: (task_idx, episode_idx, task_file, output_file)
    jobs = []
    for t_idx, task in enumerate(task_pool):
        task_id = getattr(task, "task_id", f"task_{t_idx}")
        task_dir = out_path / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Write task JSON
        task_file = task_dir / "task.json"
        if hasattr(task, "to_dict"):
            task_data = task.to_dict()
        else:
            task_data = task
        with open(task_file, "w") as f:
            json.dump(task_data, f, indent=2)

        for ep_idx in range(group_size):
            output_file = task_dir / f"trajectory_ep{ep_idx}.json"
            jobs.append((t_idx, ep_idx, str(task_file), str(output_file)))

    total_jobs = len(jobs)
    job_idx = 0
    active: List[tuple] = []  # (t_idx, ep_idx, output_file, proc, fh)
    completed = []

    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0

    print(f"[parallel] {total_jobs} episodes across {len(task_pool)} tasks, max_workers={max_workers}")
    print(f"[parallel] GPUs: {gpu_ids} ({len(gpu_ids)} devices)")

    try:
        while True:
            # Reap finished processes
            still_active = []
            for t_idx, ep_idx, output_file, proc, fh in active:
                if proc.poll() is not None:
                    fh.close()
                    if proc.returncode != 0:
                        print(
                            f"[parallel] WARNING: worker for task {t_idx} ep {ep_idx} "
                            f"exited with code {proc.returncode}",
                            file=sys.stderr,
                        )
                    completed.append((t_idx, ep_idx, output_file))
                else:
                    still_active.append((t_idx, ep_idx, output_file, proc, fh))
            active = still_active

            # Spawn new workers
            while len(active) < max_workers and job_idx < total_jobs:
                t_idx, ep_idx, task_file, output_file = jobs[job_idx]

                cmd = [
                    sys.executable, worker_script,
                    "--task-file", task_file,
                    "--model", model_name,
                    "--episode-id", str(ep_idx),
                    "--output-file", output_file,
                    "--config-name", config_name,
                    "--max-turns", str(max_turns),
                    "--temperature", str(temperature),
                ]

                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

                log_file = log_dir / f"worker_t{t_idx}_ep{ep_idx}.log"
                fh = open(log_file, "w")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)
                active.append((t_idx, ep_idx, output_file, proc, fh))
                job_idx += 1

            done = len(completed)
            if done >= total_jobs and not active:
                break

            spinner = spinner_chars[spinner_idx % len(spinner_chars)]
            spinner_idx += 1
            status = f"[parallel] {spinner} episodes: {done}/{total_jobs} (active={len(active)})"
            if sys.stdout.isatty():
                print(f"\r{status}", end="", flush=True)

            if not active and job_idx >= total_jobs:
                break

            time.sleep(2)
    finally:
        for _, _, _, proc, fh in active:
            proc.terminate()
        for _, _, _, proc, fh in active:
            proc.wait()
            fh.close()

    if sys.stdout.isatty():
        print()
    print(f"[parallel] Done: {len(completed)}/{total_jobs} episodes collected")

    # Collect trajectories grouped by task
    trajectory_groups: List[List[Trajectory]] = [[] for _ in task_pool]

    for t_idx, ep_idx, output_file in completed:
        try:
            with open(output_file) as f:
                traj_data = json.load(f)
            traj = trajectory_from_dict(traj_data)
            trajectory_groups[t_idx].append(traj)
        except Exception as e:
            print(f"[parallel] WARNING: Could not load trajectory from {output_file}: {e}")

    return trajectory_groups
