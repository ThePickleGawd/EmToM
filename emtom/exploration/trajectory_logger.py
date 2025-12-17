"""
Trajectory logging for EMTOM benchmark.

Logs exploration trajectories in human-readable JSON format for:
1. Analysis by task generation pipeline
2. Human debugging and inspection
3. Reproducibility of experiments
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SurpriseRecord:
    """Record of a surprise moment during exploration."""

    step: int
    agent_id: str
    action: str
    target: Optional[str]
    surprise_level: int  # 1-5 scale
    explanation: str
    observation: Optional[str] = None  # The observation that triggered the surprise
    hypothesis: str = ""  # Agent's hypothesis about why this happened
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    """Record of a single exploration step."""

    step: int
    timestamp: str
    agent_actions: Dict[str, Dict[str, Any]]  # agent_id -> {action, target, reasoning}
    effects: List[Dict[str, Any]]
    observations: Dict[str, str]  # agent_id -> observation text
    surprises: List[SurpriseRecord] = field(default_factory=list)
    world_snapshot: Optional[Dict[str, Any]] = None
    inventory: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> list of items

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "agent_actions": self.agent_actions,
            "effects": self.effects,
            "observations": self.observations,
            "surprises": [s.to_dict() for s in self.surprises],
            "world_snapshot": self.world_snapshot,
            "inventory": self.inventory,
        }


class TrajectoryLogger:
    """
    Logs exploration trajectories in human-readable JSON format.

    Features:
    - Structured recording of actions, effects, observations
    - Surprise moment tracking with agent hypotheses
    - World state snapshots at configurable intervals
    - Statistics computation
    - JSON export with pretty formatting
    """

    def __init__(
        self,
        output_dir: str = "data/trajectories/emtom",
        snapshot_frequency: int = 0,  # 0 = no snapshots, N = every N steps
    ):
        """
        Initialize the trajectory logger.

        Args:
            output_dir: Directory to save trajectory files
            snapshot_frequency: How often to save world snapshots (0 = never)
        """
        self.output_dir = output_dir
        self.snapshot_frequency = snapshot_frequency
        os.makedirs(output_dir, exist_ok=True)

        self.current_episode: Dict[str, Any] = {}
        self.steps: List[StepRecord] = []
        self._started = False
        self._messages: List[str] = []  # Internal messages/logs

    def start_episode(
        self,
        agent_ids: List[str],
        mechanics_active: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start logging a new episode.

        Args:
            agent_ids: List of agent IDs in this episode
            mechanics_active: Names of active mechanics
            metadata: Additional metadata (seed, config, etc.)

        Returns:
            Episode ID
        """
        episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_episode = {
            "episode_id": episode_id,
            "start_time": datetime.now().isoformat(),
            "agent_ids": agent_ids,
            "mechanics_active": mechanics_active or [],
            "metadata": metadata or {},
            "scene_inventory": {},  # Will store all objects/furniture in scene
            "steps": [],
            "surprise_summary": [],
            "statistics": {},
        }
        self.steps = []
        self._messages = []
        self._started = True

        return episode_id

    def set_scene_inventory(
        self,
        rooms: List[str],
        furniture: List[str],
        objects: List[str],
        articulated_furniture: List[str],
    ) -> None:
        """
        Set the complete scene inventory.

        This ensures task generation only uses real objects from the scene.

        Args:
            rooms: List of room names in the scene
            furniture: List of all furniture names
            objects: List of all object names (pickable items)
            articulated_furniture: List of furniture that can be opened/closed
        """
        if not self._started:
            raise RuntimeError("Must call start_episode() before setting inventory")

        self.current_episode["scene_inventory"] = {
            "rooms": rooms,
            "furniture": furniture,
            "objects": objects,
            "articulated_furniture": articulated_furniture,
        }

    def log_message(self, message: str) -> None:
        """
        Log a message (for internal tracking, not agent observations).

        Args:
            message: Message to log
        """
        timestamped = f"[{datetime.now().isoformat()}] {message}"
        self._messages.append(timestamped)

    def log_step(
        self,
        step: int,
        agent_actions: Dict[str, Dict[str, Any]],
        effects: List[Dict[str, Any]],
        observations: Dict[str, str],
        surprises: Optional[List[SurpriseRecord]] = None,
        world_snapshot: Optional[Dict[str, Any]] = None,
        inventory: Optional[Dict[str, List[str]]] = None,
    ) -> StepRecord:
        """
        Log a single exploration step.

        Args:
            step: Step number
            agent_actions: Per-agent action info {agent_id: {action, target, reasoning}}
            effects: List of effect dicts
            observations: Per-agent observation strings
            surprises: List of surprise records
            world_snapshot: Optional world state snapshot
            inventory: Per-agent inventory {agent_id: [item1, item2, ...]}

        Returns:
            The created StepRecord
        """
        if not self._started:
            raise RuntimeError("Must call start_episode() before logging steps")

        # Maybe take snapshot
        if self.snapshot_frequency > 0 and step % self.snapshot_frequency == 0:
            if world_snapshot is None:
                world_snapshot = {"note": "Snapshot frequency set but no snapshot provided"}

        record = StepRecord(
            step=step,
            timestamp=datetime.now().isoformat(),
            agent_actions=agent_actions,
            effects=effects,
            observations=observations,
            surprises=surprises or [],
            world_snapshot=world_snapshot,
            inventory=inventory or {},
        )
        self.steps.append(record)
        return record

    def log_surprise(
        self,
        step: int,
        agent_id: str,
        action: str,
        target: Optional[str],
        surprise_level: int,
        explanation: str,
        hypothesis: str = "",
    ) -> SurpriseRecord:
        """
        Log a surprise moment (can be called separately from log_step).

        Args:
            step: Step when surprise occurred
            agent_id: Agent who was surprised
            action: Action that caused surprise
            target: Target of the action
            surprise_level: 1-5 scale of how surprised
            explanation: Why this was surprising
            hypothesis: Agent's hypothesis about cause

        Returns:
            The created SurpriseRecord
        """
        record = SurpriseRecord(
            step=step,
            agent_id=agent_id,
            action=action,
            target=target,
            surprise_level=surprise_level,
            explanation=explanation,
            hypothesis=hypothesis,
        )

        # Add to the corresponding step if it exists
        for step_record in self.steps:
            if step_record.step == step:
                step_record.surprises.append(record)
                break

        return record

    def get_recent_actions(
        self, agent_id: str, n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get the N most recent actions for an agent.

        Useful for providing context to curiosity model.
        """
        actions = []
        for step_record in reversed(self.steps):
            if agent_id in step_record.agent_actions:
                action_info = step_record.agent_actions[agent_id].copy()
                action_info["step"] = step_record.step
                action_info["observation"] = step_record.observations.get(agent_id, "")
                actions.append(action_info)
                if len(actions) >= n:
                    break
        return list(reversed(actions))

    def get_all_surprises(self) -> List[SurpriseRecord]:
        """Get all surprise records from the episode."""
        all_surprises = []
        for step_record in self.steps:
            all_surprises.extend(step_record.surprises)
        return all_surprises

    def finalize_episode(self) -> Dict[str, Any]:
        """
        Finalize the episode and compute statistics.

        Returns:
            Complete episode data as a dictionary
        """
        if not self._started:
            raise RuntimeError("No episode to finalize")

        self.current_episode["end_time"] = datetime.now().isoformat()
        self.current_episode["steps"] = [s.to_dict() for s in self.steps]
        self.current_episode["surprise_summary"] = [
            s.to_dict() for s in self.get_all_surprises()
        ]
        self.current_episode["messages"] = self._messages
        self.current_episode["statistics"] = self._compute_statistics()

        # Save to file
        filename = f"trajectory_{self.current_episode['episode_id']}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.current_episode, f, indent=2)

        self._started = False
        return self.current_episode

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the episode."""
        all_surprises = self.get_all_surprises()

        # Count actions per agent
        actions_per_agent: Dict[str, int] = {}
        unique_actions: set = set()
        for step_record in self.steps:
            for agent_id, action_info in step_record.agent_actions.items():
                actions_per_agent[agent_id] = actions_per_agent.get(agent_id, 0) + 1
                unique_actions.add(action_info.get("action", ""))

        # Surprise statistics
        surprise_by_agent: Dict[str, int] = {}
        surprise_levels: List[int] = []
        for surprise in all_surprises:
            surprise_by_agent[surprise.agent_id] = (
                surprise_by_agent.get(surprise.agent_id, 0) + 1
            )
            surprise_levels.append(surprise.surprise_level)

        return {
            "total_steps": len(self.steps),
            "total_surprises": len(all_surprises),
            "actions_per_agent": actions_per_agent,
            "unique_actions": len(unique_actions),
            "surprises_per_agent": surprise_by_agent,
            "avg_surprise_level": (
                sum(surprise_levels) / len(surprise_levels)
                if surprise_levels
                else 0.0
            ),
        }

    def to_narrative(self) -> str:
        """
        Generate a human-readable narrative of the trajectory.

        Useful for passing to LLM for task generation.
        """
        lines = []
        lines.append(f"Episode: {self.current_episode.get('episode_id', 'unknown')}")
        lines.append(
            f"Agents: {', '.join(self.current_episode.get('agent_ids', []))}"
        )
        lines.append(
            f"Mechanics: {', '.join(self.current_episode.get('mechanics_active', []))}"
        )
        lines.append("")

        for step_record in self.steps:
            lines.append(f"--- Step {step_record.step} ---")
            for agent_id, action_info in step_record.agent_actions.items():
                action = action_info.get("action", "unknown")
                target = action_info.get("target", "")
                target_str = f" on {target}" if target else ""
                lines.append(f"{agent_id}: {action}{target_str}")
                if agent_id in step_record.observations:
                    lines.append(f"  -> {step_record.observations[agent_id]}")

            for surprise in step_record.surprises:
                lines.append(
                    f"  [SURPRISE] {surprise.agent_id}: {surprise.explanation}"
                )
                if surprise.hypothesis:
                    lines.append(f"    Hypothesis: {surprise.hypothesis}")

            lines.append("")

        return "\n".join(lines)

    def load_trajectory(self, filepath: str) -> Dict[str, Any]:
        """Load a trajectory from a JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def list_trajectories(self) -> List[str]:
        """List all trajectory files in the output directory."""
        files = []
        for f in os.listdir(self.output_dir):
            if f.startswith("trajectory_") and f.endswith(".json"):
                files.append(os.path.join(self.output_dir, f))
        return sorted(files)

    def save_prompts(
        self,
        prompts: Dict[str, str],
        traces: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Save LLM prompts and traces for each agent, similar to DecentralizedEvaluationRunner.

        This creates:
        - prompts/{agent_id}/prompt-{episode_id}-{agent_id}.txt
        - traces/{agent_id}/trace-{episode_id}-{agent_id}.txt

        Args:
            prompts: Dict mapping agent_id -> full prompt string
            traces: Optional dict mapping agent_id -> trace string (without special tokens)
        """
        if not self._started and not self.current_episode:
            print("[TrajectoryLogger] Warning: No episode to save prompts for")
            return

        episode_id = self.current_episode.get("episode_id", "unknown")

        # Save prompts
        for agent_id, prompt in prompts.items():
            # Extract numeric ID if agent_id is like "agent_0"
            agent_num = agent_id.split("_")[-1] if "_" in agent_id else agent_id

            prompt_dir = os.path.join(self.output_dir, "prompts", agent_num)
            os.makedirs(prompt_dir, exist_ok=True)

            prompt_file = os.path.join(
                prompt_dir, f"prompt-exploration-{episode_id}-{agent_num}.txt"
            )
            with open(prompt_file, "w") as f:
                f.write(prompt)
            print(f"[TrajectoryLogger] Saved prompt for {agent_id} to {prompt_file}")

        # Save traces if provided
        if traces:
            for agent_id, trace in traces.items():
                agent_num = agent_id.split("_")[-1] if "_" in agent_id else agent_id

                trace_dir = os.path.join(self.output_dir, "traces", agent_num)
                os.makedirs(trace_dir, exist_ok=True)

                trace_file = os.path.join(
                    trace_dir, f"trace-exploration-{episode_id}-{agent_num}.txt"
                )
                with open(trace_file, "w") as f:
                    f.write(trace)
                print(f"[TrajectoryLogger] Saved trace for {agent_id} to {trace_file}")

    def save_planner_log(
        self,
        planner_data: Dict[str, Any],
    ) -> None:
        """
        Save planner log similar to DecentralizedEvaluationRunner.

        Creates:
        - planner-log/planner-log-{episode_id}.json

        Args:
            planner_data: Dict containing planner info per step
        """
        if not self._started and not self.current_episode:
            print("[TrajectoryLogger] Warning: No episode to save planner log for")
            return

        episode_id = self.current_episode.get("episode_id", "unknown")

        log_dir = os.path.join(self.output_dir, "planner-log")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"planner-log-exploration-{episode_id}.json")
        with open(log_file, "w") as f:
            json.dump(planner_data, f, indent=2, default=str)
        print(f"[TrajectoryLogger] Saved planner log to {log_file}")
