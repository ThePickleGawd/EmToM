"""
Exploration runner for EMTOM benchmark.

Uses CuriosityModel for LLM-guided action selection and SurpriseDetector
to identify unexpected outcomes during exploration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .base import EMTOMBaseRunner

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


class ExplorationRunner(EMTOMBaseRunner):
    """
    Runner for LLM-guided exploration with curiosity model.

    This wraps the existing HabitatExplorer functionality but uses
    the standardized EMTOMBaseRunner setup.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.curiosity_model = None
        self.surprise_detector = None
        self.trajectory_logger = None
        self._llm_client = None  # LLM client for agent tools
        self._explorer = None  # Internal HabitatExplorer instance

    def setup(
        self,
        env_interface: "EnvironmentInterface",
        task_data: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        max_steps: int = 50,
        save_video: bool = True,
        save_fpv: bool = True,
    ) -> None:
        """
        Setup exploration runner.

        Args:
            env_interface: Initialized EnvironmentInterface
            task_data: Optional task data with mechanics
            output_dir: Output directory
            max_steps: Maximum exploration steps
            save_video: Whether to save third-person video
            save_fpv: Whether to save first-person video
        """
        # Use base setup for environment and game manager
        super().setup(env_interface, task_data, output_dir)

        # Setup exploration-specific components
        self._setup_curiosity()
        self._setup_trajectory_logger()

        # Create exploration config
        from emtom.exploration.habitat_explorer import HabitatExplorationConfig

        self._exploration_config = HabitatExplorationConfig(
            max_steps=max_steps,
            agent_ids=["agent_0"],  # Single agent for exploration
            log_path=self.output_dir,
            save_video=save_video,
            play_video=False,
            save_fpv=save_fpv,
        )

    def _setup_curiosity(self) -> None:
        """Initialize curiosity model and surprise detector."""
        from habitat_llm.llm import instantiate_llm
        from emtom.exploration.curiosity import CuriosityModel
        from emtom.exploration.surprise_detector import SurpriseDetector

        print("[ExplorationRunner] Setting up LLM client...")
        self._llm_client = instantiate_llm("openai_chat")
        print(f"[ExplorationRunner] Using model: {self._llm_client.generation_params.model}")

        # Get LLM config if available
        llm_config = None
        if hasattr(self.config, 'evaluation') and hasattr(self.config.evaluation, 'agents'):
            agent_list = list(self.config.evaluation.agents.values())
            if agent_list and hasattr(agent_list[0], 'planner') and hasattr(agent_list[0].planner, 'llm'):
                llm_config = agent_list[0].planner.llm

        self.curiosity_model = CuriosityModel(self._llm_client, llm_config=llm_config)
        self.surprise_detector = SurpriseDetector(self._llm_client)

        # Pass LLM to agent tools (required for FindReceptacleTool etc.)
        for uid, agent in self.agents.items():
            agent.pass_llm_to_tools(self._llm_client)
            print(f"[ExplorationRunner] Passed LLM to agent_{uid} tools")

        print("[ExplorationRunner] Curiosity model and surprise detector ready")

    def _setup_trajectory_logger(self) -> None:
        """Initialize trajectory logger."""
        from emtom.exploration.trajectory_logger import TrajectoryLogger

        self.trajectory_logger = TrajectoryLogger(
            output_dir=self.output_dir,
            snapshot_frequency=0,
        )
        print(f"[ExplorationRunner] Trajectory logger ready: {self.output_dir}")

    def run(
        self,
        max_steps: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run exploration loop.

        Args:
            max_steps: Override max steps (uses config value if None)
            metadata: Optional metadata to include in trajectory

        Returns:
            Episode data dict with trajectory, surprises, video paths
        """
        from emtom.exploration.habitat_explorer import HabitatExplorer

        # Update max_steps if provided
        if max_steps is not None:
            self._exploration_config.max_steps = max_steps

        # Get agent (use first agent for exploration)
        agent = self.agents.get(0)

        # Create HabitatExplorer with our setup
        self._explorer = HabitatExplorer(
            env_interface=self.env_interface,
            game_manager=self.game_manager,
            curiosity_model=self.curiosity_model,
            surprise_detector=self.surprise_detector,
            agent=agent,
            config=self._exploration_config,
        )

        # Run exploration
        print(f"\n[ExplorationRunner] Starting exploration for {self._exploration_config.max_steps} steps...")
        episode_data = self._explorer.run(metadata=metadata)

        # Copy trajectory to standard location
        self._copy_trajectory(episode_data)

        return episode_data

    def _copy_trajectory(self, episode_data: Dict[str, Any]) -> None:
        """Copy trajectory file to data/emtom/trajectories for task generation."""
        import shutil
        from pathlib import Path

        episode_id = episode_data.get("episode_id", "unknown")
        source_file = f"{self.output_dir}/trajectory_{episode_id}.json"

        if os.path.exists(source_file):
            dest_dir = Path("data/emtom/trajectories")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"trajectory_{episode_id}.json"
            shutil.copy2(source_file, dest_file)
            print(f"[ExplorationRunner] Trajectory copied to: {dest_file}")

    def get_surprise_moments(self) -> List[Dict[str, Any]]:
        """Get list of surprise moments detected during exploration."""
        if self._explorer:
            return [s.to_dict() for s in self._explorer.surprise_moments]
        return []
