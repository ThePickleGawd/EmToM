"""Exploration system for EMTOM benchmark."""

from dataclasses import dataclass, field
from typing import List

from emtom.exploration.curiosity import (
    ActionChoice,
    CuriosityModel,
)
from emtom.exploration.surprise_detector import (
    SurpriseAssessment,
    SurpriseDetector,
)
from emtom.exploration.trajectory_logger import (
    SurpriseRecord,
    StepRecord,
    TrajectoryLogger,
)

# Habitat-integrated exploration
from emtom.exploration.habitat_explorer import (
    HabitatExplorationConfig,
    HabitatExplorer,
    HabitatWorldAdapter,
    HabitatStepResult,
)


@dataclass
class ExplorationConfig:
    """
    Configuration for EMTOM exploration.

    NOTE: For Habitat-based exploration, use HabitatExplorationConfig instead.
    This class is for legacy/generic exploration configuration.
    """

    max_steps: int = 100
    agent_ids: List[str] = field(default_factory=lambda: ["agent_0"])
    log_path: str = "data/trajectories/emtom"
    snapshot_frequency: int = 0  # 0 = no snapshots
    stop_on_terminal: bool = True


__all__ = [
    # Config
    "ExplorationConfig",
    # Curiosity
    "ActionChoice",
    "CuriosityModel",
    # Habitat Explorer
    "HabitatExplorationConfig",
    "HabitatExplorer",
    "HabitatWorldAdapter",
    "HabitatStepResult",
    # Surprise Detection
    "SurpriseAssessment",
    "SurpriseDetector",
    # Logging
    "SurpriseRecord",
    "StepRecord",
    "TrajectoryLogger",
]
