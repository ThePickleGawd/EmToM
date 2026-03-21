"""Task generation pipeline for EMTOM benchmark.

This package is used both from inside the full PARTNR/EMTOM codebase and from
lightweight external-agent harnesses.

Some submodules (e.g., the agentic generator) depend on optional heavy
dependencies (omegaconf, model clients, etc.). Importing those at package import
(time) can break the lightweight CLI.

Keep `emtom.task_gen` importable with a minimal environment by only importing
optional components behind `try/except`.
"""

from __future__ import annotations

# Always-available components
from emtom.task_gen.trajectory_analyzer import (
    DiscoveredMechanic,
    TaskOpportunity,
    TrajectoryAnalysis,
    TrajectoryAnalyzer,
)
from emtom.task_gen.task_generator import GeneratedTask, MechanicBinding
from emtom.task_gen.judge import Judge, Judgment, CouncilVerdict, CriterionScore

# Optional components
try:  # pragma: no cover
    from emtom.task_gen.agent import TaskGeneratorAgent  # type: ignore
except Exception:  # pragma: no cover
    TaskGeneratorAgent = None  # type: ignore

__all__ = [
    "DiscoveredMechanic",
    "TaskOpportunity",
    "TrajectoryAnalysis",
    "TrajectoryAnalyzer",
    "GeneratedTask",
    "MechanicBinding",
    "Judge",
    "Judgment",
    "CouncilVerdict",
    "CriterionScore",
    "TaskGeneratorAgent",
]
