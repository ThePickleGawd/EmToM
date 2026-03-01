"""Configuration for evolutionary difficulty task generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_MODEL_LADDER = [
    "gpt-5-mini",
    "sonnet",
    "gpt-5.2",
]

DEFAULT_EVOLVE_FOCUS = "either"
DEFAULT_EVOLVE_CATEGORY = "any"


@dataclass
class EvolutionConfig:
    model_ladder: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_LADDER))
    generator_model: str = "gpt-5.2"
    tasks_per_round: int = 20
    icl_failure_ratio: float = 0.9
    icl_total_examples: int = 10
    # Destination directory for evolved task JSONs.
    output_dir: str = "data/emtom/tasks"
    max_workers: int = 50
    # Match run_emtom.sh default judge threshold.
    judge_threshold: float = 0.7
    # Generate until pass rate drops to this percent, then advance to next model.
    target_pass_rate: float = 20.0
    # Source directory for seed tasks (None = generate from scratch).
    seed_tasks_dir: Optional[str] = "data/emtom/tasks"
    # Minimum seed pool size — generate extra if copied tasks < this.
    seed_pool_size: int = 30
    # What to optimize when creating upgraded tasks.
    # difficulty = lower benchmark pass rate
    # tom = higher-order ToM distribution
    # either = both (default)
    focus: str = DEFAULT_EVOLVE_FOCUS
    # Category selection for newly generated upgrades.
    # Supports single category, comma-separated list, or "any"/"all".
    category: str = DEFAULT_EVOLVE_CATEGORY
    # Allowed k-levels for task generation (None = random per task).
    k_levels: Optional[List[int]] = None
