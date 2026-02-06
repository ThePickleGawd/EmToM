"""Configuration for evolutionary difficulty task generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


DEFAULT_MODEL_LADDER = [
    "ministral-3-8b",
    "haiku",
    "gpt-5-mini",
    "sonnet",
    "gpt-5.1",
    "gpt-5.2",
]


@dataclass
class EvolutionConfig:
    model_ladder: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_LADDER))
    generator_model: str = "gpt-5.2"
    tasks_per_round: int = 20
    seed_pool_size: int = 30
    icl_failure_ratio: float = 0.9
    icl_total_examples: int = 10
    output_dir: str = "outputs/evolve"
    max_workers: int = 50
    # Match run_emtom.sh default judge threshold.
    judge_threshold: float = 0.7
    seed_query: str = (
        "Simple 2-agent cooperative task. No mechanics. "
        "Basic object rearrangement with minimal information asymmetry."
    )
