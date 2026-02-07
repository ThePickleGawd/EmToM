"""Configuration for evolutionary difficulty task generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_MODEL_LADDER = [
    "gpt-5-mini",
    "sonnet",
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
    # Generate until pass rate drops to this percent, then advance to next model.
    target_pass_rate: float = 20.0
    # Optional seed query. When None, generator uses its own default prompting.
    seed_query: Optional[str] = None
