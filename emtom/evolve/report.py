"""Report generation for evolutionary difficulty pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from emtom.evolve.benchmark_wrapper import BenchmarkResults
from emtom.evolve.config import EvolutionConfig


def generate_report(
    config: EvolutionConfig,
    tier_results: Dict[str, BenchmarkResults],
    output_dir: str,
) -> None:
    """Generate report.json and report.md in the output directory.

    Args:
        config: Evolution configuration.
        tier_results: Mapping of tier name -> BenchmarkResults.
        output_dir: Directory to write reports to.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_data = {
        "config": asdict(config),
        "tiers": {},
    }

    for tier_name, results in tier_results.items():
        tier_data = {
            "model": results.model,
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "pass_rate": results.pass_rate,
            "avg_completion": _avg_completion(results),
            "results": [asdict(r) for r in results.results],
        }
        report_data["tiers"][tier_name] = tier_data

    with open(out / "report.json", "w") as f:
        json.dump(report_data, f, indent=2)

    md = _build_markdown(config, tier_results)
    with open(out / "report.md", "w") as f:
        f.write(md)

    print(f"[evolve] Reports written to {out}/report.json and {out}/report.md")


def _avg_completion(results: BenchmarkResults) -> float:
    """Compute average percent_complete across non-skipped results."""
    if not results.results:
        return 0.0
    total = sum(r.percent_complete for r in results.results)
    return total / len(results.results)


def _build_markdown(
    config: EvolutionConfig,
    tier_results: Dict[str, BenchmarkResults],
) -> str:
    """Build a human-readable Markdown report."""
    lines = [
        "# Evolutionary Difficulty Report",
        "",
        f"**Generator model**: {config.generator_model}",
        f"**Tasks per round**: {config.tasks_per_round}",
        f"**Seed pool size**: {config.seed_pool_size}",
        f"**Model ladder**: {' -> '.join(config.model_ladder)}",
        "",
        "## Results by Tier",
        "",
        "| Tier | Model | Tasks | Pass Rate | Avg Completion |",
        "|------|-------|-------|-----------|----------------|",
    ]

    for i, (tier_name, results) in enumerate(tier_results.items()):
        avg_comp = _avg_completion(results)
        lines.append(
            f"| {i + 1} | {results.model} | {results.total} | "
            f"{results.pass_rate:.1f}% | {avg_comp:.1%} |"
        )

    lines.append("")
    return "\n".join(lines)
