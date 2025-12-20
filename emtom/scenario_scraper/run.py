#!/usr/bin/env python3
"""
Scenario Scraper CLI for EMTOM.

Usage:
    python -m emtom.scenario_scraper.run scrape              # Scrape premises from web
    python -m emtom.scenario_scraper.run generate            # Generate scenarios from premises
    python -m emtom.scenario_scraper.run generate --scratch  # Generate scenarios without premises
    python -m emtom.scenario_scraper.run all                 # Scrape + generate
    python -m emtom.scenario_scraper.run stats               # Show scraping statistics

Examples:
    # Scrape all curated sources
    python -m emtom.scenario_scraper.run scrape

    # Generate 50 scenarios from scraped premises
    python -m emtom.scenario_scraper.run generate --count 50

    # Generate 20 scenarios from scratch (no scraping)
    python -m emtom.scenario_scraper.run generate --scratch --count 20

    # Full pipeline: scrape then generate 100 scenarios
    python -m emtom.scenario_scraper.run all --count 100
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_scrape(args):
    """Scrape premises from curated sources."""
    from .scraper import ScenarioScraper
    from .sources import SOURCES

    print("=== Scraping Scenario Premises ===")
    print(f"Output directory: {args.premises_dir}")

    scraper = ScenarioScraper(
        output_dir=args.premises_dir,
        rate_limit_delay=args.delay,
    )

    premises = scraper.scrape_all(SOURCES)
    print(f"\nTotal premises scraped: {len(premises)}")

    return premises


def cmd_generate(args):
    """Generate scenarios using LLM."""
    from .generator import ScenarioGenerator

    print("=== Generating Scenarios ===")
    print(f"Model: {args.model}")
    print(f"Count: {args.count}")
    print(f"Output: {args.output_dir}")

    generator = ScenarioGenerator(
        model=args.model,
        temperature=args.temperature,
    )

    if args.scratch:
        # Generate from scratch (no premises needed)
        print("\nGenerating scenarios from scratch (no premises)...")
        files = generator.generate_batch_from_scratch(
            count=args.count,
            output_dir=args.output_dir,
            delay=args.delay,
        )
    else:
        # Load premises and generate
        premises_path = Path(args.premises_dir) / "premises.json"

        if not premises_path.exists():
            print(f"\nError: No premises found at {premises_path}")
            print("Run 'scrape' first, or use --scratch to generate without premises.")
            sys.exit(1)

        with open(premises_path) as f:
            premises = json.load(f)

        print(f"\nLoaded {len(premises)} premises")

        # Limit to requested count
        premises = premises[: args.count]

        files = generator.generate_all(
            premises=premises,
            output_dir=args.output_dir,
            delay=args.delay,
        )

    print(f"\nGenerated {len(files)} scenario files")
    return files


def cmd_all(args):
    """Run full pipeline: scrape + generate."""
    print("=== Full Pipeline: Scrape + Generate ===\n")

    # Step 1: Scrape
    premises = cmd_scrape(args)

    if not premises:
        print("\nNo premises scraped, using scratch generation...")
        args.scratch = True

    # Step 2: Generate
    files = cmd_generate(args)

    print(f"\n=== Pipeline Complete ===")
    print(f"Premises: {len(premises)}")
    print(f"Scenarios: {len(files)}")


def cmd_stats(args):
    """Show statistics about scraped data."""
    from .scraper import ScenarioScraper

    scraper = ScenarioScraper(output_dir=args.premises_dir)
    stats = scraper.get_stats()

    print("=== Scraping Statistics ===")
    print(f"Total premises: {stats.get('total', 0)}")
    print("\nBy category:")
    for key, value in sorted(stats.items()):
        if key != "total":
            print(f"  {key}: {value}")

    # Count generated scenarios
    output_path = Path(args.output_dir)
    if output_path.exists():
        txt_files = list(output_path.glob("*.txt"))
        print(f"\nGenerated scenarios: {len(txt_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Scraper for EMTOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "action",
        choices=["scrape", "generate", "all", "stats"],
        help="Action to perform",
    )

    # Output directories
    parser.add_argument(
        "--output-dir",
        default="data/emtom/scenarios/scraped",
        help="Directory for generated .txt files",
    )
    parser.add_argument(
        "--premises-dir",
        default="data/emtom/scenarios/raw",
        help="Directory for raw premises JSON",
    )

    # Generation options
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of scenarios to generate (default: 100)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="LLM model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Generate scenarios from scratch without premises",
    )

    # Rate limiting
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Dispatch to action handler
    if args.action == "scrape":
        cmd_scrape(args)
    elif args.action == "generate":
        cmd_generate(args)
    elif args.action == "all":
        cmd_all(args)
    elif args.action == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
