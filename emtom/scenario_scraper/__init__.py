"""
Scenario Scraper for EMTOM.

Scrapes scenario premises from video games, movies, and interactive fiction,
then uses LLM to generate EMTOM-compatible scenario narratives.

Usage:
    python -m emtom.scenario_scraper.run scrape    # Scrape premises from web
    python -m emtom.scenario_scraper.run generate  # Generate scenarios with LLM
    python -m emtom.scenario_scraper.run all       # Both steps

Dependencies:
    pip install beautifulsoup4 requests openai
"""

# Lazy imports to avoid import errors when dependencies not installed
__all__ = ["SOURCES", "ScenarioScraper", "ScenarioGenerator"]


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name == "SOURCES":
        from .sources import SOURCES
        return SOURCES
    elif name == "ScenarioScraper":
        from .scraper import ScenarioScraper
        return ScenarioScraper
    elif name == "ScenarioGenerator":
        from .generator import ScenarioGenerator
        return ScenarioGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
