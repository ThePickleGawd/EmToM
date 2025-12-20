"""
Web scraper for scenario premises.

Scrapes plot summaries and game descriptions from Wikipedia and other sources.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


class ScenarioScraper:
    """Scrape plot summaries from various sources."""

    # Headers to mimic a browser request
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Section headers to look for in Wikipedia articles
    PLOT_SECTIONS = [
        "plot",
        "gameplay",
        "premise",
        "story",
        "synopsis",
        "narrative",
        "plot summary",
        "game mechanics",
        "overview",
    ]

    def __init__(
        self,
        output_dir: str = "data/emtom/scenarios/raw",
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory to save scraped premises
            rate_limit_delay: Seconds to wait between requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _clean_text(self, text: str) -> str:
        """Clean scraped text by removing citations and extra whitespace."""
        # Remove citation brackets [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        return text.strip()

    def _find_section(
        self, soup: BeautifulSoup, section_names: List[str]
    ) -> Optional[Any]:
        """Find a section header by name."""
        for header in soup.find_all(["h2", "h3"]):
            # Get the text content, handling span elements
            header_text = header.get_text().strip().lower()
            # Remove [edit] links
            header_text = re.sub(r"\[edit\]", "", header_text).strip()

            for section_name in section_names:
                if section_name in header_text:
                    return header
        return None

    def _get_section_content(self, section_header) -> List[str]:
        """Get paragraphs from a section until the next header."""
        paragraphs = []
        for sibling in section_header.find_next_siblings():
            # Stop at next header
            if sibling.name in ["h2", "h3"]:
                break
            # Collect paragraph text
            if sibling.name == "p":
                text = self._clean_text(sibling.get_text())
                if text and len(text) > 50:  # Skip very short paragraphs
                    paragraphs.append(text)
            # Also check for nested paragraphs in divs
            elif sibling.name == "div":
                for p in sibling.find_all("p"):
                    text = self._clean_text(p.get_text())
                    if text and len(text) > 50:
                        paragraphs.append(text)
        return paragraphs

    def scrape_wikipedia(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract main content from a Wikipedia article.

        Grabs all readable text and lets the LLM decide what's relevant.

        Args:
            url: Wikipedia URL to scrape

        Returns:
            Dict with source, title, and content, or None if failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error fetching {url}: {e}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Get article title
        title_elem = soup.find("h1", {"id": "firstHeading"})
        title = title_elem.get_text().strip() if title_elem else "Unknown"

        # Get all content from the main article body
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            print(f"  No content found for: {title}")
            return None

        # Collect all paragraphs from the article
        paragraphs = []
        for p in content.find_all("p"):
            text = self._clean_text(p.get_text())
            if text and len(text) > 30:  # Skip very short paragraphs
                paragraphs.append(text)

        if not paragraphs:
            print(f"  No text content for: {title}")
            return None

        # Join all paragraphs (limit to ~3000 chars to avoid token limits)
        full_text = "\n\n".join(paragraphs)
        if len(full_text) > 3000:
            full_text = full_text[:3000] + "..."

        return {
            "source": url,
            "title": title,
            "premise": full_text,
        }

    def scrape_url(self, url: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single URL with category tagging.

        Args:
            url: URL to scrape
            category: Category label for this source

        Returns:
            Dict with scraped data, or None if failed
        """
        print(f"  Scraping: {url}")

        # Determine scraper based on URL
        if "wikipedia.org" in url:
            result = self.scrape_wikipedia(url)
        else:
            # Default to Wikipedia-style scraping
            result = self.scrape_wikipedia(url)

        if result:
            result["category"] = category

        return result

    def scrape_all(
        self,
        sources: Dict[str, List[str]],
        save_intermediate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scrape all sources and return list of premises.

        Args:
            sources: Dict mapping category names to lists of URLs
            save_intermediate: Whether to save results after each URL

        Returns:
            List of scraped premise dicts
        """
        premises = []
        failed_urls = []

        for category, urls in sources.items():
            if category == "search_queries":
                continue  # Handle separately

            print(f"\n[{category}] Scraping {len(urls)} URLs...")

            for url in urls:
                result = self.scrape_url(url, category)

                if result:
                    premises.append(result)
                    print(f"    Success: {result['title']}")
                else:
                    failed_urls.append({"url": url, "category": category})

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                # Save intermediate results
                if save_intermediate and len(premises) % 10 == 0:
                    self._save_premises(premises)

        # Final save
        self._save_premises(premises)

        # Save failed URLs for debugging
        if failed_urls:
            failed_path = self.output_dir / "failed_urls.json"
            with open(failed_path, "w") as f:
                json.dump(failed_urls, f, indent=2)
            print(f"\nFailed URLs saved to: {failed_path}")

        print(f"\n=== Scraping Complete ===")
        print(f"Scraped: {len(premises)} premises")
        print(f"Failed: {len(failed_urls)} URLs")

        return premises

    def _save_premises(self, premises: List[Dict[str, Any]]) -> None:
        """Save premises to JSON file."""
        output_path = self.output_dir / "premises.json"
        with open(output_path, "w") as f:
            json.dump(premises, f, indent=2, ensure_ascii=False)

    def load_premises(self) -> List[Dict[str, Any]]:
        """Load previously scraped premises."""
        premises_path = self.output_dir / "premises.json"
        if premises_path.exists():
            with open(premises_path) as f:
                return json.load(f)
        return []

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about scraped premises."""
        premises = self.load_premises()
        stats = {"total": len(premises)}

        # Count by category
        for premise in premises:
            category = premise.get("category", "unknown")
            stats[category] = stats.get(category, 0) + 1

        return stats
