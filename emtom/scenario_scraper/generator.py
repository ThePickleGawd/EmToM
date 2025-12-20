"""
LLM-based scenario generator.

Uses gpt-5.2 to adapt scraped premises into EMTOM-compatible household scenarios.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai


class ScenarioGenerator:
    """Use LLM to generate EMTOM scenarios from raw premises."""

    SYSTEM_PROMPT = """You write concise scenario briefings for a two-agent puzzle benchmark.

FORMAT (follow exactly):
1. ONE sentence of minimal context (who you are, why you're here)
2. ONE sentence stating the GOAL explicitly and clearly
3. ONE sentence about what kind of challenges exist (without specifics)

LENGTH: 3-4 sentences total. Maximum 75 words.

STYLE:
- Direct and functional, not literary or atmospheric
- No flowery descriptions, metaphors, or mood-setting
- State the goal in clear, actionable terms

CRITICAL RULES:
- NEVER describe how to solve puzzles
- NEVER reveal where objects are located
- NEVER give step-by-step instructions
- NEVER describe specific actions to take

GOOD EXAMPLE:
"You've been hired to retrieve a valuable item from an abandoned house. Your goal: find and open the locked safe hidden somewhere in the house. The rooms contain puzzles, locked containers, and hidden mechanisms that require cooperation to solve."

BAD EXAMPLE (too verbose):
"The rain taps against the window as shadows dance across the walls. An ornate box sits on the table, its brass corners catching the lamplight mysteriously..."

Output ONLY the scenario text. No titles or metadata."""

    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.8,
    ):
        """
        Initialize the generator.

        Args:
            model: OpenAI model to use
            temperature: Sampling temperature (higher = more creative)
        """
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()

    def generate_scenario(
        self,
        premise: Dict[str, Any],
        retry_count: int = 3,
    ) -> Optional[str]:
        """
        Generate a single scenario from a raw premise.

        Args:
            premise: Dict with title, category, and premise text
            retry_count: Number of retries on failure

        Returns:
            Generated scenario text, or None if failed
        """
        user_prompt = f"""Source: {premise.get('title', 'Unknown')} ({premise.get('category', 'unknown')})

Premise: {premise.get('premise', '')}

Adapt into a 3-4 sentence household puzzle scenario. State the goal explicitly. No atmospheric prose."""

        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except openai.RateLimitError as e:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

            except openai.APIError as e:
                print(f"    API error (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"    Unexpected error: {e}")
                return None

        return None

    def generate_from_scratch(
        self,
        theme: str = "mystery",
        retry_count: int = 3,
    ) -> Optional[str]:
        """
        Generate a scenario without a premise (purely creative).

        Args:
            theme: Theme hint (mystery, escape_room, treasure_hunt, etc.)
            retry_count: Number of retries on failure

        Returns:
            Generated scenario text
        """
        user_prompt = f"""Theme: {theme}

Create a 3-4 sentence household puzzle scenario. State the goal explicitly. No atmospheric prose."""

        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"    Error generating from scratch: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)

        return None

    def generate_all(
        self,
        premises: List[Dict[str, Any]],
        output_dir: str = "data/emtom/scenarios/scraped",
        start_index: int = 1,
        delay: float = 0.5,
    ) -> List[str]:
        """
        Generate scenarios for all premises and save as .txt files.

        Args:
            premises: List of premise dicts
            output_dir: Directory to save .txt files
            start_index: Starting index for filenames
            delay: Seconds to wait between API calls

        Returns:
            List of generated filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []
        failed = []

        print(f"\n=== Generating {len(premises)} scenarios ===\n")

        for i, premise in enumerate(premises):
            idx = start_index + i
            filename = f"scenario_{idx:03d}.txt"
            filepath = output_path / filename

            # Skip if already exists
            if filepath.exists():
                print(f"[{idx}] Skipping {filename} (already exists)")
                generated.append(filename)
                continue

            print(f"[{idx}] Generating from: {premise.get('title', 'Unknown')}")

            scenario = self.generate_scenario(premise)

            if scenario:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(scenario)
                generated.append(filename)
                print(f"    Saved: {filename} ({len(scenario)} chars)")
            else:
                failed.append({"index": idx, "title": premise.get("title")})
                print(f"    FAILED")

            # Rate limiting
            time.sleep(delay)

        # Summary
        print(f"\n=== Generation Complete ===")
        print(f"Generated: {len(generated)} scenarios")
        print(f"Failed: {len(failed)} premises")

        if failed:
            failed_path = output_path / "failed_generation.json"
            with open(failed_path, "w") as f:
                json.dump(failed, f, indent=2)
            print(f"Failed list saved to: {failed_path}")

        return generated

    def generate_batch_from_scratch(
        self,
        count: int = 10,
        themes: Optional[List[str]] = None,
        output_dir: str = "data/emtom/scenarios/scraped",
        start_index: int = 1,
        delay: float = 0.5,
    ) -> List[str]:
        """
        Generate multiple scenarios from scratch (no premises).

        Args:
            count: Number of scenarios to generate
            themes: List of themes to cycle through
            output_dir: Directory to save .txt files
            start_index: Starting index for filenames
            delay: Seconds between API calls

        Returns:
            List of generated filenames
        """
        if themes is None:
            themes = [
                "mystery",
                "escape_room",
                "treasure_hunt",
                "family_secret",
                "haunted_house",
                "detective",
                "time_pressure",
                "collaboration_puzzle",
            ]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated = []

        print(f"\n=== Generating {count} scenarios from scratch ===\n")

        for i in range(count):
            idx = start_index + i
            theme = themes[i % len(themes)]
            filename = f"scenario_{idx:03d}.txt"
            filepath = output_path / filename

            if filepath.exists():
                print(f"[{idx}] Skipping {filename} (already exists)")
                generated.append(filename)
                continue

            print(f"[{idx}] Generating {theme} scenario...")

            scenario = self.generate_from_scratch(theme)

            if scenario:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(scenario)
                generated.append(filename)
                print(f"    Saved: {filename}")
            else:
                print(f"    FAILED")

            time.sleep(delay)

        print(f"\n=== Generated {len(generated)} scenarios ===")
        return generated


def load_scraped_scenarios(
    directory: str = "data/emtom/scenarios/scraped",
) -> List[str]:
    """
    Load pure-text scenarios from scraped .txt files.

    Args:
        directory: Path to directory containing .txt files

    Returns:
        List of scenario texts
    """
    path = Path(directory)
    scenarios = []

    for txt_file in sorted(path.glob("*.txt")):
        with open(txt_file, encoding="utf-8") as f:
            scenarios.append(f.read())

    return scenarios
