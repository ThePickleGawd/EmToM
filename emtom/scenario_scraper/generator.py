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

    SYSTEM_PROMPT = """You are a creative writer adapting entertainment premises into household puzzle scenarios for a two-agent collaboration benchmark.

CONSTRAINTS:
- Setting: A residential house with common rooms (kitchen, bedroom, living room, bathroom, study, basement, attic, etc.)
- Agents: Two people who must work together to solve puzzles
- Elements: Include puzzles, hidden items, locked containers, clues, and mysteries
- Tone: Immersive narrative, paragraph format, engaging prose
- Length: 3-5 paragraphs (200-400 words)

YOUR TASK:
Adapt the given premise into a household puzzle scenario. Keep the mystery/puzzle essence but ground it in realistic domestic items (furniture, appliances, everyday objects like cabinets, drawers, fridges, tables, beds, etc.).

The scenario should:
1. Set up an intriguing situation that requires exploration
2. Hint at hidden items or secrets to discover
3. Suggest that collaboration between the two agents is beneficial
4. Create tension or urgency without being too dark
5. Feel like a puzzle/escape room experience in a normal house

Output ONLY the narrative text. No titles, headers, metadata, or explanations."""

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
        user_prompt = f"""Adapt this premise into a household puzzle scenario:

Title: {premise.get('title', 'Unknown')}
Category: {premise.get('category', 'unknown')}

Original Premise:
{premise.get('premise', '')}

Write an immersive scenario narrative for two agents exploring a house together. The scenario should feel like an escape room or mystery game set in a residential home."""

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
        user_prompt = f"""Create an original {theme} scenario for two agents exploring a house.

The scenario should be set in a normal residential home and feel like a puzzle/escape room experience. Include hints about hidden items, locked containers, and mysteries to solve.

Write 3-5 paragraphs of immersive narrative."""

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
