"""Diversity tracking for task generation.

Maintains a persistent log of structural patterns from generated tasks
to guide the LLM toward creating structurally diverse tasks.
"""

import json
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


SUMMARIZE_PROMPT = """Summarize this task's STRUCTURAL PATTERN in one short phrase (5-15 words).

Focus on the MECHANICS, STRUCTURE, and WIN CONDITION - not the narrative:
- What is the win condition? (race to find X, collect items, arrange objects, sabotage opponent, etc.)
- What creates the core dependency? (locked containers, information asymmetry, room restrictions, etc.)
- What forces agent collaboration? (distributed knowledge, sequential handoffs, parallel workstreams, etc.)
- What mechanics are central? (remote triggers, state mirroring, conditional unlocks, etc.)

Examples of good structural summaries:
- "Race to find radio hidden in locked container"
- "Collect and arrange items with distributed location knowledge"
- "Sabotage opponent's progress via remote trigger"
- "Room restriction forces sequential item handoff to unlock goal"
- "Parallel collection with shared final placement objective"
- "Information asymmetry: one knows what, other knows where"

BAD summaries (too generic):
- "Race to find item" (missing: what makes it unique?)
- "Teams compete" (missing: how do they compete?)

Task JSON:
{task_json}

Respond with ONLY the structural pattern phrase, nothing else."""


class DiversityTracker:
    """Tracks structural patterns of generated tasks for diversity."""

    # Persistent location for diversity log (shared across all runs)
    DEFAULT_LOG_DIR = Path("data/emtom/meta")

    def __init__(self, llm: Optional["BaseLLM"] = None, log_dir: Optional[Path] = None):
        """
        Initialize diversity tracker.

        Args:
            llm: LLM for summarizing tasks into patterns (optional, for read-only mode)
            log_dir: Override directory for diversity_log.json (defaults to data/emtom/tasks)
        """
        self.log_dir = Path(log_dir) if log_dir else self.DEFAULT_LOG_DIR
        self.log_file = self.log_dir / "diversity_log.json"
        self.llm = llm
        self.patterns: List[dict] = []
        self._load()

    def _load(self) -> None:
        """Load existing patterns from disk."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    self.patterns = data.get("patterns", [])
            except (json.JSONDecodeError, IOError):
                self.patterns = []

    def _save(self) -> None:
        """Save patterns to disk."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump({"patterns": self.patterns}, f, indent=2)

    def summarize_task(self, task_data: dict) -> str:
        """
        Use LLM to summarize a task into a structural pattern.

        Args:
            task_data: The task JSON data

        Returns:
            Short structural pattern description
        """
        if not self.llm:
            # Fallback: extract basic pattern from task structure
            return self._extract_basic_pattern(task_data)

        prompt = SUMMARIZE_PROMPT.format(task_json=json.dumps(task_data, indent=2))

        try:
            response = self.llm.generate(prompt)
            # Clean up response - take first line, strip whitespace
            pattern = response.strip().split("\n")[0].strip()
            # Remove quotes if present
            pattern = pattern.strip('"\'')
            return pattern[:100]  # Cap length
        except Exception:
            return self._extract_basic_pattern(task_data)

    def _extract_basic_pattern(self, task_data: dict) -> str:
        """Fallback pattern extraction without LLM."""
        parts = []

        # Try to infer win condition from task description and subtasks
        task_desc = task_data.get("task", "").lower()
        category = task_data.get("category", "")

        # Detect common win condition patterns
        if "race" in task_desc or "first" in task_desc or "before" in task_desc:
            parts.append("race")
        elif "collect" in task_desc or "gather" in task_desc:
            parts.append("collection")
        elif "arrange" in task_desc or "place" in task_desc or "organize" in task_desc:
            parts.append("arrangement")
        elif "find" in task_desc or "locate" in task_desc:
            parts.append("search")
        elif "unlock" in task_desc or "open" in task_desc:
            parts.append("unlocking")

        # Check mechanics
        mechanics = task_data.get("active_mechanics", [])
        if not mechanics:
            bindings = task_data.get("mechanic_bindings", []) or task_data.get("mechanics", [])
            extracted = []
            for b in bindings:
                if isinstance(b, dict):
                    mech_type = b.get("mechanic_type") or b.get("type")
                else:
                    mech_type = b
                if mech_type:
                    extracted.append(mech_type)
            # Preserve order, drop duplicates
            seen = set()
            mechanics = []
            for m in extracted:
                if m not in seen:
                    mechanics.append(m)
                    seen.add(m)
        if mechanics:
            parts.append(f"mechanics: {', '.join(mechanics[:2])}")

        # Check locked containers
        if task_data.get("locked_containers"):
            parts.append("locked containers")

        # Check agent secrets
        secrets = task_data.get("agent_secrets", {})
        if any(secrets.values()):
            parts.append("information asymmetry")

        # Check items
        items = task_data.get("items", [])
        if items:
            item_types = set()
            for item in items:
                item_id = item.get("item_id", "")
                if "key" in item_id:
                    item_types.add("keys")
                elif "radio" in item_id:
                    item_types.add("radio")
            if item_types:
                parts.append(f"items: {', '.join(item_types)}")

        # Add category context
        if category:
            parts.insert(0, f"[{category}]")

        return " + ".join(parts) if parts else "basic task structure"

    def add_pattern(self, task_id: str, task_data: dict) -> str:
        """
        Summarize a task and add its pattern to the log.

        Args:
            task_id: Unique identifier for the task
            task_data: The task JSON data

        Returns:
            The generated pattern summary
        """
        pattern = self.summarize_task(task_data)

        self.patterns.append({
            "task_id": task_id,
            "pattern": pattern,
            "category": task_data.get("category", "unknown"),
            "num_agents": task_data.get("num_agents", 0),
            "num_subtasks": len(task_data.get("subtasks", [])),
        })

        self._save()
        return pattern

    def get_patterns_for_prompt(self, limit: int = 20) -> str:
        """
        Format recent patterns for injection into prompts.

        Args:
            limit: Maximum number of patterns to include

        Returns:
            Formatted string for prompt injection
        """
        if not self.patterns:
            return "No previous tasks yet. Be creative with your first task structure!"

        recent = self.patterns[-limit:]
        lines = []
        for i, p in enumerate(recent, 1):
            lines.append(f"{i}. [{p['category']}] {p['pattern']}")

        return "\n".join(lines)

    def get_pattern_count(self) -> int:
        """Return total number of tracked patterns."""
        return len(self.patterns)
