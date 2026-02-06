"""
Scenario System for EMTOM Task Generation.

Provides scenario templates that define videogame-like settings for tasks.
Each scenario includes:
- Theme and narrative framing
- Required items to find
- Clue templates for item locations
- Theory of Mind elements (knowledge splits, collaboration)
"""

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ScenarioTemplate:
    """
    A scenario template that can be instantiated with actual scene objects.

    Scenarios define the narrative framing for a task, including:
    - Story text with placeholders
    - Required items to find
    - Clue templates for each item
    - Optional collaboration/ToM requirements
    """
    id: str
    theme: str  # escape_room, treasure_hunt, mystery, collaboration
    title_template: str
    story_template: str
    items_needed: List[str]
    clue_templates: Dict[str, str]  # {category, room, riddle}
    mechanics_compatible: List[str] = field(default_factory=list)
    requires_collaboration: bool = False
    requires_both_agents: bool = False
    agent_knowledge_split: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "theme": self.theme,
            "title_template": self.title_template,
            "story_template": self.story_template,
            "items_needed": self.items_needed,
            "clue_templates": self.clue_templates,
            "mechanics_compatible": self.mechanics_compatible,
            "requires_collaboration": self.requires_collaboration,
            "requires_both_agents": self.requires_both_agents,
            "agent_knowledge_split": self.agent_knowledge_split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioTemplate":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            theme=data["theme"],
            title_template=data["title_template"],
            story_template=data["story_template"],
            items_needed=data.get("items_needed", ["small_key"]),
            clue_templates=data.get("clue_templates", {}),
            mechanics_compatible=data.get("mechanics_compatible", []),
            requires_collaboration=data.get("requires_collaboration", False),
            requires_both_agents=data.get("requires_both_agents", False),
            agent_knowledge_split=data.get("agent_knowledge_split"),
        )


@dataclass
class InstantiatedScenario:
    """
    A scenario instantiated with actual scene objects.

    Contains the filled-in story text, actual item locations,
    and generated clues pointing to those locations.
    """
    template: ScenarioTemplate
    title: str
    story_context: str
    suggested_locations: List[str]
    item_locations: Dict[str, str]  # item_id -> container
    clues: List[Dict[str, str]]  # [{type, text, points_to}, ...]
    agent_secrets: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "template_id": self.template.id,
            "theme": self.template.theme,
            "title": self.title,
            "story_context": self.story_context,
            "suggested_locations": self.suggested_locations,
            "item_locations": self.item_locations,
            "clues": self.clues,
            "agent_secrets": self.agent_secrets,
            "requires_collaboration": self.template.requires_collaboration,
            "requires_both_agents": self.template.requires_both_agents,
        }


class ScenarioLoader:
    """Load and manage scenario templates."""

    _instance = None
    _templates: Dict[str, List[ScenarioTemplate]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_templates()
        return cls._instance

    def _load_templates(self) -> None:
        """Load all scenario templates from YAML files."""
        scenarios_dir = Path(__file__).parent.parent.parent / "data" / "emtom" / "scenarios"

        for yaml_file in scenarios_dir.glob("*.yaml"):
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            if data:
                for theme, scenarios in data.items():
                    if isinstance(scenarios, list):
                        if theme not in self._templates:
                            self._templates[theme] = []
                        for scenario_data in scenarios:
                            template = ScenarioTemplate.from_dict(scenario_data)
                            self._templates[theme].append(template)

    def get_all_themes(self) -> List[str]:
        """Get all available scenario themes."""
        return list(self._templates.keys())

    def get_all_scenarios(self, theme: Optional[str] = None) -> List[ScenarioTemplate]:
        """Get all scenarios, optionally filtered by theme."""
        if theme:
            return self._templates.get(theme, [])
        return [s for scenarios in self._templates.values() for s in scenarios]

    def get_random_scenario(self, theme: Optional[str] = None) -> Optional[ScenarioTemplate]:
        """Get a random scenario, optionally filtered by theme."""
        scenarios = self.get_all_scenarios(theme)
        if not scenarios:
            return None
        return copy.deepcopy(random.choice(scenarios))

    def get_compatible_scenario(
        self,
        mechanics: List[str],
        theme: Optional[str] = None,
    ) -> Optional[ScenarioTemplate]:
        """Get a scenario compatible with the given mechanics."""
        scenarios = self.get_all_scenarios(theme)

        # Filter to scenarios compatible with at least one mechanic
        compatible = [
            s for s in scenarios
            if not s.mechanics_compatible or any(m in s.mechanics_compatible for m in mechanics)
        ]

        if not compatible:
            # Fall back to any scenario if no compatible ones found
            compatible = scenarios

        if not compatible:
            return None

        return copy.deepcopy(random.choice(compatible))

    def get_collaboration_scenario(self) -> Optional[ScenarioTemplate]:
        """Get a scenario that requires collaboration."""
        all_scenarios = self.get_all_scenarios()
        collab = [s for s in all_scenarios if s.requires_collaboration or s.requires_both_agents]
        if not collab:
            return None
        return copy.deepcopy(random.choice(collab))


class ScenarioInstantiator:
    """Instantiate scenario templates with actual scene objects."""

    def __init__(self, clue_generator=None):
        """
        Initialize the instantiator.

        Args:
            clue_generator: Optional ClueGenerator instance. If not provided,
                           clues from the template are used directly.
        """
        self.clue_generator = clue_generator

    def instantiate(
        self,
        template: ScenarioTemplate,
        scene_inventory: Dict[str, Any],
        item_locations: Dict[str, str],
        primary_room: str = "room",
    ) -> InstantiatedScenario:
        """
        Instantiate a scenario template with actual scene objects.

        Args:
            template: The scenario template to instantiate
            scene_inventory: Dict with rooms, furniture, objects from the scene
            item_locations: Mapping of item_id -> container_id
            primary_room: The main room name for the scenario

        Returns:
            InstantiatedScenario with filled-in story and clues
        """
        # Generate suggested locations list
        suggested_locations = list(item_locations.values())

        # Add a few decoy locations from the scene
        furniture = scene_inventory.get("furniture", [])
        if furniture:
            decoys = [f for f in furniture if f not in suggested_locations]
            random.shuffle(decoys)
            suggested_locations.extend(decoys[:2])

        # Generate clues for each item
        clues = []
        for item_id, container in item_locations.items():
            room = self._get_room_for_object(container, scene_inventory)

            # Use clue generator if available, otherwise use template clues
            if self.clue_generator:
                clues.append({
                    "type": "category",
                    "text": self.clue_generator.generate_category_clue(container),
                    "points_to": container,
                })
                clues.append({
                    "type": "room",
                    "text": self.clue_generator.generate_room_clue(container, room),
                    "points_to": container,
                })
                clues.append({
                    "type": "riddle",
                    "text": self.clue_generator.generate_riddle_clue(container),
                    "points_to": container,
                })
            else:
                # Use template clues directly
                for clue_type, clue_text in template.clue_templates.items():
                    clues.append({
                        "type": clue_type,
                        "text": clue_text.format(room=room) if "{room}" in clue_text else clue_text,
                        "points_to": container,
                    })

        # Get a riddle clue for the story
        riddle_clue = next(
            (c["text"] for c in clues if c["type"] == "riddle"),
            "Look carefully and you shall find"
        )
        category_clue = next(
            (c["text"] for c in clues if c["type"] == "category"),
            "Hidden in plain sight"
        )
        room_clue = next(
            (c["text"] for c in clues if c["type"] == "room"),
            f"Check the {primary_room}"
        )

        # Format suggested locations for display
        suggested_list = ", ".join(suggested_locations[:5])

        # Instantiate the story template
        story_context = template.story_template.format(
            room=primary_room,
            riddle_clue=riddle_clue,
            category_clue=category_clue,
            room_clue=room_clue,
            suggested_locations=suggested_list,
            agent_specific_clue="{agent_specific_clue}",  # Placeholder for per-agent
            item_name="artifact",  # Default item name
        )

        # Generate title
        title = template.title_template.format(
            room=primary_room,
            item_name="Artifact",
        )

        # Handle agent-specific knowledge
        agent_secrets = None
        if template.agent_knowledge_split:
            agent_secrets = copy.deepcopy(template.agent_knowledge_split)

        return InstantiatedScenario(
            template=template,
            title=title,
            story_context=story_context,
            suggested_locations=suggested_locations,
            item_locations=item_locations,
            clues=clues,
            agent_secrets=agent_secrets,
        )

    def _get_room_for_object(
        self,
        object_id: str,
        scene_inventory: Dict[str, Any],
    ) -> str:
        """Get the room containing an object. Returns 'room' as default."""
        # This would need scene graph access for accurate mapping
        # For now, return a generic room or first room in inventory
        rooms = scene_inventory.get("rooms", ["room"])
        if rooms:
            return rooms[0].replace("_", " ").title()
        return "room"


# Convenience functions
def get_scenario_loader() -> ScenarioLoader:
    """Get the singleton scenario loader instance."""
    return ScenarioLoader()


def get_random_scenario(theme: Optional[str] = None) -> Optional[ScenarioTemplate]:
    """Get a random scenario template."""
    return get_scenario_loader().get_random_scenario(theme)


def get_compatible_scenario(
    mechanics: List[str],
    theme: Optional[str] = None,
) -> Optional[ScenarioTemplate]:
    """Get a scenario compatible with the given mechanics."""
    return get_scenario_loader().get_compatible_scenario(mechanics, theme)
