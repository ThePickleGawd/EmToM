"""
Clue Generator for EMTOM Scenario System.

Generates three types of clues that point to item locations:
1. Category hints: "The key is hidden in furniture"
2. Room hints: "Check the kitchen carefully"
3. Riddle-style: "Where food stays cold, secrets are told"

Clues are designed to guide without being too obvious.
"""

import random
import re
from typing import Dict, List, Optional


# =============================================================================
# OBJECT TYPE DETECTION
# =============================================================================

# Patterns to detect object types from Habitat object names
OBJECT_TYPE_PATTERNS = {
    "cabinet": [r"cabinet", r"cupboard", r"armoire", r"wardrobe"],
    "drawer": [r"drawer", r"chest_of_drawers", r"dresser", r"nightstand"],
    "fridge": [r"fridge", r"refrigerator", r"freezer", r"icebox"],
    "table": [r"table", r"desk", r"counter", r"bench"],
    "shelf": [r"shelf", r"bookshelf", r"shelving", r"rack"],
    "closet": [r"closet", r"locker", r"storage"],
    "box": [r"box", r"crate", r"container", r"bin"],
    "bed": [r"bed", r"mattress", r"cot"],
    "couch": [r"couch", r"sofa", r"loveseat", r"settee"],
    "chair": [r"chair", r"stool", r"seat"],
    "appliance": [r"washer", r"dryer", r"oven", r"microwave", r"dishwasher"],
}

# Semantic categories for objects
OBJECT_CATEGORIES = {
    "storage": ["cabinet", "drawer", "closet", "box", "shelf"],
    "furniture": ["table", "bed", "couch", "chair"],
    "appliance": ["fridge", "appliance"],
    "cold_storage": ["fridge"],
    "seating": ["couch", "chair", "bed"],
}


def get_object_type(object_id: str) -> str:
    """
    Determine the type of object from its ID.

    Args:
        object_id: Habitat object identifier (e.g., "cabinet_42", "fridge_17")

    Returns:
        Object type string (e.g., "cabinet", "fridge", "unknown")
    """
    object_lower = object_id.lower()

    for obj_type, patterns in OBJECT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, object_lower):
                return obj_type

    return "unknown"


def get_object_category(object_id: str) -> str:
    """
    Get the semantic category of an object.

    Args:
        object_id: Habitat object identifier

    Returns:
        Category string (e.g., "storage", "furniture", "appliance")
    """
    obj_type = get_object_type(object_id)

    for category, types in OBJECT_CATEGORIES.items():
        if obj_type in types:
            return category

    return "furniture"  # Default category


# =============================================================================
# CLUE TEMPLATES
# =============================================================================

# Category clue templates by object category
CATEGORY_CLUE_TEMPLATES = {
    "storage": [
        "The key is hidden inside storage furniture",
        "Look in places where things are kept",
        "Check containers with doors or drawers",
        "Hidden among stored items",
        "Search where objects are tucked away",
    ],
    "furniture": [
        "The key is hidden in furniture",
        "Look in everyday household furniture",
        "Check the furniture in the room",
        "Hidden in plain sight, among common furniture",
    ],
    "appliance": [
        "Look in household appliances",
        "Check the machines in the room",
        "Hidden in something that runs on power",
    ],
    "cold_storage": [
        "Look where things are kept cold",
        "Check where food is preserved",
        "Hidden in something that keeps things fresh",
    ],
    "seating": [
        "Look near where people sit or rest",
        "Check comfortable furniture",
        "Hidden in restful places",
    ],
}

# Room clue templates
ROOM_CLUE_TEMPLATES = [
    "Search the {room} carefully",
    "The answer lies in the {room}",
    "Focus your attention on the {room}",
    "The {room} holds what you seek",
    "Look around the {room}",
    "Explore the {room} thoroughly",
    "The {room} is your best bet",
    "Something is hidden in the {room}",
]

# Riddle templates by object type
RIDDLE_TEMPLATES = {
    "cabinet": [
        "Behind wooden doors, treasures are stored",
        "Look where dishes hide from view",
        "What has doors but isn't a room?",
        "Hinges creak to reveal what's inside",
        "Tall and wooden, holding kitchen secrets",
    ],
    "drawer": [
        "Where papers rest and drawers slide, secrets often like to hide",
        "Pull to reveal what's concealed",
        "Sliding secrets wait in rows",
        "What opens with a pull, not a turn?",
        "Stacked horizons of hidden things",
    ],
    "fridge": [
        "Where food stays cold, secrets are told",
        "Chill seekers find more than they expected",
        "Cold comfort holds warm secrets",
        "What keeps things fresh but hides things too?",
        "Behind the cold door, mysteries wait",
    ],
    "table": [
        "Beneath where meals are shared, something waits prepared",
        "Four legs support more than you know",
        "Where we gather, secrets scatter",
        "Look under where you eat",
    ],
    "shelf": [
        "Between the books, secrets took",
        "Height helps hide what's tucked inside",
        "Rows of knowledge guard the key",
        "What holds books might hold more",
    ],
    "closet": [
        "In the dark behind the door, find what you're looking for",
        "Where clothes hang, secrets swing",
        "Open the door to find what's stored",
        "Privacy conceals more than clothes",
    ],
    "box": [
        "Six sides keep secrets inside",
        "What's packed away might save the day",
        "Cardboard walls guard hidden halls",
        "Lift the lid to find what's hid",
    ],
    "bed": [
        "Where dreams rest, look beneath for the best",
        "Under slumber, treasures number",
        "Sleep guards secrets deep",
        "The restful place hides a hidden space",
    ],
    "couch": [
        "Between cushions, fortunes hide",
        "Comfort conceals what's beneath the seals",
        "Sink in to find what's hidden within",
        "Soft surfaces hide hard evidence",
    ],
    "chair": [
        "Take a seat and find what's discrete",
        "Beneath where you rest, find the best",
        "Four legs hold more than meets the eye",
    ],
    "appliance": [
        "Machines keep more than power",
        "What runs and hums might hide some sums",
        "Electric secrets in metal chests",
    ],
    "unknown": [
        "Look carefully and you shall find",
        "Hidden in plain sight",
        "What seems ordinary may surprise",
        "The answer is closer than you think",
    ],
}


# =============================================================================
# CLUE GENERATOR CLASS
# =============================================================================

class ClueGenerator:
    """
    Generate clues that point to item locations without being too obvious.

    Three clue types:
    - Category: Points to a type of object (furniture, storage, appliance)
    - Room: Points to a specific room
    - Riddle: Poetic hint that describes the object
    """

    def generate_category_clue(self, container: str) -> str:
        """
        Generate a category clue for a container.

        Args:
            container: The Habitat object ID (e.g., "cabinet_42")

        Returns:
            A hint about the category of object (e.g., "Hidden in storage furniture")
        """
        category = get_object_category(container)
        templates = CATEGORY_CLUE_TEMPLATES.get(category, CATEGORY_CLUE_TEMPLATES["furniture"])
        return random.choice(templates)

    def generate_room_clue(self, container: str, room: str) -> str:
        """
        Generate a room clue.

        Args:
            container: The Habitat object ID (unused, for interface consistency)
            room: The room name (e.g., "kitchen", "living room")

        Returns:
            A hint about which room to search
        """
        # Clean up room name
        room_clean = room.replace("_", " ").title()
        template = random.choice(ROOM_CLUE_TEMPLATES)
        return template.format(room=room_clean)

    def generate_riddle_clue(self, container: str) -> str:
        """
        Generate a riddle-style clue for a container.

        Args:
            container: The Habitat object ID (e.g., "fridge_17")

        Returns:
            A poetic hint about the object type
        """
        obj_type = get_object_type(container)
        templates = RIDDLE_TEMPLATES.get(obj_type, RIDDLE_TEMPLATES["unknown"])
        return random.choice(templates)

    def generate_all_clues(
        self,
        container: str,
        room: str,
    ) -> List[Dict[str, str]]:
        """
        Generate all three clue types for a container.

        Args:
            container: The Habitat object ID
            room: The room name

        Returns:
            List of clue dicts with type, text, and points_to fields
        """
        return [
            {
                "type": "category",
                "text": self.generate_category_clue(container),
                "points_to": container,
            },
            {
                "type": "room",
                "text": self.generate_room_clue(container, room),
                "points_to": container,
            },
            {
                "type": "riddle",
                "text": self.generate_riddle_clue(container),
                "points_to": container,
            },
        ]


# Convenience functions
def generate_category_clue(container: str) -> str:
    """Generate a category clue for a container."""
    return ClueGenerator().generate_category_clue(container)


def generate_room_clue(container: str, room: str) -> str:
    """Generate a room clue."""
    return ClueGenerator().generate_room_clue(container, room)


def generate_riddle_clue(container: str) -> str:
    """Generate a riddle-style clue for a container."""
    return ClueGenerator().generate_riddle_clue(container)
