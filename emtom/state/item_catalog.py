"""
Item Catalog for EMTOM Inventory System.

Provides a catalog of items that can be randomly selected and instantiated at runtime.

Items:
- Small Key: Common key that unlocks small containers (drawers, cabinets, boxes)
- Big Key: Rare key that unlocks final game objectives
- Radio: Enables the Communicate action for inter-agent communication
"""

import copy
import random
from typing import Dict, List, Optional

from emtom.state.items import ItemDefinition, ItemType


# ============ KEY ITEMS ============
KEY_ITEMS = [
    ItemDefinition(
        item_id="small_key",
        name="Small Key",
        description="A small brass key that unlocks drawers and cabinets.",
        item_type=ItemType.KEY,
    ),
    ItemDefinition(
        item_id="big_key",
        name="Big Key",
        description="A large ornate key. This looks important.",
        item_type=ItemType.KEY,
    ),
]


# ============ TOOL ITEMS ============
TOOL_ITEMS = [
    ItemDefinition(
        item_id="radio",
        name="Two-Way Radio",
        description="A handheld radio for communication with others.",
        item_type=ItemType.TOOL,
        grants_action="Communicate",
        action_description="Communicate[message]: Send a message to other agents.",
        consumable=False,
    ),
]


# Full catalog indexed by item_id
ITEM_CATALOG: Dict[str, ItemDefinition] = {
    item.item_id: item
    for item in KEY_ITEMS + TOOL_ITEMS
}


def get_random_items(
    count: int = 1,
    item_type: Optional[ItemType] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ItemDefinition]:
    """
    Get random items from the catalog.

    Args:
        count: Number of items to select
        item_type: Filter by ItemType (KEY or TOOL). None = any type.
        exclude_ids: List of item_ids to exclude from selection

    Returns:
        List of ItemDefinition copies with unique instance IDs
    """
    exclude = set(exclude_ids or [])
    candidates = [
        item for item in ITEM_CATALOG.values()
        if item.item_id not in exclude
        and (item_type is None or item.item_type == item_type)
    ]

    if not candidates:
        return []

    selected = random.sample(candidates, min(count, len(candidates)))

    # Return copies with unique instance IDs
    result = []
    for i, item in enumerate(selected):
        instance = copy.deepcopy(item)
        instance.item_id = f"{item.item_id}_{i+1}"  # Unique instance ID
        result.append(instance)
    return result


def get_random_small_keys(count: int = 1) -> List[ItemDefinition]:
    """
    Get multiple small keys (common, can find many).

    Args:
        count: Number of small keys to generate

    Returns:
        List of small key ItemDefinitions with unique IDs
    """
    result = []
    template = ITEM_CATALOG.get("small_key")
    if template:
        for i in range(count):
            instance = copy.deepcopy(template)
            instance.item_id = f"small_key_{i+1}"
            result.append(instance)
    return result


def get_big_key() -> ItemDefinition:
    """
    Get the big key (rare, unlocks final objective).

    Returns:
        Big key ItemDefinition with unique ID
    """
    template = ITEM_CATALOG.get("big_key")
    if template:
        instance = copy.deepcopy(template)
        instance.item_id = "big_key_1"
        return instance
    return None


def get_radio() -> ItemDefinition:
    """
    Get a radio (enables Communicate action).

    Returns:
        Radio ItemDefinition with unique ID
    """
    template = ITEM_CATALOG.get("radio")
    if template:
        instance = copy.deepcopy(template)
        instance.item_id = "radio_1"
        return instance
    return None


def get_item_template(item_id: str) -> Optional[ItemDefinition]:
    """
    Get an item template by its base ID.

    Args:
        item_id: The base item ID (e.g., "small_key", "big_key", "radio")

    Returns:
        ItemDefinition or None if not found
    """
    return ITEM_CATALOG.get(item_id)


def get_all_keys() -> List[ItemDefinition]:
    """Get all KEY-type items from the catalog."""
    return [item for item in ITEM_CATALOG.values() if item.item_type == ItemType.KEY]


def get_all_tools() -> List[ItemDefinition]:
    """Get all TOOL-type items from the catalog."""
    return [item for item in ITEM_CATALOG.values() if item.item_type == ItemType.TOOL]
