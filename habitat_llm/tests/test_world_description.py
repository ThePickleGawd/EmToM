#!/usr/bin/env python3

from habitat_llm.llm.instruct.utils import get_world_descr
from habitat_llm.world_model import Furniture, House, Object, Room
from habitat_llm.world_model.world_graph import WorldGraph


def _build_world_graph(include_object: bool = False) -> WorldGraph:
    graph = WorldGraph()
    house = House("house", {"type": "house"})
    room = Room("kitchen_0", {"type": "room"})
    table = Furniture("table_0", {"type": "furniture", "components": []})

    graph.add_node(house)
    graph.add_node(room)
    graph.add_node(table)
    graph.add_edge(house, room, "has", "in")
    graph.add_edge(room, table, "has", "in")

    if include_object:
        obj = Object("apple_0", {"type": "object"})
        graph.add_node(obj)
        graph.add_edge(table, obj, "under", "on")

    return graph


def test_world_description_starts_with_static_layout_and_no_objects() -> None:
    description = get_world_descr(_build_world_graph(), include_room_name=True)

    assert "Furniture:" in description
    assert "kitchen_0: table_0" in description
    assert "Objects:\nNo objects found yet" in description


def test_world_description_uses_exact_object_handles_for_discovered_objects() -> None:
    description = get_world_descr(_build_world_graph(include_object=True), include_room_name=True)

    assert "apple_0: table_0 in kitchen_0" in description
