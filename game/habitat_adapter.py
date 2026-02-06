"""
Adapter that exposes the minimal EnvironmentAdapter API on top of the
EnvironmentInterface used in PARTNR/Habitat.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from game.game import EnvironmentAdapter
from habitat_llm.world_model.entity import Room


class HabitatEnvironmentAdapter(EnvironmentAdapter):
    """
    Translates game-level calls into EnvironmentInterface operations.
    """

    def __init__(self, env_interface):
        self.env = env_interface

    def list_rooms(self) -> List[str]:
        rooms = self.env.full_world_graph.get_all_rooms()
        return [room.name for room in rooms]

    def list_objects(self) -> List[str]:
        """
        Return all known object names from the world graph.
        """
        try:
            return [obj.name for obj in self.env.full_world_graph.get_all_objects()]
        except Exception:
            return []

    def list_switches(self) -> List[str]:
        """
        Placeholder for switch enumeration if environment exposes them;
        currently unused but kept for API symmetry.
        """
        return []

    def get_agent_room(self, agent_id: str) -> Optional[str]:
        agent_name = f"agent_{agent_id}"
        try:
            agent_node = self.env.full_world_graph.get_node_from_name(agent_name)
            neighbors = self.env.full_world_graph.get_neighbors_of_type(
                agent_node, Room
            )
            if neighbors:
                return neighbors[0].name
        except Exception:
            return None
        return None

    def move_agent_to_room(self, agent_id: str, room_name: str) -> Dict[str, Any]:
        """
        Basic teleporter that moves the articulated agent base to the center of the room's bounding box.
        For safety, we rely on the navmesh snap. If anything fails, return an error instead of throwing.
        """
        sim = self.env.sim
        if (
            sim is None
            or not hasattr(sim, "semantic_scene")
            or sim.semantic_scene is None
        ):
            return {"ok": False, "error": "Simulator not ready"}

        target_region = None
        for region in sim.semantic_scene.regions:
            if getattr(region, "id", None) is not None and hasattr(region, "id"):
                # region.id maps to region_id_to_name in PerceptionSim; name may be similar to room_name
                if room_name in self.env.perception.region_id_to_name.values():
                    # find matching by mapped name
                    if self.env.perception.region_id_to_name.get(region.id) == room_name:
                        target_region = region
                        break
            if getattr(region, "name", "") == room_name:
                target_region = region
                break
        if target_region is None:
            return {"ok": False, "error": f"Unknown room {room_name}"}

        center = target_region.aabb.center()
        snapped = sim.pathfinder.snap_point(center, sim.largest_island_idx)
        if snapped is None:
            return {"ok": False, "error": "Could not snap to navmesh"}

        try:
            agent_mgr = sim.agents_mgr[int(agent_id)]
            agent_mgr.articulated_agent.base_pos = snapped
        except Exception as e:
            return {"ok": False, "error": f"Failed move: {e}"}

        return {"ok": True, "room": room_name}

    def send_message(self, sender_id: str, receiver_id: str, message: str) -> None:
        try:
            self.env.post_agent_message(int(sender_id), message, target_uids=[int(receiver_id)])
        except Exception:
            # Fallback to no-op if types mismatch; game layer can still track messages if desired.
            pass
