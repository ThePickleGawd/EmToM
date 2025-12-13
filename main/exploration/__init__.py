"""
Room exploration module for Habitat environments.

This module provides utilities for systematically exploring all rooms in a scene,
navigating to furniture within each room, and collecting exploration data.
"""

from main.exploration.room_explorer import (
    RoomExplorer,
    RoomExplorationResult,
    FullExplorationResult,
    LiveDisplay,
)

__all__ = ["RoomExplorer", "RoomExplorationResult", "FullExplorationResult", "LiveDisplay"]
