"""
Epistemic layer for EmToM PDDL.

Derives observability models from task structure (room restrictions,
mechanic bindings) to determine what each agent can/cannot observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from emtom.task_gen.task_generator import GeneratedTask


@dataclass
class ObservabilityModel:
    """
    Models what each agent can and cannot observe.

    Derived automatically from room_restrictions + mechanic_bindings.
    Used to construct epistemic init states and compute ToM depth.
    """

    # agent -> rooms they cannot enter/see
    restricted_rooms: Dict[str, Set[str]] = field(default_factory=dict)

    # mechanic effects that are hidden from some agents
    # Maps: trigger_object -> set of agents who can't observe the effect
    hidden_effects: Dict[str, Set[str]] = field(default_factory=dict)

    # agent -> set of agents they can message (None = unrestricted)
    message_targets: Dict[str, Optional[Set[str]]] = field(default_factory=dict)

    # agent -> message limit (None = unlimited)
    message_limits: Dict[str, Optional[int]] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: "GeneratedTask") -> "ObservabilityModel":
        """Build observability model from a GeneratedTask."""
        model = cls()
        num_agents = task.num_agents
        all_agents = {f"agent_{i}" for i in range(num_agents)}

        for binding in task.mechanic_bindings:
            mtype = binding.mechanic_type

            if mtype == "room_restriction":
                rooms = set(binding.restricted_rooms or [])
                for agent in (binding.for_agents or []):
                    model.restricted_rooms.setdefault(agent, set()).update(rooms)

            elif mtype in ("remote_control", "state_mirroring"):
                # If the target object is in a different room than the trigger,
                # agents at the trigger can't directly observe the target effect.
                trigger = binding.trigger_object
                if trigger:
                    # All agents are potentially unaware of remote effects
                    # unless they're in the target room. This is a conservative
                    # over-approximation refined at planning time.
                    model.hidden_effects[trigger] = set(all_agents)

            elif mtype == "limited_bandwidth":
                # Extract per-agent message limits from binding
                raw_limits = getattr(binding, '_raw_data', {})
                if isinstance(raw_limits, dict):
                    ml = raw_limits.get("message_limits", {})
                else:
                    ml = {}
                for agent_id, limit in ml.items():
                    if isinstance(limit, (int, float)):
                        model.message_limits[agent_id] = int(limit)

        # Message targets from task
        if task.message_targets:
            for agent_id, targets in task.message_targets.items():
                model.message_targets[agent_id] = set(targets)

        return model

    def agent_can_observe_room(self, agent: str, room: str) -> bool:
        """Check if an agent can observe events in a room."""
        return room not in self.restricted_rooms.get(agent, set())

    def agent_can_observe_effect(self, agent: str, trigger_object: str) -> bool:
        """Check if an agent can observe the effect of a trigger."""
        hidden = self.hidden_effects.get(trigger_object, set())
        return agent not in hidden

    def get_unobservable_agents(self, room: str) -> Set[str]:
        """Get agents that cannot observe events in a room."""
        result = set()
        for agent, rooms in self.restricted_rooms.items():
            if room in rooms:
                result.add(agent)
        return result

    def has_information_asymmetry(self) -> bool:
        """Check if the task has any information asymmetry between agents."""
        return bool(self.restricted_rooms or self.hidden_effects)
