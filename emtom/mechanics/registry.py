"""
Mechanic registry for EMTOM benchmark.

Provides a decorator-based registration system for mechanics,
enabling plug-and-play mechanics selection via configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from emtom.core.mechanic import Mechanic, MechanicCategory

# Global registry of mechanic classes
_MECHANIC_REGISTRY: Dict[str, Type[Mechanic]] = {}


def register_mechanic(name: Optional[str] = None):
    """
    Decorator to register a mechanic class.

    Usage:
        @register_mechanic("inverse_open")
        class InverseOpenMechanic(Mechanic):
            ...

        # Or use class name automatically:
        @register_mechanic()
        class InverseOpenMechanic(Mechanic):
            name = "inverse_open"  # Uses this if no decorator arg
    """

    def decorator(cls: Type[Mechanic]) -> Type[Mechanic]:
        mechanic_name = name or getattr(cls, "name", cls.__name__)
        if mechanic_name in _MECHANIC_REGISTRY:
            raise ValueError(
                f"Mechanic '{mechanic_name}' is already registered "
                f"(by {_MECHANIC_REGISTRY[mechanic_name].__name__})"
            )
        _MECHANIC_REGISTRY[mechanic_name] = cls
        return cls

    return decorator


class MechanicRegistry:
    """
    Central registry for all available mechanics.

    Provides methods to query, instantiate, and compose mechanics
    from configuration.
    """

    @staticmethod
    def get(name: str) -> Type[Mechanic]:
        """
        Get a mechanic class by name.

        Args:
            name: Registered name of the mechanic

        Returns:
            The mechanic class

        Raises:
            KeyError: If mechanic name is not registered
        """
        if name not in _MECHANIC_REGISTRY:
            available = ", ".join(sorted(_MECHANIC_REGISTRY.keys()))
            raise KeyError(
                f"Unknown mechanic: '{name}'. Available: {available}"
            )
        return _MECHANIC_REGISTRY[name]

    @staticmethod
    def list_all() -> List[str]:
        """List all registered mechanic names."""
        return sorted(_MECHANIC_REGISTRY.keys())

    @staticmethod
    def list_by_category(category: MechanicCategory) -> List[str]:
        """List mechanic names in a specific category."""
        return sorted(
            name
            for name, cls in _MECHANIC_REGISTRY.items()
            if getattr(cls, "category", None) == category
        )

    @staticmethod
    def list_categories() -> Dict[str, List[str]]:
        """List all mechanics grouped by category."""
        result: Dict[str, List[str]] = {}
        for name, cls in _MECHANIC_REGISTRY.items():
            cat = getattr(cls, "category", MechanicCategory.COMPOUND)
            cat_name = cat.value if isinstance(cat, MechanicCategory) else str(cat)
            if cat_name not in result:
                result[cat_name] = []
            result[cat_name].append(name)
        return {k: sorted(v) for k, v in result.items()}

    @staticmethod
    def is_registered(name: str) -> bool:
        """Check if a mechanic is registered."""
        return name in _MECHANIC_REGISTRY

    @staticmethod
    def instantiate(name: str, **params) -> Mechanic:
        """
        Create an instance of a mechanic.

        Args:
            name: Registered name of the mechanic
            **params: Parameters to pass to the mechanic constructor

        Returns:
            Instantiated mechanic
        """
        cls = MechanicRegistry.get(name)
        return cls(**params)

    @staticmethod
    def instantiate_from_config(config: Dict[str, Any]) -> List[Mechanic]:
        """
        Create mechanic instances from a YAML config dict.

        Config format:
            mechanics:
              - name: inverse_open
                params:
                  affected_objects: []
              - name: remote_switch
                params:
                  mappings: null

        Args:
            config: Configuration dictionary

        Returns:
            List of instantiated mechanics
        """
        mechanics = []
        mechanic_configs = config.get("mechanics", [])

        for mech_cfg in mechanic_configs:
            if isinstance(mech_cfg, str):
                # Simple string format: just the mechanic name
                name = mech_cfg
                params = {}
            elif isinstance(mech_cfg, dict):
                # Full format with name and params
                name = mech_cfg["name"]
                params = mech_cfg.get("params", {})
            else:
                raise ValueError(f"Invalid mechanic config format: {mech_cfg}")

            mechanic = MechanicRegistry.instantiate(name, **params)
            mechanics.append(mechanic)

        return mechanics

    @staticmethod
    def instantiate_by_category(
        category: MechanicCategory,
        count: Optional[int] = None,
        **default_params,
    ) -> List[Mechanic]:
        """
        Instantiate all (or some) mechanics in a category.

        Args:
            category: Category to instantiate from
            count: If specified, only instantiate this many (random selection)
            **default_params: Default parameters for all mechanics

        Returns:
            List of instantiated mechanics
        """
        import random

        names = MechanicRegistry.list_by_category(category)
        if count is not None and count < len(names):
            names = random.sample(names, count)

        return [
            MechanicRegistry.instantiate(name, **default_params)
            for name in names
        ]

    @staticmethod
    def get_info(name: str) -> Dict[str, Any]:
        """
        Get information about a registered mechanic.

        Returns:
            Dict with name, category, description, class
        """
        cls = MechanicRegistry.get(name)
        return {
            "name": name,
            "category": getattr(cls, "category", MechanicCategory.COMPOUND).value,
            "description": getattr(cls, "description", ""),
            "class": cls.__name__,
        }

    @staticmethod
    def describe_all() -> str:
        """Get a human-readable description of all registered mechanics."""
        lines = ["Registered Mechanics:", "=" * 40]

        by_category = MechanicRegistry.list_categories()
        for cat_name, names in sorted(by_category.items()):
            lines.append(f"\n{cat_name.upper()}:")
            for name in names:
                info = MechanicRegistry.get_info(name)
                desc = info["description"][:60] + "..." if len(info["description"]) > 60 else info["description"]
                lines.append(f"  - {name}: {desc}")

        return "\n".join(lines)

    @staticmethod
    def instantiate_from_bindings(bindings: List[Dict[str, Any]]) -> List[Mechanic]:
        """
        Create mechanic instances from task binding format.

        This is used for benchmark tasks where specific object bindings
        are defined in the task file.

        Binding format (from MechanicBinding):
            {
                "mechanic_type": "remote_control",
                "trigger_object": "fridge_58",
                "target_object": "chest_of_drawers_52",
                "target_state": "is_open",
                "count": 3  # for counting_state
            }

        Args:
            bindings: List of binding dicts

        Returns:
            List of instantiated and bound mechanics
        """
        # Group bindings by mechanic type
        bindings_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for binding in bindings:
            mech_type = binding.get("mechanic_type")
            if mech_type:
                if mech_type not in bindings_by_type:
                    bindings_by_type[mech_type] = []
                bindings_by_type[mech_type].append(binding)

        # Create mechanics and apply explicit bindings
        mechanics = []
        for mech_type, type_bindings in bindings_by_type.items():
            try:
                mechanic = MechanicRegistry.instantiate(mech_type)
                if hasattr(mechanic, 'bind_explicit'):
                    mechanic.bind_explicit(type_bindings)
                mechanics.append(mechanic)
            except KeyError:
                print(f"[MechanicRegistry] Warning: Unknown mechanic type '{mech_type}'")

        return mechanics


def clear_registry() -> None:
    """Clear all registered mechanics (useful for testing)."""
    _MECHANIC_REGISTRY.clear()


def get_registry() -> Dict[str, Type[Mechanic]]:
    """Get the raw registry dict (useful for testing)."""
    return _MECHANIC_REGISTRY
