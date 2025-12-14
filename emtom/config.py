"""
Configuration loading for EMTOM benchmark.

Loads YAML configs and instantiates mechanics, exploration settings, etc.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from emtom.mechanics.mechanic import Mechanic
from emtom.exploration import ExplorationConfig

# Default config directory
CONFIG_DIR = Path(__file__).parent / "conf"


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def find_config(name: str, config_type: str) -> Path:
    """
    Find a config file by name.

    Args:
        name: Config name (without .yaml extension)
        config_type: "mechanics", "exploration", or "task_gen"

    Returns:
        Path to the config file

    Raises:
        FileNotFoundError: If config not found
    """
    # Check if it's already a full path
    if os.path.isfile(name):
        return Path(name)

    # Add .yaml if needed
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    # Look in config directory
    config_path = CONFIG_DIR / config_type / name
    if config_path.exists():
        return config_path

    raise FileNotFoundError(
        f"Config '{name}' not found in {CONFIG_DIR / config_type}"
    )


def load_mechanics_config(
    config: Union[str, Path, Dict[str, Any]] = "default"
) -> List[Mechanic]:
    """
    Load mechanics from a config file or dict.

    Args:
        config: Config name, path, or dict

    Returns:
        List of instantiated Mechanic objects
    """
    from emtom.mechanics.registry import MechanicRegistry

    if isinstance(config, dict):
        config_data = config
    else:
        config_path = find_config(str(config), "mechanics")
        config_data = load_yaml(config_path)

    return MechanicRegistry.instantiate_from_config(config_data)


def load_exploration_config(
    config: Union[str, Path, Dict[str, Any]] = "default"
) -> Dict[str, Any]:
    """
    Load exploration config from a file or dict.

    Args:
        config: Config name, path, or dict

    Returns:
        Exploration configuration dict
    """
    if isinstance(config, dict):
        return config

    config_path = find_config(str(config), "exploration")
    return load_yaml(config_path)


def create_exploration_config(
    config: Union[str, Path, Dict[str, Any]] = "default"
) -> ExplorationConfig:
    """
    Create an ExplorationConfig from a config file.

    Args:
        config: Config name, path, or dict

    Returns:
        ExplorationConfig instance
    """
    config_data = load_exploration_config(config)
    exploration = config_data.get("exploration", {})

    return ExplorationConfig(
        max_steps=exploration.get("max_steps", 100),
        agent_ids=exploration.get("agent_ids", ["agent_0"]),
        log_path=exploration.get("log_path", "data/trajectories/emtom"),
        snapshot_frequency=exploration.get("snapshot_frequency", 0),
        stop_on_terminal=exploration.get("stop_on_terminal", True),
    )


def load_task_gen_config(
    config: Union[str, Path, Dict[str, Any]] = "default"
) -> Dict[str, Any]:
    """
    Load task generation config from a file or dict.

    Args:
        config: Config name, path, or dict

    Returns:
        Task generation configuration dict
    """
    if isinstance(config, dict):
        return config

    config_path = find_config(str(config), "task_gen")
    return load_yaml(config_path)


def list_available_configs(config_type: str) -> List[str]:
    """
    List available config files of a given type.

    Args:
        config_type: "mechanics", "exploration", or "task_gen"

    Returns:
        List of config names (without .yaml extension)
    """
    config_dir = CONFIG_DIR / config_type
    if not config_dir.exists():
        return []

    return [
        f.stem for f in config_dir.glob("*.yaml")
    ]


class EMTOMConfig:
    """
    Combined configuration for EMTOM runs.

    Loads and holds mechanics, exploration, and task generation configs.
    """

    def __init__(
        self,
        mechanics_config: Union[str, Path, Dict] = "default",
        exploration_config: Union[str, Path, Dict] = "default",
        task_gen_config: Union[str, Path, Dict] = "default",
    ):
        self.mechanics_config_name = str(mechanics_config)
        self.exploration_config_name = str(exploration_config)
        self.task_gen_config_name = str(task_gen_config)

        self._mechanics_data = load_yaml(
            find_config(self.mechanics_config_name, "mechanics")
        ) if isinstance(mechanics_config, str) else mechanics_config

        self._exploration_data = load_exploration_config(exploration_config)
        self._task_gen_data = load_task_gen_config(task_gen_config)

    def get_mechanics(self) -> List[Mechanic]:
        """Get instantiated mechanics."""
        return load_mechanics_config(self._mechanics_data)

    def get_exploration_config(self) -> ExplorationConfig:
        """Get exploration configuration."""
        return create_exploration_config(self._exploration_data)

    def get_curiosity_config(self) -> Dict[str, Any]:
        """Get curiosity model configuration."""
        return self._exploration_data.get("curiosity", {})

    def get_surprise_config(self) -> Dict[str, Any]:
        """Get surprise detection configuration."""
        return self._exploration_data.get("surprise_detection", {})

    def get_task_gen_config(self) -> Dict[str, Any]:
        """Get task generation configuration."""
        return self._task_gen_data

    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dict."""
        return {
            "mechanics": self._mechanics_data,
            "exploration": self._exploration_data,
            "task_gen": self._task_gen_data,
        }
