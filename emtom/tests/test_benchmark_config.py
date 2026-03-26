from __future__ import annotations

from omegaconf import OmegaConf

from emtom.examples.run_habitat_benchmark import ensure_benchmark_observation_config


def test_ensure_benchmark_observation_config_forces_private_partial_obs() -> None:
    config = OmegaConf.create(
        {
            "world_model": {"partial_obs": False},
            "agent_asymmetry": False,
        }
    )

    ensure_benchmark_observation_config(config)

    assert config.world_model.partial_obs is True
    assert config.agent_asymmetry is True
    assert config.benchmark_observation_mode == "text"
    assert config.benchmark_run_mode == "standard"
    assert config.benchmark_vision.selector_prompt_name == "emtom_frame_selector"
