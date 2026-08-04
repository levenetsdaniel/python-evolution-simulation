"""Tests for Hydra config composition and runtime config shape."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from config.registry import register_configs


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _compose(config_name: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return OmegaConf.to_object(cfg)


def test_sim_config_contains_all_runtime_subconfigs():
    cfg = _compose("sim_config")

    assert cfg.environment.temp_start == 20.0
    assert cfg.population.initial_size == 100
    assert cfg.individual.age_scale == 80.0
    assert cfg.fitness.score_floor == 0.05


def test_sim_config_allows_environment_group_override():
    cfg = _compose("sim_config", ["environment=warming"])

    assert cfg.environment.temp_start == -15.0
    assert cfg.environment.temp_step == 0.1


def test_sim_config_allows_nested_individual_and_fitness_overrides():
    cfg = _compose(
        "sim_config",
        [
            "individual.age_scale=120.0",
            "fitness.score_floor=0.02",
            "population.initial_size=250",
        ],
    )

    assert cfg.individual.age_scale == 120.0
    assert cfg.fitness.score_floor == 0.02
    assert cfg.population.initial_size == 250


def test_profile_config_contains_full_simulation_config():
    cfg = _compose("profile_config")

    assert cfg.simulation_config.environment.temp_start == 20.0
    assert cfg.simulation_config.population.initial_size == 100
    assert cfg.simulation_config.individual.age_scale == 80.0
    assert cfg.simulation_config.fitness.score_floor == 0.05


def test_profile_config_allows_nested_simulation_overrides():
    cfg = _compose(
        "profile_config",
        [
            "environment@simulation_config.environment=chaos",
            "simulation_config.population.initial_size=300",
            "simulation_config.individual.age_scale=95.0",
            "simulation_config.fitness.score_floor=0.01",
        ],
    )

    assert cfg.simulation_config.environment.food_availability == 4000.0
    assert cfg.simulation_config.population.initial_size == 300
    assert cfg.simulation_config.individual.age_scale == 95.0
    assert cfg.simulation_config.fitness.score_floor == 0.01
