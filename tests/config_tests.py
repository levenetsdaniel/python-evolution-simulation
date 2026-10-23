"""Tests for Hydra config composition and runtime config shape."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest

from config.registry import register_configs
from config.sim_config import (
    EnvironmentConfig,
    IndividualConfig,
    PopulationConfig,
    SimConfig,
    SpatialGuidanceConfig,
    validate_sim_config,
)


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _compose(config_name: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return OmegaConf.to_object(cfg)


def test_sim_config_contains_all_runtime_subconfigs():
    cfg = _compose("sim_config")

    assert cfg == SimConfig()
    assert cfg.environment.temp_start == 20.0
    assert cfg.population.initial_size == 100
    assert cfg.individual.age_scale == 160.0
    assert cfg.fitness.score_floor == 0.05
    assert not cfg.spatial_guidance.enabled


def test_sim_config_allows_spatial_guidance_overrides():
    cfg = _compose("sim_config", ["spatial_guidance.enabled=true",
                                  "spatial_guidance.advisor_path=artifacts/models/test.cbm",
                                  "spatial_guidance.shift_strength=0.4"])
    assert cfg.spatial_guidance.enabled
    assert cfg.spatial_guidance.advisor_path == "artifacts/models/test.cbm"
    assert cfg.spatial_guidance.shift_strength == 0.4


def test_sim_config_allows_environment_group_override():
    cfg = _compose("sim_config", ["environment=warming"])

    assert cfg.environment.temp_start == -15.0
    assert cfg.environment.temp_step == 0.1
    assert cfg.environment.food_regeneration_rate == 0.85


def test_sim_config_allows_cnn_and_local_map_overrides():
    cfg = _compose("sim_config", ["spatial_cnn.enabled=true", "spatial_cnn.epochs=3",
                                  "spatial.local_map_size=32", "spatial.local_map_span=100"])
    assert cfg.spatial_cnn.enabled and cfg.spatial_cnn.epochs == 3
    assert cfg.spatial.local_map_size == 32 and cfg.spatial.local_map_span == 100
    validate_sim_config(cfg)


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


def test_compare_config_contains_separate_simulation_and_ml_settings():
    cfg = _compose("compare_config")

    assert cfg.simulation.n_steps == 600
    assert cfg.simulation.population.initial_size == 100
    assert cfg.simulation.mode == "classic"
    assert cfg.advisor_path == "artifacts/models/classic_advisor.cbm"
    assert cfg.shift_strength == 0.7


@pytest.mark.parametrize("mode", ["spatial", "classic"])
def test_advisor_training_config_uses_a_separate_model_for_each_mode(mode):
    cfg = _compose("advisor_config", [f"simulation.mode={mode}"])
    assert cfg.simulation.mode == mode
    assert cfg.save_path == f"artifacts/models/{mode}_advisor.cbm"
    assert cfg.simulation.n_steps == 100


def test_compare_config_allows_nested_simulation_and_comparison_overrides():
    cfg = _compose(
        "compare_config",
        [
            "environment@simulation.environment=warming",
            "simulation.population.initial_size=220",
            "advisor_path=artifacts/models/test.cbm",
            "x_trait=size",
        ],
    )

    assert cfg.simulation.environment.temp_start == -15.0
    assert cfg.simulation.population.initial_size == 220
    assert cfg.advisor_path == "artifacts/models/test.cbm"
    assert cfg.x_trait == "size"


def test_resource_stress_scenarios_use_limited_regeneration():
    famine = _compose("sim_config", ["environment=famine"])
    ice_age = _compose("sim_config", ["environment=ice_age"])

    assert famine.environment.food_availability == 2500.0
    assert famine.environment.food_regeneration_rate == 0.15
    assert ice_age.environment.food_regeneration_rate == 0.45


@pytest.mark.parametrize(
    "config",
    [
        SimConfig(environment=EnvironmentConfig(min_temperature=10, max_temperature=10)),
        SimConfig(environment=EnvironmentConfig(temp_step=float("nan"))),
        SimConfig(environment=EnvironmentConfig(food_availability=-1)),
        SimConfig(environment=EnvironmentConfig(food_regeneration_rate=1.1)),
        SimConfig(population=PopulationConfig(genome_labels=["size"])),
        SimConfig(population=PopulationConfig(mutation_std=-0.1)),
        SimConfig(population=PopulationConfig(reproduction_rate=1.1)),
        SimConfig(population=PopulationConfig(attack_energy_cost=-1.0)),
        SimConfig(population=PopulationConfig(attack_risk_aversion=1.1)),
        SimConfig(individual=IndividualConfig(energy_capacity=0.0)),
        SimConfig(individual=IndividualConfig(energy_initial=101.0)),
        SimConfig(n_steps=-1),
        SimConfig(spatial_guidance=SpatialGuidanceConfig(enabled=True, advisor_path="")),
        SimConfig(spatial_guidance=SpatialGuidanceConfig(enabled=True, shift_strength=1.1)),
        SimConfig(mode="classic", spatial_guidance=SpatialGuidanceConfig(enabled=True)),
    ],
)
def test_validate_sim_config_rejects_invalid_values(config):
    with pytest.raises(ValueError, match="Invalid simulation config"):
        validate_sim_config(config)
