"""
Presets of initial conditions and climate dynamics.

Using:
    from config.scenarios import warming
    config = warming()                                # default
    config = warming(SimConfig(seed=7, n_steps=800))  # with custom parameters

    config = SCENARIOS["harsh_seasons"]()
"""

from .sim_config import EnvironmentConfig

SCENARIOS: dict[str, EnvironmentConfig] = {
    "default": EnvironmentConfig(),
    "stable": EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.0,
        hazard_level_start=0.1,
        hazard_step=0.0,
    ),
    "warming": EnvironmentConfig(
        food_availability=10000.0,
        temp_start=-15.0,
        temp_step=0.1,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ),
    "ice_age": EnvironmentConfig(
        food_availability=7000.0,
        food_step=-10.0,
        temp_start=30.0,
        temp_step=-0.08,
        hazard_level_start=0.1,
        hazard_step=0.0005,
    ),
    "harsh_seasons": EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.4,
        temp_reset=-10.0,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ),
    "famine": EnvironmentConfig(
        food_availability=2500.0,
        food_step=-5.0,
        temp_start=20.0,
        temp_step=0.05,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ),
    "hazardous": EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.05,
        hazard_level_start=0.2,
        hazard_step=0.005,
    ),
    "chaos": EnvironmentConfig(
        food_availability=4000.0,
        temp_start=10.0,
        temp_step=0.3,
        temp_reset=-15.0,
        hazard_level_start=0.2,
        hazard_step=0.003,
    ),
}
