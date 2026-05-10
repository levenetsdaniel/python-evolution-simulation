"""
Presets of initial conditions and climate dynamics.

Using:
    from config.scenarios import warming
    config = warming()                                # default
    config = warming(SimConfig(seed=7, n_steps=800))  # with custom parameters

    config = SCENARIOS["harsh_seasons"]()
"""

from dataclasses import replace
from typing import Callable

from .sim_config import SimConfig, EnvironmentConfig


def _with_env(base: SimConfig | None, env: EnvironmentConfig) -> SimConfig:
    """Replace the environment in `base` (or in a fresh default `SimConfig`)."""
    return replace(base or SimConfig(), environment=env)


def stable(base: SimConfig | None = None) -> SimConfig:
    """Control scenario: static environment, no selective pressure from climate."""
    return _with_env(base, EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.0,
        hazard_level_start=0.1,
        hazard_step=0.0,
    ))


def warming(base: SimConfig | None = None) -> SimConfig:
    """
    Prolonged warming: cold start, monotonic heating.
    Stresses heat_resistance.
    """
    return _with_env(base, EnvironmentConfig(
        food_availability=10000.0,
        temp_start=-15.0,
        temp_step=0.1,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ))


def ice_age(base: SimConfig | None = None) -> SimConfig:
    """
    Cooling combined with a shrinking food supply. Warm start, monotonic decline.
    Stresses cold_resistance and metabolic economy.
    """
    return _with_env(base, EnvironmentConfig(
        food_availability=7000.0,
        food_step=-10,
        temp_start=30.0,
        temp_step=-0.08,
        hazard_level_start=0.1,
        hazard_step=0.0005,
    ))


def harsh_seasons(base: SimConfig | None = None) -> SimConfig:
    """
    Fast thermal cycles: ~150 steps per cycle (heating + reset).
    A hard test for simultaneous adaptation to heat and cold — a juicy target
    for the advisor.
    """
    return _with_env(base, EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.4,
        temp_reset=-10.0,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ))


def famine(base: SimConfig | None = None) -> SimConfig:
    """
    Famine: 4x less food than base.
    Stresses competition, metabolic rate, and size.
    """
    return _with_env(base, EnvironmentConfig(
        food_availability=2500.0,
        food_step=-5.0,
        temp_start=20.0,
        temp_step=0.05,
        hazard_level_start=0.1,
        hazard_step=0.001,
    ))


def hazardous(base: SimConfig | None = None) -> SimConfig:
    """Опасная среда: hazard растёт в 5 раз быстрее, старт выше. Тест на resilience."""
    return _with_env(base, EnvironmentConfig(
        food_availability=10000.0,
        temp_start=20.0,
        temp_step=0.05,
        hazard_level_start=0.2,
        hazard_step=0.005,
    ))


def chaos(base: SimConfig | None = None) -> SimConfig:
    """Hostile environment: hazard grows 5x faster from a higher start. Stresses resilience."""
    return _with_env(base, EnvironmentConfig(
        food_availability=4000.0,
        temp_start=10.0,
        temp_step=0.3,
        temp_reset=-15.0,
        hazard_level_start=0.2,
        hazard_step=0.003,
    ))


SCENARIOS: dict[str, Callable[[SimConfig | None], SimConfig]] = {
    "stable": stable,
    "warming": warming,
    "ice_age": ice_age,
    "harsh_seasons": harsh_seasons,
    "famine": famine,
    "hazardous": hazardous,
    "chaos": chaos,
}
