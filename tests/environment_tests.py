"""Unit tests for core/environment.py."""

import mesa
import numpy as np
import pytest

from config.sim_config import EnvironmentConfig, PopulationConfig
from core.environment import Environment
from core.individual import Individual

LABELS = PopulationConfig().genome_labels
N = len(LABELS)

class _RecordingPopulation:
    """Stub population — records compete() calls without performing them."""

    def __init__(self):
        self.compete_calls: list[tuple] = []

    def compete(self, hungry, fed):
        self.compete_calls.append((list(hungry), list(fed)))


class StubModel(mesa.Model):
    """Minimal mesa.Model exposing what Environment touches."""

    def __init__(self, seed=0):
        super().__init__(rng=seed)
        self.population = _RecordingPopulation()


def make_ind(model, *, genome=None, food_eaten=0, alive=True):
    ind = Individual(
        model=model,
        genome=genome if genome is not None else np.full(N, 0.5),
        genome_labels=LABELS,
        parent_ids=None,
    )
    ind.food_eaten = food_eaten
    ind.is_alive = alive
    return ind

def test_init_sets_params_from_config_and_independent_prev_copy():
    cfg = EnvironmentConfig(
        temp_start=15.0,
        optimum_temp_start=0.7,
        food_availability=5000.0,
        hazard_level_start=0.2,
    )
    env = Environment(StubModel(), config=cfg)

    assert env.time == 0
    assert env.current_params == {
        "temperature": 15.0,
        "optimum_temperature": 0.7,
        "food_availability": 5000.0,
        "hazard_level": 0.2,
    }
    assert env.prev_params == env.current_params
    assert env.prev_params is not env.current_params  # independent copies

def test_step_advances_params_and_snapshots_prev():
    cfg = EnvironmentConfig(
        temp_start=20.0, temp_step=0.05,
        optimum_temp_start=0.5, optimum_temp_step=0.001,
        hazard_level_start=0.1, hazard_step=0.01,
        food_availability=10000.0,
    )
    m = StubModel()
    env = Environment(m, config=cfg)
    make_ind(m)

    env.step()

    assert env.time == 1
    assert env.current_params["temperature"] == pytest.approx(20.05)
    assert env.current_params["optimum_temperature"] == pytest.approx(0.501)
    assert env.current_params["hazard_level"] == pytest.approx(0.11)

    assert env.prev_params["temperature"] == pytest.approx(20.0)
    assert env.prev_params["optimum_temperature"] == pytest.approx(0.5)
    assert env.prev_params["hazard_level"] == pytest.approx(0.1)


def test_step_resets_temperature_and_optimum_when_max_reached():
    cfg = EnvironmentConfig(
        temp_start=49.99, temp_step=0.05,
        max_temperature=50.0, temp_reset=-5.0,
        optimum_temp_reset=0.1,
        food_availability=10000.0,
    )
    m = StubModel()
    env = Environment(m, config=cfg)
    make_ind(m)

    env.step()

    assert env.current_params["temperature"] == cfg.temp_reset
    assert env.current_params["optimum_temperature"] == cfg.optimum_temp_reset

def test_food_distribution_resets_food_eaten_at_start():
    """All alive agents start the round with food_eaten=0, regardless of
    carryover from previous rounds (e.g. food stolen via compete())."""
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=0.0))

    a1 = make_ind(m, food_eaten=50)
    a2 = make_ind(m, food_eaten=80)

    env.food_distribution()

    assert a1.food_eaten == 0
    assert a2.food_eaten == 0


def test_food_distribution_feeds_agents_within_capacity():
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=1000.0))

    a1 = make_ind(m)
    a2 = make_ind(m)

    env.food_distribution()

    assert a1.food_eaten == a1.food_need
    assert a2.food_eaten == a2.food_need
    assert env.current_params["food_availability"] == pytest.approx(
        1000.0 - a1.food_need - a2.food_need
    )
    assert m.population.compete_calls == []


def test_food_distribution_invokes_compete_when_some_agents_remain_hungry():
    m = StubModel()
    a1 = make_ind(m)
    a2 = make_ind(m)
    env = Environment(m, config=EnvironmentConfig(food_availability=100.0))

    env.food_distribution()

    assert len(m.population.compete_calls) == 1
    hungry, fed = m.population.compete_calls[0]
    assert len(hungry) == 1
    assert len(fed) == 1