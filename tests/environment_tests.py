"""Tests for core/environment.py."""

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

    def remove_food(self):
        for a in self.model.agents:
            if a.is_alive:
                a.food_eaten = 0

    def compete(self, hungry, fed):
        self.compete_calls.append((list(hungry), list(fed)))


class StubModel(mesa.Model):
    """Minimal mesa.Model exposing what Environment touches."""

    def __init__(self, seed=0):
        super().__init__(rng=seed)
        self.population = _RecordingPopulation()
        self.population.model = self


def make_ind(model, *, genome=None, food_eaten=0, alive=True):
    """Creates agent"""
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
        food_availability=5000.0,
        hazard_level_start=0.2,
    )
    env = Environment(StubModel(), config=cfg)

    assert env.time == 0
    assert env.current_params == {
        "temperature": 15.0,
        "food_availability": 5000.0,
        "hazard_level": 0.2,
    }
    assert env.prev_params == env.current_params
    assert env.prev_params is not env.current_params

def test_step_advances_params_and_snapshots_prev():
    cfg = EnvironmentConfig(
        temp_start=20.0, temp_step=0.05,
        hazard_level_start=0.1, hazard_step=0.01,
        food_availability=10000.0,
    )
    m = StubModel()
    env = Environment(m, config=cfg)
    make_ind(m)

    env.step()

    assert env.time == 1
    assert env.current_params["temperature"] == pytest.approx(20.05)
    assert env.current_params["hazard_level"] == pytest.approx(0.11)

    assert env.prev_params["temperature"] == pytest.approx(20.0)
    assert env.prev_params["hazard_level"] == pytest.approx(0.1)


def test_step_reflects_temperature_at_max_boundary_without_a_jump():
    cfg = EnvironmentConfig(
        temp_start=49.99, temp_step=0.05,
        max_temperature=50.0,
        food_availability=10000.0,
    )
    m = StubModel()
    env = Environment(m, config=cfg)
    make_ind(m)

    env.step()

    assert env.current_params["temperature"] == pytest.approx(49.96)

    env.step()

    assert env.current_params["temperature"] == pytest.approx(49.91)


def test_step_reflects_hazard_level_at_probability_bounds():
    cfg = EnvironmentConfig(
        food_availability=10000.0,
        hazard_level_start=0.98,
        hazard_step=0.05,
    )
    m = StubModel()
    env = Environment(m, config=cfg)
    make_ind(m)

    env.step()

    assert env.current_params["hazard_level"] == pytest.approx(0.97)

    env.step()

    assert env.current_params["hazard_level"] == pytest.approx(0.92)

def test_food_distribution_resets_food_eaten_at_start():
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=0.0))

    a1 = make_ind(m, food_eaten=50)
    a2 = make_ind(m, food_eaten=80)

    env.food_distribution()

    assert a1.food_eaten == 0
    assert a2.food_eaten == 0


def test_food_distribution_handles_empty_population():
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=100.0))

    env.food_distribution()

    assert env.current_food == 0.0


def test_food_distribution_depletes_current_food_but_preserves_capacity():
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=1000.0))

    a1 = make_ind(m)
    a2 = make_ind(m)

    env.food_distribution()

    assert a1.food_eaten == a1.food_need
    assert a2.food_eaten == a2.food_need
    assert env.current_food == pytest.approx(1000.0 - a1.food_need - a2.food_need)
    assert env.current_params["food_availability"] == 1000.0
    assert m.population.compete_calls == []


def test_food_distribution_feeds_agent_when_food_exactly_matches_need():
    m = StubModel()
    agent = make_ind(m)
    env = Environment(m, config=EnvironmentConfig(food_availability=agent.food_need))

    env.food_distribution()

    assert agent.food_eaten == agent.food_need
    assert env.current_food == 0.0
    assert m.population.compete_calls == []


def test_step_keeps_food_capacity_constant():
    m = StubModel()
    make_ind(m)
    env = Environment(m, config=EnvironmentConfig(food_availability=1000.0, food_regeneration_rate=0.25))

    env.step()
    env.step()

    assert env.food_capacity == 1000.0
    assert env.current_params["food_availability"] == 1000.0


@pytest.mark.parametrize(
    ("regeneration_rate", "expected_food"),
    [
        (0.0, 310.0),
        (0.25, 460.0),
        (1.0, 910.0),
    ],
)
def test_step_regenerates_a_share_of_missing_food_before_distribution(regeneration_rate, expected_food):
    m = StubModel()
    agent = make_ind(m)
    env = Environment(
        m,
        config=EnvironmentConfig(
            food_availability=1000.0,
            food_regeneration_rate=regeneration_rate,
        ),
    )
    env.current_food = 400.0

    env.step()

    assert agent.food_eaten == agent.food_need
    assert env.food_capacity == 1000.0
    assert env.current_food == pytest.approx(expected_food)


def test_food_distribution_resets_current_food_each_round():
    m = StubModel()
    env = Environment(m, config=EnvironmentConfig(food_availability=1000.0))
    make_ind(m)

    env.food_distribution()
    assert env.current_food < 1000.0

    env.food_distribution()
    assert env.current_food > 0
    assert env.current_food < 1000.0
    assert env.current_params["food_availability"] == 1000.0


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
