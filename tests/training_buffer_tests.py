"""Tests for core/training_buffer.py."""

import json

import mesa
import numpy as np
import pytest

from config.sim_config import EnvironmentConfig, PopulationConfig
from core.enums import DeathCause
from core.individual import Individual
from core.population import Population
from core.environment import Environment
from core.training_buffer import TrainingBuffer

LABELS = PopulationConfig().genome_labels
N = len(LABELS)
ECFG = EnvironmentConfig()
MAX_FOOD = ECFG.food_availability


def _record_simple(buf, child_id=1, *, env_params=None, env_delta=None,
                   pop_mean_genome=None, pre_genome=None, deltas=None,
                   p1_fitness=0.4, p2_fitness=0.6, pop_mean_fitness=0.5):
    """Helper: feed record_birth with sensible defaults."""
    buf.record_birth(
        child_id=child_id,
        pre_mutation_genome=pre_genome if pre_genome is not None else np.full(N, 0.5),
        mutation_deltas=deltas if deltas is not None else np.zeros(N),
        p1_fitness=p1_fitness,
        p2_fitness=p2_fitness,
        env_params=env_params if env_params is not None else {
            "temperature": 20.0, "food_availability": MAX_FOOD, "hazard_level": 0.1,
        },
        env_delta=env_delta if env_delta is not None else {
            "temperature": 0.0, "food_availability": 0.0, "hazard_level": 0.0,
        },
        pop_mean_genome=pop_mean_genome if pop_mean_genome is not None else np.full(N, 0.5),
        pop_mean_fitness=pop_mean_fitness,
    )


def test_record_birth_keeps_sample_in_pending_until_fitness_arrives():
    buf = TrainingBuffer()
    _record_simple(buf, child_id=42)

    assert 42 in buf._pending
    assert buf.samples == []


def test_record_fitness_finalizes_into_samples_with_improvement():
    buf = TrainingBuffer()
    _record_simple(buf, child_id=42, p1_fitness=0.4, p2_fitness=0.6)
    buf.record_fitness(42, child_fitness=0.8)

    assert 42 not in buf._pending
    assert len(buf.samples) == 1
    sample = buf.samples[0]
    assert sample.child_fitness == pytest.approx(0.8)
    assert sample.parent_mean_fitness == pytest.approx(0.5)
    assert sample.fitness_improvement == pytest.approx(0.3)


def test_record_fitness_for_unknown_id_is_silent():
    """The simulation may call record_fitness for randomly-init agents that
    never went through record_birth. It must not crash or pollute samples."""
    buf = TrainingBuffer()
    buf.record_fitness(999, child_fitness=0.5)

    assert buf.samples == []
    assert buf._pending == {}


def test_record_birth_normalizes_env_params():
    buf = TrainingBuffer()
    midpoint_temp = (ECFG.min_temperature + ECFG.max_temperature) / 2
    _record_simple(buf, child_id=1, env_params={
        "temperature": midpoint_temp,
        "food_availability": MAX_FOOD,
        "hazard_level": 0.3,
    })
    buf.record_fitness(1, 0.5)

    temp_norm, food_norm, hazard = buf.samples[0].env_params
    assert temp_norm == pytest.approx(0.5)
    assert food_norm == pytest.approx(1.0)
    assert hazard == pytest.approx(0.3)


def test_record_birth_normalizes_env_delta():
    buf = TrainingBuffer()
    _record_simple(buf, child_id=1, env_delta={
        "temperature": ECFG.max_temperature - ECFG.min_temperature,
        "food_availability": MAX_FOOD / 2,
        "hazard_level": 0.05,
    })
    buf.record_fitness(1, 0.5)

    d_temp, d_food, d_hazard = buf.samples[0].env_delta
    assert d_temp == pytest.approx(1.0)
    assert d_food == pytest.approx(0.5)
    assert d_hazard == pytest.approx(0.05)


def test_to_numpy_raises_on_empty_buffer():
    buf = TrainingBuffer()
    _record_simple(buf, child_id=1)

    with pytest.raises(ValueError):
        buf.to_numpy()


def test_to_numpy_returns_correctly_shaped_arrays_and_normalized_weights():
    buf = TrainingBuffer()
    _record_simple(buf, child_id=1, p1_fitness=0.4, p2_fitness=0.6)
    _record_simple(buf, child_id=2, p1_fitness=0.4, p2_fitness=0.6)
    buf.record_fitness(1, child_fitness=0.7)
    buf.record_fitness(2, child_fitness=0.6)

    x, y, w = buf.to_numpy()

    expected_x_cols = N + 3 + 3 + N + 2
    assert x.shape == (2, expected_x_cols)
    assert y.shape == (2, N)
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]


def test_save_writes_valid_json_with_all_samples(tmp_path):
    buf = TrainingBuffer()
    _record_simple(buf, child_id=1)
    buf.record_fitness(1, 0.6)

    out = tmp_path / "samples.json"
    buf.save(out)

    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["child_fitness"] == pytest.approx(0.6)
    assert len(data[0]["pre_mutation_genome"]) == N


def test_buffer_receives_per_step_capacity_after_environment_consumed_food():
    class _RecordingBuffer:
        def __init__(self): self.births = []

        def record_birth(self, **kwargs): self.births.append(kwargs)

        def record_fitness(self, *args, **kwargs): pass

    class IntegrationModel(mesa.Model):
        def __init__(self):
            super().__init__(rng=0)
            self.training_buffer = _RecordingBuffer()
            self.births_this_step = 0
            self.deaths_this_step = {dc: 0 for dc in DeathCause}

    m = IntegrationModel()
    capacity = 10000.0
    m.environment = Environment(m, config=EnvironmentConfig(food_availability=capacity))
    m.population = Population(m)

    parent = Individual(model=m, genome=np.full(N, 0.5),
                        genome_labels=LABELS, parent_ids=None)
    parent.fitness = 0.5

    m.environment.step()
    assert m.environment.current_food < capacity

    m.population.record_birth(
        child_id=1, p1=parent, p2=parent,
        pre_mutation_genome=np.full(N, 0.5),
        mutation_deltas=np.zeros(N),
    )

    recorded = m.training_buffer.births[0]
    assert recorded["env_params"]["food_availability"] == capacity
    assert recorded["env_delta"]["food_availability"] == 0.0
