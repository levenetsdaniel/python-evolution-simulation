"""Tests for core/individual.py."""

from types import SimpleNamespace

import mesa
import numpy as np
import pytest

from config.sim_config import IndividualConfig, PopulationConfig
from core.enums import DeathCause, Gender
from core.fitness import fitness as fitness_fn
from core.individual import Individual

LABELS = PopulationConfig().genome_labels
ICFG = IndividualConfig()
N = len(LABELS)

DEFAULT_ENV = {
    "temperature": 20.0,
    "optimum_temperature": 1.0,
    "food_availability": 10000.0,
    "hazard_level": 0.0,
}

class _Buffer:
    def __init__(self):
        self.records: list[tuple] = []

    def record_fitness(self, child_id, fitness):
        self.records.append((child_id, fitness))


class StubModel(mesa.Model):
    """Minimal mesa.Model with everything Individual touches."""

    def __init__(self, env=None, seed=0):
        super().__init__(rng=seed)
        self.environment = SimpleNamespace(current_params=dict(env or DEFAULT_ENV))
        self.training_buffer = _Buffer()


class _FixedRNG:
    """rng stub: random() always returns a fixed value (controls the death roll)."""

    def __init__(self, value): self.value = value
    def random(self): return self.value


NEVER_DIES = _FixedRNG(1.0)
ALWAYS_DIES = _FixedRNG(0.0)


def make_ind(model=None, genome=None, **kwargs):
    return Individual(
        model=model or StubModel(),
        genome=genome if genome is not None else np.full(N, 0.5),
        genome_labels=LABELS,
        parent_ids=kwargs.pop("parent_ids", None),
        **kwargs,
    )

def test_init_sets_all_fields():
    genome = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    m = StubModel()
    ind = make_ind(model=m, genome=genome, age=3, generation=5, parent_ids=(42, 99))

    assert ind.is_alive and ind.death_cause is None
    assert ind.fitness is None
    assert ind.food_eaten == 0
    assert ind.age == 3 and ind.generation == 5
    assert ind.parent_ids == (42, 99)
    assert ind.gender in list(Gender)
    assert ind in m.agents
    assert isinstance(ind.genome, np.ndarray) and np.issubdtype(ind.genome.dtype, np.floating)
    assert ind.genes_map == dict(zip(LABELS, genome))


def test_food_need_floor_and_formula():
    assert make_ind(genome=np.zeros(N)).food_need == 1

    genome = np.zeros(N)
    for i, lab in enumerate(LABELS):
        if lab in ("size", "resilience", "speed", "aggressiveness"):
            genome[i] = 0.5
    expected = int(round((
        0.5 * ICFG.food_need_size_coef
        + 0.5 * ICFG.food_need_resilience_coef
        + 0.5 * ICFG.food_need_speed_coef
        + 0.5 * ICFG.food_need_aggr_coef
    ) * 10))
    assert make_ind(genome=genome).food_need == expected

def test_getitem_matches_genes_map():
    ind = make_ind(genome=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    for label in LABELS:
        assert ind[label] == ind.genes_map[label]


def test_satiation_unfed_full_and_overfed():
    ind = make_ind()
    assert ind.satiation == 0.0
    ind.food_eaten = ind.food_need
    assert ind.satiation == pytest.approx(1.0)
    ind.food_eaten = ind.food_need * 2
    assert ind.satiation == pytest.approx(2.0)


def test_compute_fitness_matches_pure_function_and_does_not_mutate():
    m = StubModel()
    ind = make_ind(model=m)
    ind.food_eaten = ind.food_need
    snapshot = dict(ind.genes_map)

    expected = fitness_fn({**ind.genes_map, "satiation": 1.0}, m.environment.current_params)
    assert ind.compute_fitness() == pytest.approx(expected)
    assert ind.genes_map == snapshot

def test_step_records_fitness_and_increments_age_when_alive():
    m = StubModel()
    ind = make_ind(model=m, age=4)
    ind.food_eaten = ind.food_need
    m.rng = NEVER_DIES

    ind.step()

    assert ind.is_alive and ind.death_cause is None
    assert ind.age == 5
    assert ind.fitness is not None
    assert m.training_buffer.records == [(ind.unique_id, ind.fitness)]

def test_step_age_cause_when_old_and_healthy():
    m = StubModel()
    ind = make_ind(model=m, age=300)
    ind.food_eaten = ind.food_need
    m.rng = ALWAYS_DIES

    ind.step()

    assert ind.death_cause == DeathCause.AGE


def test_step_fitness_cause_when_young_and_unhealthy():
    m = StubModel()
    ind = make_ind(model=m, age=1)
    ind.food_eaten = ind.food_need
    m.rng = ALWAYS_DIES

    ind.step()

    assert ind.death_cause == DeathCause.FITNESS

def test_random_init_genome_in_unit_interval_and_no_parents():
    m = StubModel()
    ind = Individual.random_init(m, LABELS)

    assert len(ind.genome) == len(LABELS)
    assert (ind.genome >= 0.0).all() and (ind.genome <= 1.0).all()
    assert ind.age == 2
    assert ind.parent_ids is None

def test_from_parents_inherits_labels_ids_and_genome():
    m = StubModel()
    p1, p2 = make_ind(model=m), make_ind(model=m)
    genome = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    child = Individual.from_parents(m, p1, p2, genome, generation=10)

    assert child.parent_ids == (p1.unique_id, p2.unique_id)
    assert child.genome_labels == p1.genome_labels
    np.testing.assert_array_equal(child.genome, genome)
    assert child.generation == 10