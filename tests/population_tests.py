"""Tests for core/population.py."""

from types import SimpleNamespace

import mesa
import numpy as np
import pytest

from config.sim_config import PopulationConfig
from core.enums import DeathCause, Gender
from core.individual import Individual
from core.population import Population
from core.mutation_patterns import BaselineMutation

LABELS = PopulationConfig().genome_labels
N = len(LABELS)

DEFAULT_ENV = {
    "temperature": 20.0,
    "optimum_temperature": 1.0,
    "food_availability": 10000.0,
    "hazard_level": 0.0,
}

class _RecordingBuffer:
    """Captures both record_birth and record_fitness calls."""

    def __init__(self):
        self.births: list[dict] = []
        self.fitness_records: list[tuple] = []

    def record_birth(self, **kwargs):
        self.births.append(kwargs)

    def record_fitness(self, child_id, fitness):
        self.fitness_records.append((child_id, fitness))


class StubModel(mesa.Model):
    """Minimal mesa.Model exposing what Population touches."""

    def __init__(self, env=None, seed=0):
        super().__init__(rng=seed)
        env = env if env is not None else DEFAULT_ENV
        self.environment = SimpleNamespace(
            current_params=dict(env),
            prev_params=dict(env),
        )
        self.training_buffer = _RecordingBuffer()
        self.births_this_step = 0
        self.deaths_this_step = {dc: 0 for dc in DeathCause}
        self.mutation_strategy = BaselineMutation(PopulationConfig().mutation_std)


class _StubRNG:
    """rng stub: random() returns fixed value, choice picks first / index 0."""

    def __init__(self, random_val=0.0):
        self.random_val = random_val

    def random(self):
        return self.random_val

    def choice(self, seq, *args, **kwargs):
        if isinstance(seq, (int, np.integer)):
            return 0
        return list(seq)[0]

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return loc
        return np.zeros(size)


def make_ind(model, *, gender=None, age=2, fitness=0.5, alive=True,
             death_cause=None, genome=None, food_eaten=0):
    """Build an Individual with controlled state (no rng-driven fields)."""
    ind = Individual(
        model=model,
        genome=genome if genome is not None else np.full(N, 0.5),
        genome_labels=LABELS,
        parent_ids=None,
        age=age,
    )
    if gender is not None:
        ind.gender = gender
    ind.fitness = fitness
    ind.is_alive = alive
    ind.death_cause = death_cause
    ind.food_eaten = food_eaten
    return ind

def test_init_and_initialize_creates_initial_size_agents():
    m = StubModel()
    pop = Population(m, config=PopulationConfig(initial_size=5))

    assert pop.generation == 0
    assert pop.initial_size == 5
    assert len(m.agents) == 0

    pop.initialize()
    assert len(m.agents) == 5
    assert all(isinstance(a, Individual) for a in m.agents)


def test_counts_track_genders_and_alive_status():
    m = StubModel()
    pop = Population(m)

    assert pop.count_females == 0
    assert pop.count_males == 0
    assert pop.actual_pop_size == 0

    make_ind(m, gender=Gender.FEMALE, alive=True)
    make_ind(m, gender=Gender.FEMALE, alive=True)
    make_ind(m, gender=Gender.MALE, alive=True)
    make_ind(m, gender=Gender.MALE, alive=False, death_cause=DeathCause.AGE)

    assert pop.count_females == 2
    assert pop.count_males == 1
    assert pop.actual_pop_size == 3


def test_avg_properties_compute_means_over_alive_agents_only():
    m = StubModel()
    pop = Population(m)

    genomes = [
        np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]),
        np.array([0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3]),
    ]
    ages = [10, 20, 30]
    fitnesses = [0.5, 0.7, 0.9]

    for g, a, f in zip(genomes, ages, fitnesses):
        make_ind(m, age=a, fitness=f, genome=g)

    make_ind(m, age=999, fitness=0.0, alive=False,
             death_cause=DeathCause.AGE,
             genome=np.full(N, 1.0))

    expected = np.mean(np.stack(genomes), axis=0)
    assert pop.avg_age == pytest.approx(np.mean(ages))
    assert pop.avg_fitness == pytest.approx(np.mean(fitnesses))
    assert pop.avg_heat_resistance == pytest.approx(expected[0])
    assert pop.avg_cold_resistance == pytest.approx(expected[1])
    assert pop.avg_metabolic_rate == pytest.approx(expected[2])
    assert pop.avg_resilience == pytest.approx(expected[3])
    assert pop.avg_size == pytest.approx(expected[4])
    assert pop.avg_speed == pytest.approx(expected[5])
    assert pop.avg_aggressiveness == pytest.approx(expected[6])

def test_remove_dead_drops_dead_agents_and_counts_by_cause():
    m = StubModel()
    pop = Population(m)

    a_alive1 = make_ind(m, alive=True)
    a_alive2 = make_ind(m, alive=True)
    a_age = make_ind(m, alive=False, death_cause=DeathCause.AGE)
    a_fit1 = make_ind(m, alive=False, death_cause=DeathCause.FITNESS)
    a_fit2 = make_ind(m, alive=False, death_cause=DeathCause.FITNESS)
    a_thr = make_ind(m, alive=False, death_cause=DeathCause.THRESHOLD)
    a_comp = make_ind(m, alive=False, death_cause=DeathCause.COMPETITION)

    pop.remove_dead()

    remaining = list(m.agents)
    assert a_alive1 in remaining and a_alive2 in remaining
    assert all(a not in remaining for a in (a_age, a_fit1, a_fit2, a_thr, a_comp))
    assert m.deaths_this_step == {
        DeathCause.AGE: 1,
        DeathCause.FITNESS: 2,
        DeathCause.THRESHOLD: 1,
        DeathCause.COMPETITION: 1,
    }

def test_compete_winning_attacker_steals_food_with_zero_aggression_victim():
    m = StubModel()
    pop = Population(m)

    attacker = make_ind(m, food_eaten=0,
                        genome=np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0]))
    victim = make_ind(m, food_eaten=100,
                      genome=np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.0]))
    m.rng = _StubRNG(random_val=0.0)

    pop.compete(hungry=[attacker], fed=[victim])

    assert attacker.food_eaten + victim.food_eaten == 100
    assert attacker.food_eaten > 0
    assert victim.food_eaten >= 0
    assert victim.is_alive is True


def test_compete_victim_can_die_with_competition_cause():
    m = StubModel()
    pop = Population(m)

    attacker = make_ind(m, food_eaten=0,
                        genome=np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0]))
    victim = make_ind(m, food_eaten=100,
                      genome=np.array([0.5, 0.5, 0.5, 0.0, 0.1, 0.5, 1.0]))
    m.rng = _StubRNG(random_val=0.0)

    fed = [victim]
    pop.compete(hungry=[attacker], fed=fed)

    assert victim.is_alive is False
    assert victim.death_cause == DeathCause.COMPETITION
    assert victim not in fed


def test_compete_with_empty_fed_list_is_a_noop():
    m = StubModel()
    pop = Population(m)

    attacker = make_ind(m, food_eaten=0)
    pop.compete(hungry=[attacker], fed=[])

    assert attacker.food_eaten == 0


def test_compete_handles_zero_power_agents_without_warning():
    import warnings

    m = StubModel()
    pop = Population(m)

    zero_genome = np.zeros(N)
    attacker = make_ind(m, food_eaten=0, genome=zero_genome)
    victim = make_ind(m, food_eaten=100, genome=zero_genome)
    m.rng = _StubRNG(random_val=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pop.compete(hungry=[attacker], fed=[victim])

    assert not np.isnan(attacker.food_eaten)
    assert not np.isnan(victim.food_eaten)

def test_reproduce_no_offspring_without_one_of_the_genders():
    m = StubModel()
    pop = Population(m)

    make_ind(m, gender=Gender.FEMALE, age=5, fitness=0.5, food_eaten=0)
    make_ind(m, gender=Gender.FEMALE, age=5, fitness=0.5, food_eaten=0)

    initial_count = len(m.agents)
    pop.reproduce()

    assert len(m.agents) == initial_count
    assert m.births_this_step == 0


def test_reproduce_no_offspring_when_all_below_min_reproduction_age():
    m = StubModel()
    pop = Population(m, config=PopulationConfig(min_reproduction_age=10))

    make_ind(m, gender=Gender.FEMALE, age=5, fitness=0.5, food_eaten=0)
    make_ind(m, gender=Gender.MALE, age=5, fitness=0.6, food_eaten=0)

    initial_count = len(m.agents)
    pop.reproduce()

    assert len(m.agents) == initial_count
    assert m.births_this_step == 0


def test_reproduce_creates_clipped_children_and_records_births():
    m = StubModel()
    pop = Population(m)

    f1 = make_ind(m, gender=Gender.FEMALE, age=5, fitness=0.5,
                  food_eaten=0, genome=np.zeros(N))
    f2 = make_ind(m, gender=Gender.FEMALE, age=5, fitness=0.7,
                  food_eaten=0, genome=np.zeros(N))
    male = make_ind(m, gender=Gender.MALE, age=5, fitness=0.6,
                    food_eaten=0, genome=np.ones(N))

    pop.reproduce()

    children = [a for a in m.agents if a not in (f1, f2, male)]
    assert len(children) > 0
    assert m.births_this_step == len(children)
    assert len(m.training_buffer.births) == len(children)

    for child in children:
        p1_id, p2_id = child.parent_ids
        assert p1_id in (f1.unique_id, f2.unique_id)
        assert p2_id == male.unique_id
        assert (child.genome >= 0.0).all() and (child.genome <= 1.0).all()
        assert child.generation == pop.generation

def test_step_orchestrates_lifecycle_and_increments_generation():
    m = StubModel(seed=42)
    pop = Population(m, config=PopulationConfig(initial_size=20))
    pop.initialize()

    for a in m.agents:
        a.food_eaten = a.food_need

    initial_gen = pop.generation
    pop.step()

    assert pop.generation == initial_gen + 1
    assert all(a.is_alive for a in m.agents)
    assert len(m.agents) > 0