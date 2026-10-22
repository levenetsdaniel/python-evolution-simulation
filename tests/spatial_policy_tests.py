"""Energy priorities, local mate search and physical reproduction distance."""

from dataclasses import replace
from math import hypot

import numpy as np
import pytest

from config.sim_config import IndividualConfig, SpatialConfig
from core.enums import Gender
from core.environment import FoodSource
from core.spatial import SpatialWorld
from core.spatial_policy import HeuristicPolicy, MoveAction


def make_world(**settings):
    config = replace(SpatialConfig(initial_population=0, cluster_count=0), **settings)
    costs = IndividualConfig(basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0, fitness_death_threshold=0,
                             fitness_death_prob_coef=0, age_scale=1e9)
    return SpatialWorld(config, seed=42, individual_config=costs)


def add_agent(world, x, y=50, *, energy=150, gender=Gender.FEMALE, age=3):
    agent = world.add_organism(np.full(7, 0.5), x, y, energy=energy, heading=0)
    agent.gender, agent.age = gender, age
    return agent


def observe(world, agent):
    return world.observation_provider.observe(world, agent)


def test_hunger_takes_priority_over_an_available_partner():
    world = make_world()
    mother = add_agent(world, 50, energy=30)
    add_agent(world, 60, gender=Gender.MALE)
    world.food_sources.append(FoodSource(40, 50, 100, 100, 0, 2))
    observation = observe(world, mother)
    assert observation.nearest_mate is not None
    assert HeuristicPolicy.goal(observation) == "forage"
    action = world.policy.choose(observation, world.rng, 0.5)
    assert action.dx < 0 and action.dy == 0


def test_sated_agents_seek_partners_instead_of_food():
    world = make_world()
    mother = add_agent(world, 50)
    add_agent(world, 60, gender=Gender.MALE)
    world.food_sources.append(FoodSource(40, 50, 100, 100, 0, 2))
    observation = observe(world, mother)
    assert HeuristicPolicy.goal(observation) == "mate"
    assert world.policy.choose(observation, world.rng, 0.5).dx > 0


def test_agents_approach_each_other_and_reproduce_only_after_arrival():
    world = make_world()
    mother = add_agent(world, 50)
    father = add_agent(world, 60, gender=Gender.MALE)
    world.step()
    assert world.history[-1]["BirthCount"] == 0
    assert hypot(mother.state.x - father.state.x, mother.state.y - father.state.y) > world.mating_distance
    world.run(2)
    assert world.history[-1]["BirthCount"] == 1
    assert hypot(mother.state.x - father.state.x, mother.state.y - father.state.y) <= world.mating_distance


def test_agents_wait_at_food_center_or_inside_mating_interaction_radius():
    world = make_world()
    mother = add_agent(world, 50, energy=30)
    world.food_sources.append(FoodSource(50, 50, 100, 100, 0, 3))
    assert world.policy.choose(observe(world, mother), world.rng, 0.5) == MoveAction(0, 0)
    mother.energy = 150
    add_agent(world, 52, gender=Gender.MALE)
    assert world.policy.choose(observe(world, mother), world.rng, 0.5) == MoveAction(0, 0)


def test_ready_agents_without_partners_gather_at_food_sources():
    world = make_world()
    agent = add_agent(world, 50)
    world.food_sources.append(FoodSource(52, 50, 100, 100, 0, 10))
    observation = observe(world, agent)
    assert HeuristicPolicy.goal(observation) == "rendezvous"
    assert world.policy.choose(observation, world.rng, 0.5) == MoveAction(2, 0)


def test_foraging_hysteresis_is_lifetime_state_and_observations_are_read_only():
    world = make_world(reproduction_rate=0)
    agent = add_agent(world, 50, energy=80)
    world.step({agent.unique_id: MoveAction(0, 0)})
    assert agent.state.foraging
    agent.energy = 110
    assert observe(world, agent).foraging
    assert observe(world, agent).foraging
    world.step({agent.unique_id: MoveAction(0, 0)})
    assert agent.state.foraging
    agent.energy = 140
    assert not observe(world, agent).foraging
    assert agent.state.foraging  # Reading an observation does not commit memory.
    world.step({agent.unique_id: MoveAction(0, 0)})
    assert not agent.state.foraging
    agent.energy = 110
    assert not observe(world, agent).foraging


def test_only_visible_ready_opposite_sex_agents_are_mate_candidates():
    world = make_world(base_perception=6, perception_trait_scale=0)
    mother = add_agent(world, 50)
    add_agent(world, 51, gender=Gender.MALE, age=0)
    add_agent(world, 51, gender=Gender.FEMALE)
    add_agent(world, 51, gender=Gender.MALE, energy=0)
    add_agent(world, 57, gender=Gender.MALE)
    assert observe(world, mother).nearest_mate is None
    assert not mother.local_map[4].any()
    add_agent(world, 55, gender=Gender.MALE)
    observation = observe(world, mother)
    assert observation.nearest_mate.distance == 5
    assert observation.local_map[4].max() == 1


def test_ready_agents_explore_at_their_speed_when_no_partner_is_visible():
    world = make_world()
    agent = add_agent(world, 50)
    observation = observe(world, agent)
    assert HeuristicPolicy.goal(observation) == "explore"
    action = world.policy.choose(observation, world.rng, 0.5)
    assert hypot(action.dx, action.dy) == pytest.approx(world.speed(agent))


def test_sated_immature_agents_and_full_populations_rest():
    world = make_world(max_population=2)
    mother = add_agent(world, 50, age=0)
    assert HeuristicPolicy.goal(observe(world, mother)) == "rest"
    mother.age = 3
    add_agent(world, 55, gender=Gender.MALE)
    assert not observe(world, mother).ready_to_reproduce
    assert world.policy.choose(observe(world, mother), world.rng, 0.5) == MoveAction(0, 0)


@pytest.mark.parametrize("distance,births", [(2.999, 1), (3.0, 1), (3.001, 0), (5, 0)])
def test_reproduction_has_a_strict_two_cell_radius(distance, births):
    world = make_world()
    mother = add_agent(world, 50)
    father = add_agent(world, 50 + distance, gender=Gender.MALE)
    assert world.config.mating_radius == 2
    assert world.mating_distance == 3
    row = world.step({mother.unique_id: MoveAction(0, 0), father.unique_id: MoveAction(0, 0)})
    assert row["BirthCount"] == births


@pytest.mark.parametrize("size,distance", [(32, 6), (64, 3), (128, 1.5)])
def test_mating_radius_uses_the_configured_physical_cell_size(size, distance):
    assert make_world(local_map_size=size).mating_distance == distance


@pytest.mark.parametrize("low,high", [(0.9, 0.8), (0.6, 1.1), (0.6, 0.6)])
def test_invalid_foraging_thresholds_are_rejected(low, high):
    with pytest.raises(ValueError, match="threshold"):
        make_world(foraging_threshold=low, satiation_threshold=high)
