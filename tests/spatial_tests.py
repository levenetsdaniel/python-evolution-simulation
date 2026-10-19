"""Physics, lifecycle, genetics, and reproducibility of the headless spatial mode."""

from dataclasses import replace

import numpy as np
import pytest
import mesa

from config.sim_config import IndividualConfig, SpatialConfig
from core.enums import DeathCause, Gender
from core.individual import Individual
from core.spatial import FoodSource, MoveAction, SpatialIndividual, SpatialWorld


def small_world(*, individual_config=None, **overrides):
    config = replace(SpatialConfig(initial_population=0, cluster_count=0,
                                   reproduction_rate=0), **overrides)
    costs = individual_config or IndividualConfig(basal_energy_cost=0, speed_energy_cost=0,
                                                  aggressiveness_energy_cost=0, age_scale=1e9)
    return SpatialWorld(config, seed=17, individual_config=costs)


def genes(world, **traits):
    values = dict.fromkeys(world.genome_labels, 0.5)
    values.update(traits)
    return np.array([values[label] for label in world.genome_labels])


def test_coordinates_movement_energy_and_walls():
    world = small_world(base_speed=2, speed_trait_scale=0, movement_energy_cost=2)
    agent = world.add_organism(genes(world), 99, 20, energy=20)
    assert isinstance(world, mesa.Model)
    assert isinstance(agent, SpatialIndividual)
    assert isinstance(agent, Individual)
    assert agent in world.agents
    assert agent.pos == (99, 20)
    with pytest.raises(ValueError, match="coordinates"):
        world.add_organism(genes(world), -1, 0)
    world.step({agent.unique_id: MoveAction(10, 0)})
    assert agent.state.x == 100
    assert agent.energy == pytest.approx(18)
    world.step({agent.unique_id: MoveAction(-10, 0)})
    assert agent.state.x == 98
    assert agent.pos == (98, 20)
    assert agent.energy == pytest.approx(14)
    assert world.steps == world.step_count == 2
    assert world.time == 2


def test_zero_energy_cannot_move_and_starvation_removes_agent():
    world = small_world(individual_config=IndividualConfig(basal_energy_cost=1,
                                                           speed_energy_cost=0,
                                                           aggressiveness_energy_cost=0,
                                                           age_scale=1e9))
    agent = world.add_organism(genes(world), 5, 5, energy=0)
    row = world.step({agent.unique_id: MoveAction(1, 0)})
    assert (agent.state.x, agent.state.y) == (5, 5)
    assert not agent.is_alive
    assert agent.death_cause == DeathCause.ENERGY
    assert row["Deaths_total"] == 1
    assert row["PopulationSize"] == 0
    assert row["AvgLifespan"] == 1
    assert agent not in world.agents
    assert agent.pos is None


def test_spatial_mode_uses_baseline_metabolic_cost_and_age_mortality():
    costs = IndividualConfig(basal_energy_cost=1, speed_energy_cost=2,
                             aggressiveness_energy_cost=3, age_scale=1e9)
    world = small_world(individual_config=costs)
    agent = world.add_organism(genes(world), 5, 5, energy=20)
    world.step({agent.unique_id: MoveAction(0, 0)})
    assert agent.energy == pytest.approx(20 - agent.metabolic_cost())
    assert agent.age == 1

    aging_world = small_world(individual_config=replace(costs, age_scale=0.01))
    elder = aging_world.add_organism(genes(aging_world), 5, 5, energy=100)
    row = aging_world.step({elder.unique_id: MoveAction(0, 0)})
    assert elder.death_cause == DeathCause.AGE
    assert row["Deaths_age"] == 1
    assert row["Deaths_energy"] == 0


def test_food_competition_and_regeneration():
    world = small_world(food_bite=5, food_energy=2)
    world.food_sources.append(FoodSource(10, 10, 3, 6, 1, 2))
    first = world.add_organism(genes(world), 10, 10, energy=10)
    second = world.add_organism(genes(world), 10, 10, energy=10)
    world.step({first.unique_id: MoveAction(0, 0), second.unique_id: MoveAction(0, 0)})
    assert first.energy == pytest.approx(13)
    assert second.energy == pytest.approx(13)
    assert world.food_sources[0].stock == pytest.approx(1)  # 3 consumed, then 1 restored
    world.food_sources[0].stock = 6
    world.update_resources()
    assert world.food_sources[0].stock == 6


def test_offspring_inherits_mutated_genome_near_parent_within_limit():
    world = small_world(reproduction_rate=1, min_reproduction_age=0,
                        reproduction_energy=50, offspring_energy=20,
                        mutation_std=0, max_population=3)
    p1 = world.add_organism(genes(world, speed=0), 0, 0, energy=150)
    p2 = world.add_organism(genes(world, speed=1), 1, 1, energy=100)
    p1.gender, p2.gender = Gender.FEMALE, Gender.MALE
    world.step({p1.unique_id: MoveAction(0, 0), p2.unique_id: MoveAction(0, 0)})
    children = [o for o in world.organisms.values() if o.parent_ids]
    assert len(children) == 1
    child = children[0]
    assert child.parent_ids == (p1.unique_id, p2.unique_id)
    assert child["speed"] in (0, 1)
    assert child.energy == 20
    assert 0 <= child.state.x <= world.width and 0 <= child.state.y <= world.height
    assert np.hypot(child.state.x - p1.state.x, child.state.y - p1.state.y) <= world.config.offspring_offset
    assert p1.energy == 130
    assert world.history[-1]["BirthCount"] == 1
    world.step({key: MoveAction(0, 0) for key in world.organisms})
    assert len(world.organisms) == 3


def test_mutation_uses_existing_operator_and_clips_traits():
    world = small_world(reproduction_rate=1, min_reproduction_age=0,
                        reproduction_energy=50, offspring_energy=20,
                        mutation_std=1, max_population=3)
    p1 = world.add_organism(genes(world), 10, 10)
    p2 = world.add_organism(genes(world), 11, 10)
    p1.energy = 150
    p1.gender, p2.gender = Gender.FEMALE, Gender.MALE
    world.step({p1.unique_id: MoveAction(0, 0), p2.unique_id: MoveAction(0, 0)})
    child = next(o for o in world.organisms.values() if o.parent_ids)
    assert np.any(child.genome != p1.genome)
    assert np.all((child.genome >= 0) & (child.genome <= 1))


def test_supplied_actions_override_policy_and_invalid_actions_fail():
    world = small_world(base_speed=1, speed_trait_scale=0)
    agent = world.add_organism(genes(world), 20, 20)
    with pytest.raises(ValueError, match="finite"):
        world.step({agent.unique_id: MoveAction(float("nan"), 0)})
    assert world.step_count == 0
    assert world.steps == 0
    world.step({agent.unique_id: MoveAction(0, 0)})
    assert (agent.state.x, agent.state.y) == (20, 20)


def test_seed_reproducibility_and_csv(tmp_path):
    config = SpatialConfig(initial_population=12, max_population=30, cluster_count=2,
                           sources_per_cluster=2)
    first = SpatialWorld(config, seed=12)
    second = SpatialWorld(config, seed=12)
    first.run(25)
    second.run(25)
    assert first.history == second.history
    assert [(o.state.x, o.state.y, o.energy) for o in first.organisms.values()] == [
        (o.state.x, o.state.y, o.energy) for o in second.organisms.values()]
    path = tmp_path / "history.csv"
    first.export_csv(path)
    assert len(path.read_text().splitlines()) == 27
    assert "AvgSpeed" in path.read_text().splitlines()[0]
    assert list(first.datacollector.get_model_vars_dataframe()["PopulationSize"]) == [
        row["PopulationSize"] for row in first.history]


def test_fixed_seed_smoke_250_steps():
    world = SpatialWorld(SpatialConfig(initial_population=20, max_population=45,
                                       cluster_count=2, sources_per_cluster=3), seed=42)
    world.run(250)
    assert len(world.history) == 251
    assert all(0 <= row["PopulationSize"] <= 45 for row in world.history)
    assert all(0 <= o.state.x <= world.width and 0 <= o.state.y <= world.height
               for o in world.organisms.values())
    assert all(0 <= source.stock <= source.capacity for source in world.food_sources)
