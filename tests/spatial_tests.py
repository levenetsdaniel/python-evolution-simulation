"""Physics, lifecycle, genetics, and reproducibility of the headless spatial mode."""

from dataclasses import replace

import numpy as np
import pytest
import mesa

from config.sim_config import EnvironmentConfig, IndividualConfig, PopulationConfig, SpatialConfig
from config.scenarios import SCENARIOS
from core.enums import DeathCause, Gender
from core.environment import Environment
from core.fitness import fitness
from core.individual import Individual
from core.spatial import FoodSource, MoveAction, SpatialIndividual, SpatialWorld


def small_world(*, individual_config=None, environment_config=None,
                population_config=None, **overrides):
    config = replace(SpatialConfig(initial_population=0, cluster_count=0,
                                   reproduction_rate=0,
                                   food_disappearance_probability=0,
                                   food_spawn_probability=0), **overrides)
    costs = individual_config or IndividualConfig(basal_energy_cost=0, speed_energy_cost=0,
                                                  aggressiveness_energy_cost=0, age_scale=1e9)
    costs = replace(costs, fitness_death_threshold=0, fitness_death_prob_coef=0)
    return SpatialWorld(config, seed=17, individual_config=costs,
                        environment_config=environment_config,
                        population_config=population_config)


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


def test_spatial_uses_classic_environment_and_fitness():
    environment = EnvironmentConfig(temp_start=49, temp_step=1,
                                    hazard_level_start=0.2, hazard_step=0.1)
    world = small_world(environment_config=environment)
    assert isinstance(world.environment, Environment)
    heat_adapted = world.add_organism(genes(world, heat_resistance=1, cold_resistance=0),
                                      10, 10, energy=100)
    cold_adapted = world.add_organism(genes(world, heat_resistance=0, cold_resistance=1),
                                      11, 10, energy=100)
    row = world.step({heat_adapted.unique_id: MoveAction(0, 0),
                      cold_adapted.unique_id: MoveAction(0, 0)})
    assert row["Temperature"] == 50
    assert row["HazardLevel"] == pytest.approx(0.3)
    assert heat_adapted.fitness > cold_adapted.fitness
    assert heat_adapted.fitness == pytest.approx(fitness(
        heat_adapted.genes_map, world.environment.current_params,
        fitness_config=world.fitness_config, environment_config=environment,
        satiation=heat_adapted.satiation))
    assert row["AvgFitness"] == pytest.approx((heat_adapted.fitness + cold_adapted.fitness) / 2)
    assert world.environment.current_params["food_availability"] == row["FoodAvailable"]


@pytest.mark.parametrize(("name", "capacity", "regeneration"), [
    ("default", 100, 10), ("stable", 100, 10), ("warming", 100, 8.5),
    ("ice_age", 70, 4.5), ("harsh_seasons", 100, 6.5),
    ("famine", 25, 1.5), ("hazardous", 100, 8.5), ("chaos", 40, 3),
])
def test_scenarios_control_spatial_sources_and_climate(name, capacity, regeneration):
    config = replace(SpatialConfig(), initial_population=0, cluster_count=2, sources_per_cluster=3)
    environment = replace(SCENARIOS[name])
    world = SpatialWorld(config, seed=17, environment_config=environment)
    assert len(world.food_sources) == 6
    assert world.environment.food_capacity == pytest.approx(6 * capacity)
    assert world.history[0]["FoodAvailable"] == pytest.approx(6 * capacity)
    assert world.environment.prev_params == world.environment.current_params
    assert world.environment.current_params["temperature"] == environment.temp_start
    for source in world.food_sources:
        assert source.capacity == pytest.approx(capacity)
        assert source.stock == pytest.approx(capacity)
        assert source.regeneration == pytest.approx(regeneration)
    source = world.food_sources[0]
    source.stock = capacity / 2
    world.update_resources()
    assert source.stock == pytest.approx(min(capacity, capacity / 2 + regeneration))
    world.step()
    assert world.history[-1]["Temperature"] == pytest.approx(environment.temp_start + environment.temp_step)
    assert world.history[-1]["HazardLevel"] == pytest.approx(environment.hazard_level_start + environment.hazard_step)
    # Effective resources do not mutate the user's base configuration or presets.
    assert config.food_capacity == 100 and config.food_regeneration == 10
    assert SCENARIOS[name] == environment


def test_scenario_scaling_combines_with_custom_spatial_food_parameters():
    environment = EnvironmentConfig(food_availability=20000, food_regeneration_rate=0.5)
    world = small_world(environment_config=environment, cluster_count=1, sources_per_cluster=1,
                        food_capacity=80, food_initial=20, food_regeneration=7)
    source = world.food_sources[0]
    assert (source.capacity, source.stock, source.regeneration) == (160, 40, 3.5)
    assert world.environment.food_capacity == 160


def test_relocated_sources_keep_scenario_food_parameters():
    world = small_world(environment_config=SCENARIOS["famine"], cluster_count=1, sources_per_cluster=1,
                        food_capacity=8, food_initial=4, food_regeneration=2,
                        food_disappearance_probability=1, food_spawn_probability=1)
    previous = world.food_sources[0]
    previous.stock = 0
    world.update_resources()
    source = world.food_sources[0]
    assert source is not previous
    assert (source.capacity, source.stock) == (2, 1)
    assert source.regeneration == pytest.approx(0.3)
    assert world.environment.disappeared_this_step == world.environment.spawned_this_step == 1
    assert world.environment.food_capacity == 2
    source.stock = 0.5
    world.update_resources()
    assert source.stock == pytest.approx(0.8)


def test_zero_scenario_food_prevents_source_generation_and_respawning():
    world = small_world(environment_config=EnvironmentConfig(food_availability=0),
                        cluster_count=1, sources_per_cluster=2, food_spawn_probability=1)
    assert world.food_sources == []
    assert world.environment.food_capacity == 0
    world.run(10)
    assert world.food_sources == []
    assert all(row["FoodAvailable"] == row["FoodSourceCount"] == 0 for row in world.history)


def test_zero_environment_regeneration_leaves_existing_stock_unchanged():
    world = small_world(environment_config=EnvironmentConfig(food_regeneration_rate=0),
                        cluster_count=1, sources_per_cluster=1)
    source = world.food_sources[0]
    source.stock = 3
    world.update_resources()
    assert source.stock == 3
    assert source.regeneration == 0


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_spatial_scenario_runs_300_steps_deterministically(name):
    config = replace(SpatialConfig(), initial_population=8, max_population=20,
                     cluster_count=2, sources_per_cluster=3)
    first = SpatialWorld(config, seed=31, environment_config=SCENARIOS[name])
    second = SpatialWorld(config, seed=31, environment_config=SCENARIOS[name])
    first.run(300)
    second.run(300)
    assert first.history == second.history
    assert first.step_count == 300
    assert all(0 <= row["PopulationSize"] <= 20 and
               0 <= row["FoodAvailable"] <= first.environment.food_capacity + 1e-9 for row in first.history)
    assert all(0 <= agent.state.x <= first.width and 0 <= agent.state.y <= first.height
               for agent in first.agents)


def test_environment_can_supply_conditions_by_position():
    world = small_world(environment_config=EnvironmentConfig(temp_step=0, hazard_step=0))
    world.environment.conditions_at = lambda x, y: {
        **world.environment.current_params, "temperature": 50 if x > 50 else -30}
    cold = world.add_organism(genes(world, heat_resistance=1, cold_resistance=0),
                              10, 10, energy=100)
    hot = world.add_organism(genes(world, heat_resistance=1, cold_resistance=0),
                             90, 10, energy=100)
    world.step({cold.unique_id: MoveAction(0, 0), hot.unique_id: MoveAction(0, 0)})
    assert hot.fitness > cold.fitness


def test_fitness_threshold_causes_spatial_death():
    world = SpatialWorld(SpatialConfig(initial_population=0, cluster_count=0,
                                       reproduction_rate=0), seed=17,
                         individual_config=IndividualConfig(fitness_death_threshold=0.99,
                                                            fitness_death_prob_coef=0,
                                                            basal_energy_cost=0,
                                                            speed_energy_cost=0,
                                                            aggressiveness_energy_cost=0))
    agent = world.add_organism(genes(world), 10, 10, energy=100)
    row = world.step({agent.unique_id: MoveAction(0, 0)})
    assert agent.death_cause == DeathCause.THRESHOLD
    assert row["Deaths_threshold"] == row["Deaths_total"] == 1


def test_food_competition_and_regeneration():
    world = small_world(food_bite=5, food_energy=2,
                        population_config=PopulationConfig(wound_base=0,
                                                           attack_energy_cost=0))
    world.food_sources.append(FoodSource(10, 10, 3, 6, 1, 2))
    first = world.add_organism(genes(world), 10, 10, energy=10)
    second = world.add_organism(genes(world), 10, 10, energy=10)
    row = world.step({first.unique_id: MoveAction(0, 0), second.unique_id: MoveAction(0, 0)})
    assert first.food_eaten + second.food_eaten == pytest.approx(3)
    assert first.energy + second.energy == pytest.approx(26)
    assert row["AttacksConsidered"] == 1
    assert world.food_sources[0].stock == pytest.approx(1)  # 3 consumed, then 1 restored
    world.food_sources[0].stock = 6
    world.update_resources()
    assert world.food_sources[0].stock == 6


def test_source_priority_rotates_between_steps():
    costs = IndividualConfig(food_need_size_coef=0, food_need_resilience_coef=0,
                             food_need_speed_coef=0, food_need_aggr_coef=0,
                             basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0, age_scale=1e9)
    world = small_world(individual_config=costs, food_bite=1, food_energy=1,
                        population_config=PopulationConfig(attack_base_probability=0))
    world.food_sources.append(FoodSource(10, 10, 1, 1, 1, 2))
    first = world.add_organism(genes(world, aggressiveness=0), 10, 10, energy=10)
    second = world.add_organism(genes(world, aggressiveness=0), 10, 10, energy=10)
    actions = {first.unique_id: MoveAction(0, 0), second.unique_id: MoveAction(0, 0)}
    world.step(actions)
    assert (first.food_eaten, second.food_eaten) == (1, 0)
    world.step(actions)
    assert (first.food_eaten, second.food_eaten) == (0, 1)


def test_competition_stays_at_one_source_and_transfers_food_energy():
    costs = IndividualConfig(food_need_size_coef=0, food_need_resilience_coef=0,
                             food_need_speed_coef=0, food_need_aggr_coef=0,
                             basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0, age_scale=1e9)
    combat = PopulationConfig(wound_base=0, attack_base_probability=1,
                              attack_risk_aversion=0, attack_energy_cost=0)
    world = small_world(individual_config=costs, population_config=combat,
                        food_bite=1, food_energy=5, food_regeneration=0)
    world.food_sources.extend([FoodSource(10, 10, 1, 1, 0, 2),
                               FoodSource(90, 90, 1, 1, 0, 2)])
    victim = world.add_organism(genes(world, size=0, aggressiveness=0), 10, 10, energy=10)
    attacker = world.add_organism(genes(world, size=1, aggressiveness=1), 10, 10, energy=0)
    elsewhere = world.add_organism(genes(world), 90, 90, energy=10)
    row = world.step({agent.unique_id: MoveAction(0, 0)
                      for agent in (victim, attacker, elsewhere)})
    assert row["AttacksConsidered"] == row["AttacksStarted"] == row["AttacksWon"] == 1
    assert row["FoodStolen"] == 1
    assert row["Deaths_competition"] == 0
    assert (victim.food_eaten, attacker.food_eaten, elsewhere.food_eaten) == (0, 1, 1)
    assert (victim.energy, attacker.energy, elsewhere.energy) == (10, 5, 15)
    assert row["FoodAvailable"] == 0


def test_competition_death_keeps_its_cause(monkeypatch):
    costs = IndividualConfig(food_need_size_coef=0, food_need_resilience_coef=0,
                             food_need_speed_coef=0, food_need_aggr_coef=0,
                             basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0)
    combat = PopulationConfig(attack_base_probability=1, attack_risk_aversion=0,
                              attack_energy_cost=0)
    world = small_world(individual_config=costs, population_config=combat,
                        food_bite=1, food_energy=5, food_regeneration=0)
    world.food_sources.append(FoodSource(10, 10, 1, 1, 0, 2))
    victim = world.add_organism(genes(world, size=0, aggressiveness=0), 10, 10, energy=10)
    attacker = world.add_organism(genes(world, size=1, aggressiveness=1), 10, 10, energy=0)

    def fatal_wound(agent, power_diff):
        agent.is_alive = False
        agent.death_cause = DeathCause.COMPETITION

    monkeypatch.setattr(world.population, "_death_prob", fatal_wound)
    row = world.step({victim.unique_id: MoveAction(0, 0),
                      attacker.unique_id: MoveAction(0, 0)})
    assert row["Deaths_competition"] == row["Deaths_total"] == 1
    assert victim.unique_id not in world.organisms


def test_exhausted_source_can_disappear_and_reappear_elsewhere():
    world = small_world(cluster_count=1, sources_per_cluster=1,
                        food_initial=1, food_capacity=1, food_bite=1,
                        food_regeneration=0,
                        food_disappearance_probability=1,
                        food_spawn_probability=1)
    world.food_sources[:] = [FoodSource(10, 10, 1, 1, 0, 2)]
    agent = world.add_organism(genes(world), 10, 10, energy=10)
    row = world.step({agent.unique_id: MoveAction(0, 0)})
    assert agent.food_eaten == 1
    assert row["FoodSourcesDisappeared"] == 1
    assert row["FoodSourcesSpawned"] == 1
    assert row["FoodSourceCount"] == 1
    source = world.food_sources[0]
    assert 0 <= source.x <= world.width and 0 <= source.y <= world.height
    assert (source.x, source.y) != (10, 10)
    assert source.stock == 1


def test_exhausted_source_can_survive_and_recover():
    world = small_world(cluster_count=1, sources_per_cluster=1,
                        food_disappearance_probability=0,
                        food_spawn_probability=1)
    world.food_sources[:] = [FoodSource(10, 10, 0, 5, 2, 2)]
    world.update_resources()
    assert world.food_sources[0].stock == 2
    assert world.environment.disappeared_this_step == 0
    assert world.environment.spawned_this_step == 0


def test_policy_moves_toward_relocated_food():
    world = small_world(cluster_count=1, sources_per_cluster=1,
                        food_initial=1, food_capacity=1, food_bite=1,
                        food_regeneration=0, food_radius=1,
                        food_disappearance_probability=1,
                        food_spawn_probability=1,
                        base_perception=200, perception_trait_scale=0,
                        base_speed=2, speed_trait_scale=0)
    world.food_sources[:] = [FoodSource(10, 10, 1, 1, 0, 1)]
    agent = world.add_organism(genes(world), 10, 10, energy=10)
    world.step({agent.unique_id: MoveAction(0, 0)})
    target = world.food_sources[0]
    before = np.hypot(agent.state.x - target.x, agent.state.y - target.y)
    world.step()
    after = np.hypot(agent.state.x - target.x, agent.state.y - target.y)
    assert after < before


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


def test_spatial_accepts_the_shared_mutation_strategy_interface():
    class FixedMutation:
        def mutate(self, pre_genome, p1, p2, model):
            genome = np.ones_like(pre_genome)
            return genome, genome - pre_genome

    world = SpatialWorld(SpatialConfig(initial_population=0, cluster_count=0,
                                       reproduction_rate=1, min_reproduction_age=0,
                                       reproduction_energy=50, offspring_energy=20),
                         seed=17, mutation_strategy=FixedMutation(),
                         individual_config=IndividualConfig(fitness_death_threshold=0,
                                                            fitness_death_prob_coef=0,
                                                            basal_energy_cost=0,
                                                            speed_energy_cost=0,
                                                            aggressiveness_energy_cost=0))
    mother = world.add_organism(genes(world), 10, 10, energy=150)
    father = world.add_organism(genes(world), 11, 10, energy=100)
    mother.gender, father.gender = Gender.FEMALE, Gender.MALE
    world.step({mother.unique_id: MoveAction(0, 0), father.unique_id: MoveAction(0, 0)})
    child = next(agent for agent in world.organisms.values() if agent.parent_ids)
    assert np.all(child.genome == 1)


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
    assert "AttacksStarted" in path.read_text().splitlines()[0]
    assert "Deaths_competition" in path.read_text().splitlines()[0]
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
