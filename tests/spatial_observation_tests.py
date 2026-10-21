"""Local sensory maps for future spatial policies, independent of rendering."""

from copy import deepcopy

import numpy as np
import pytest

from config.sim_config import IndividualConfig, SpatialConfig
from core.environment import FoodSource
from core.spatial import SpatialWorld
from core.spatial_policy import LOCAL_MAP_CHANNELS, MoveAction


def world_with_map(**overrides):
    settings = dict(initial_population=0, cluster_count=0, reproduction_rate=0,
                    local_map_size=8, local_map_span=16,
                    base_perception=6, perception_trait_scale=0)
    config = SpatialConfig(**(settings | overrides))
    costs = IndividualConfig(fitness_death_threshold=0, fitness_death_prob_coef=0,
                             basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0, age_scale=1e9)
    return SpatialWorld(config, seed=42, individual_config=costs)


def add_agent(world, x=50, y=50):
    return world.add_organism(np.full(7, 0.5), x, y, heading=0)


def test_local_map_channels_shape_range_and_observation():
    world = world_with_map()
    agent = add_agent(world)
    before = deepcopy(world.rng.bit_generator.state)
    observation = world.observation_provider.observe(world, agent)
    assert LOCAL_MAP_CHANNELS == ("food", "organisms", "boundary", "visibility")
    assert observation.local_map.shape == (4, 8, 8)
    assert observation.local_map.dtype == np.float32
    assert np.all((observation.local_map >= 0) & (observation.local_map <= 1))
    assert not observation.local_map[1].any()  # The observer is excluded.
    assert world.rng.bit_generator.state == before
    np.testing.assert_array_equal(agent.local_map, observation.local_map)


def test_local_map_places_visible_objects_and_aggregates_food():
    world = world_with_map()
    agent = add_agent(world)
    neighbor = add_agent(world, 52, 50)
    world.food_sources.extend([FoodSource(50, 52, 25, 100, 0, 1),
                               FoodSource(50.1, 52.1, 75, 100, 0, 1)])
    local_map = agent.local_map
    assert local_map[0, 5, 4] == pytest.approx(0.75)
    assert local_map[1, 4, 5] == 1
    assert np.count_nonzero(local_map[0]) == np.count_nonzero(local_map[1]) == 1
    neighbor.is_alive = False
    assert not agent.local_map[1].any()


def test_local_map_masks_objects_beyond_perception_even_in_visible_cells():
    world = world_with_map()
    agent = add_agent(world)
    # This cell center is at (5, 1), inside radius 6, but these points are outside.
    world.food_sources.append(FoodSource(55.9, 51.9, 100, 100, 0, 1))
    add_agent(world, 55.9, 51.9)
    local_map = agent.local_map
    assert local_map[3, 4, 6] == 1
    assert not local_map[:2].any()
    assert np.all(local_map[:, local_map[3] == 0] == 0)


def test_local_map_distinguishes_boundaries_empty_and_unknown_cells():
    world = world_with_map()
    agent = add_agent(world, 0, 0)
    local_map = agent.local_map
    assert local_map[2, 4, 3] == local_map[3, 4, 3] == 1  # Visible outside world.
    assert local_map[2, 4, 4] == 0 and local_map[3, 4, 4] == 1  # Visible empty world.
    assert np.all(local_map[:, 0, 0] == 0)  # Outside perception.


def test_inherited_perception_changes_visibility_without_changing_map_scale():
    world = world_with_map(perception_trait_scale=4)
    genes = np.full(7, 0.5)
    index = world.genome_labels.index("resilience")
    genes[index] = 0
    first = world.add_organism(genes, 50, 50)
    genes[index] = 1
    second = world.add_organism(genes, 50, 50)
    world.food_sources.extend([FoodSource(52, 50, 50, 100, 0, 1),
                               FoodSource(57, 50, 100, 100, 0, 1)])
    narrow, wide = first.local_map, second.local_map
    assert narrow.shape == wide.shape
    assert narrow[3].sum() < wide[3].sum()
    assert narrow[0, 4, 5] == wide[0, 4, 5] == 0.5
    assert narrow[0, 4, 7] == 0 and wide[0, 4, 7] == 1


def test_local_map_uses_relative_positions_and_fixed_world_axes():
    maps = []
    for offset in (0, 10):
        world = world_with_map()
        agent = add_agent(world, 50 + offset, 50 + offset)
        agent.state.heading = offset
        world.food_sources.append(FoodSource(52 + offset, 50 + offset, 50, 100, 0, 1))
        maps.append(agent.local_map)
    np.testing.assert_array_equal(*maps)


def test_local_map_is_a_detached_snapshot_and_updates_after_movement():
    world = world_with_map(base_speed=2, speed_trait_scale=0, movement_energy_cost=0)
    agent = add_agent(world)
    neighbor = add_agent(world, 54, 50)
    before = agent.local_map
    assert before[1, 4, 6] == 1
    with pytest.raises(ValueError):
        before[1, 4, 6] = 0
    world.step({agent.unique_id: MoveAction(2, 0), neighbor.unique_id: MoveAction(0, 0)})
    assert agent.local_map[1, 4, 5] == 1
    assert before[1, 4, 6] == 1 and before[1, 4, 5] == 0


@pytest.mark.parametrize("field,value", [("local_map_size", 0), ("local_map_size", 2.5),
                                          ("local_map_span", 0), ("local_map_span", float("nan"))])
def test_invalid_local_map_config_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        SpatialWorld(SpatialConfig(**{field: value}))
