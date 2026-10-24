"""Behavioral curricula built through the ordinary spatial world API."""

from dataclasses import replace
from math import ceil, cos, pi, sin, sqrt

import numpy as np

from config.sim_config import SimConfig, SpatialCurriculumConfig
from core.enums import Gender
from core.spatial import SpatialWorld


TRAINING_SCENARIOS = ("sparse_population", "separated_mates", "depleted_food", "baseline")


def build_training_world(
    config: SimConfig, scenario: str, seed: int, curriculum: SpatialCurriculumConfig
) -> SpatialWorld:
    """Place visible groups/partners/resources; execution uses unchanged world physics."""
    if scenario not in TRAINING_SCENARIOS:
        raise ValueError(f"Unknown behavioral scenario: {scenario}")
    if scenario == "baseline":
        return SpatialWorld.from_sim_config(replace(config, seed=seed))
    if curriculum.group_count * curriculum.group_size > config.spatial.max_population:
        raise ValueError("Curriculum population exceeds spatial.max_population")
    spatial = replace(
        config.spatial, initial_population=0, cluster_count=curriculum.group_count, sources_per_cluster=2
    )
    if scenario == "depleted_food":
        spatial = replace(
            spatial, food_regeneration=spatial.food_regeneration * 0.1, food_disappearance_probability=1.0
        )
    world = SpatialWorld.from_sim_config(replace(config, seed=seed, spatial=spatial))
    grid = ceil(sqrt(curriculum.group_count))
    centers = (
        [(spatial.width / 2, spatial.height / 2)]
        if grid == 1
        else [
            (x, y)
            for y in np.linspace(spatial.height * 0.2, spatial.height * 0.8, grid)
            for x in np.linspace(spatial.width * 0.2, spatial.width * 0.8, grid)
        ]
    )
    radius = min(6.0, spatial.width / (4 * grid), spatial.height / (4 * grid))
    for group, (cx, cy) in enumerate(centers[: curriculum.group_count]):
        for offset in range(2):
            source = world.food_sources[group * 2 + offset] if world.food_sources else None
            if source is not None:
                source.x = float(np.clip(cx + (16 if offset else 0), 0, spatial.width))
                source.y = float(cy)
                if scenario == "depleted_food" and offset == 0:
                    source.stock *= curriculum.depleted_stock_fraction
        for index in range(curriculum.group_size):
            female = index % 2 == 0
            if scenario == "separated_mates":
                distance = min(curriculum.partner_distance, spatial.width / grid, spatial.height / grid)
                x = cx + (-1 if female else 1) * distance / 2
                y = cy + (index // 2 - (curriculum.group_size // 2 - 1) / 2) * 0.5
            else:
                angle = index * 2 * pi / curriculum.group_size
                x, y = cx + radius * cos(angle), cy + radius * sin(angle)
            agent = world.add_organism(
                world.rng.uniform(0.25, 0.75, len(world.genome_labels)),
                float(np.clip(x, 0, spatial.width)),
                float(np.clip(y, 0, spatial.height)),
            )
            agent.gender = Gender.FEMALE if female else Gender.MALE
            agent.age = spatial.min_reproduction_age
    # Replace the constructor's empty census with the actual task population.
    world.history.clear()
    world.datacollector = type(world.datacollector)(model_reporters=world.datacollector.model_reporters)
    world.environment._sync_food_supply()
    world.environment.prev_params = world.environment.current_params.copy()
    world._record(0, dict.fromkeys(("age", "fitness", "threshold", "energy", "competition"), 0))
    return world
