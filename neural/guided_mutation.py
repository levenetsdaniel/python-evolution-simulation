from __future__ import annotations

import numpy as np

from core.individual import Individual
from core.mutation_patterns import MutationStrategy
from neural.advisor import CatBoostAdvisor
from neural.mutation import neural_mutate


class GuidedMutation(MutationStrategy):
    """Mutation strategy driven by the CatBoost advisor."""

    def __init__(self, advisor: CatBoostAdvisor, shift_strength: float = 0.7):
        self._advisor = advisor
        self._shift_strength = shift_strength

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model) -> tuple[np.ndarray, np.ndarray]:
        env_params = model.environment.current_params
        prev_env_params = model.environment.prev_params
        env_config = model.environment.config
        min_temp = env_config.min_temperature
        temp_range = env_config.max_temperature - min_temp
        max_food = max(float(model.environment.food_capacity), 1.0)

        env_vec = [
            (env_params["temperature"] - min_temp) / temp_range,
            env_params["food_availability"] / max_food,
            env_params["hazard_level"],
        ]
        env_delta_vec = [
            (env_params["temperature"] - prev_env_params.get("temperature", env_params["temperature"])) / temp_range,
            (
                env_params["food_availability"]
                - prev_env_params.get("food_availability", env_params["food_availability"])
            ) / max_food,
            env_params["hazard_level"] - prev_env_params.get("hazard_level", env_params["hazard_level"]),
        ]

        alive = [a for a in model.agents if a.is_alive]
        pop_mean_genome = np.mean([a.genome for a in alive], axis=0)
        pop_mean_fitness = model.population.avg_fitness
        parent_mean_fitness = (p1.fitness + p2.fitness) / 2.0
        mutation_std = getattr(model.config, "mutation_std", None)
        if mutation_std is None:
            mutation_std = model.config.population.mutation_std

        return neural_mutate(
            pre_mutation_genome=pre_genome,
            advisor=self._advisor,
            env_params=env_vec,
            env_delta=env_delta_vec,
            pop_mean_genome=pop_mean_genome,
            pop_mean_fitness=pop_mean_fitness,
            parent_mean_fitness=parent_mean_fitness,
            shift_strength=self._shift_strength,
            rng=model.rng,
            mutation_std=mutation_std,
        )
