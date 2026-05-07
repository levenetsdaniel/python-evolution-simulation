from typing import TYPE_CHECKING

from core.individual import Individual
from neural.advisor import CatBoostAdvisor

if TYPE_CHECKING:
    from .model import Model

from abc import ABC, abstractmethod
import numpy as np
from config.sim_config import EnvironmentConfig, SimConfig
from neural.mutation import neural_mutate

ENV_CNF = EnvironmentConfig()
SIM_CNF = SimConfig()


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        pass


class BaselineMutation(MutationStrategy):
    """Classic mutation, using gor baseline."""

    def __init__(self, std: float):
        self._std = std

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        noise = model.rng.normal(0, self._std, size=len(pre_genome))
        genome = np.clip(pre_genome + noise, 0.0, 1.0)
        mutations_deltas = genome - pre_genome
        return genome, mutations_deltas


class NeuralMutation(MutationStrategy):
    """Neural mutation, using for neural guiding line."""

    _min_t = ENV_CNF.min_temperature
    _max_t = ENV_CNF.max_temperature
    _max_f = ENV_CNF.food_availability

    def __init__(self, advisor: CatBoostAdvisor, shift_strength: float = SIM_CNF.shift_strength):
        self._advisor = advisor
        self._shift_strength = shift_strength

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        env_params = model.environment.current_params
        prev_env_params = model.environment.prev_params

        env_vec = [
            (env_params["temperature"] - self._min_t) / (self._max_t - self._min_t),
            env_params["food_availability"] / self._max_f,
            env_params["hazard_level"],
        ]
        env_delta_vec = [
            (env_params["temperature"] - prev_env_params.get("temperature", env_params["temperature"]))
            / (self._max_t - self._min_t),
            (env_params["food_availability"] - prev_env_params.get("food_availability",
                                                                   env_params["food_availability"]))
            / self._max_f,
            env_params["hazard_level"] - prev_env_params.get("hazard_level", env_params["hazard_level"]),
        ]

        alive = [a for a in model.agents if a.is_alive]
        pop_mean_genome = np.mean([a.genome for a in alive], axis=0)

        pop_mean_fitness = model.population.avg_fitness

        parent_mean_fitness = (p1.fitness + p2.fitness) / 2.0

        return neural_mutate(
            pre_mutation_genome=pre_genome,
            advisor=self._advisor,
            env_params=env_vec,
            env_delta=env_delta_vec,
            pop_mean_genome=pop_mean_genome,
            pop_mean_fitness=pop_mean_fitness,
            parent_mean_fitness=parent_mean_fitness,
            shift_strength=self._shift_strength,
        )
