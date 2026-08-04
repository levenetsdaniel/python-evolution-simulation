from typing import TYPE_CHECKING

from core.individual import Individual
from neural.advisor import CatBoostAdvisor
from neural.trainer import train_advisor

if TYPE_CHECKING:
    from .model import Model

from abc import ABC, abstractmethod
import numpy as np
from core.training_buffer import TrainingBuffer
from neural.mutation import neural_mutate


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        pass


class BaselineMutation(MutationStrategy):
    """Classic mutation, using gor baseline."""

    def __init__(self, std: float):
        self._std = std

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[np.ndarray, np.ndarray]:
        noise = model.rng.normal(0, self._std, size=len(pre_genome))
        genome = np.clip(pre_genome + noise, 0.0, 1.0)
        mutations_deltas = genome - pre_genome
        return genome, mutations_deltas


class NeuralMutation(MutationStrategy):
    """Neural mutation, using for neural guiding line."""

    def __init__(self, advisor: CatBoostAdvisor, shift_strength: float | None = None):
        self._advisor = advisor
        self._shift_strength = 0.7 if shift_strength is None else shift_strength

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        env_params = model.environment.current_params
        prev_env_params = model.environment.prev_params
        env_config = model.config.environment
        min_temp = env_config.min_temperature
        temp_range = env_config.max_temperature - min_temp
        max_food = env_config.food_availability

        env_vec = [
            (env_params["temperature"] - min_temp) / temp_range,
            env_params["food_availability"] / max_food,
            env_params["hazard_level"],
        ]
        env_delta_vec = [
            (env_params["temperature"] - prev_env_params.get("temperature", env_params["temperature"]))
            / temp_range,
            (env_params["food_availability"] - prev_env_params.get("food_availability",
                                                                   env_params["food_availability"]))
            / max_food,
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

    def retrain(self, training_buffer: TrainingBuffer):
        """
        Retrains advisor

        Args:
            training_buffer: training buffer from model.
        """

        train_advisor(training_buffer, self._advisor)
