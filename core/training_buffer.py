import numpy as np
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from config.sim_config import EnvironmentConfig


@dataclass
class TrainingSample:
    pre_mutation_genome: list[float]
    env_params: list[float]
    env_delta: list[float]
    population_mean_genome: list[float]
    population_mean_fitness: float
    parent_mean_fitness: float

    mutation_deltas: list[float]
    child_fitness: float | None
    fitness_improvement: float | None


class TrainingBuffer:
    MIN_TEMPERATURE = EnvironmentConfig().min_temperature
    MAX_TEMPERATURE = EnvironmentConfig().max_temperature
    MAX_FOOD = EnvironmentConfig().food_availability

    def __init__(self):
        self.samples: list[TrainingSample] = []
        self._pending: dict[int, TrainingSample] = {}

    def record_birth(
            self,
            child_id: int,
            pre_mutation_genome: np.ndarray,
            mutation_deltas: np.ndarray,
            p1_fitness: float,
            p2_fitness: float,
            env_params: dict,
            env_delta: dict,
            pop_mean_genome: np.ndarray,
            pop_mean_fitness: float,
    ):
        temp_norm = (env_params["temperature"] - self.MIN_TEMPERATURE) / (self.MAX_TEMPERATURE - self.MIN_TEMPERATURE)
        food_norm = env_params["food_availability"] / self.MAX_FOOD
        env_vec = [
            temp_norm,
            food_norm,
            env_params["hazard_level"]
        ]

        delta_temp_norm = env_delta.get("temperature", 0.0) / (self.MAX_TEMPERATURE - self.MIN_TEMPERATURE)
        env_delta_vec = [
            delta_temp_norm,
            env_delta.get("food_availability", 0.0) / self.MAX_FOOD,
            env_delta.get("hazard_level", 0.0)
        ]

        sample = TrainingSample(
            pre_mutation_genome=pre_mutation_genome.tolist(),
            env_params=env_vec,
            env_delta=env_delta_vec,
            population_mean_genome=pop_mean_genome.tolist(),
            population_mean_fitness=float(pop_mean_fitness),
            parent_mean_fitness=float((p1_fitness + p2_fitness) / 2),
            mutation_deltas=mutation_deltas.tolist(),
            child_fitness=None,
            fitness_improvement=None,
        )
        self._pending[child_id] = sample

    def record_fitness(self, child_id: int, child_fitness: float):
        if child_id not in self._pending:
            return
        sample = self._pending.pop(child_id)
        sample.child_fitness = child_fitness
        sample.fitness_improvement = child_fitness - sample.parent_mean_fitness
        self.samples.append(sample)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in self.samples], f)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        completed = [s for s in self.samples if s.fitness_improvement is not None]
        if not completed:
            raise ValueError("No completed samples yet — fitness not recorded")

        x = np.array([
            s.pre_mutation_genome +
            s.env_params +
            s.env_delta +
            s.population_mean_genome +
            [s.population_mean_fitness,
             s.parent_mean_fitness]
            for s in completed
        ], dtype=float)

        y = np.array([s.mutation_deltas for s in completed], dtype=float)

        raw_weights = np.array([s.fitness_improvement for s in completed], dtype=float)
        weights = np.clip(raw_weights, 0.01, None)
        weights = weights / weights.sum()

        return x, y, weights

    def __len__(self):
        return len(self.samples)
