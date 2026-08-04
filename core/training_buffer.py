import numpy as np
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from config.sim_config import EnvironmentConfig


@dataclass
class TrainingSample:
    """
    Container for a single training sample.

    Used as the basic unit for training models.
    """

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
    """Collects and processes training data from evolutionary events."""

    MIN_TEMPERATURE = EnvironmentConfig().min_temperature
    MAX_TEMPERATURE = EnvironmentConfig().max_temperature
    MAX_FOOD = EnvironmentConfig().food_availability

    def __init__(self, environment_config: EnvironmentConfig | None = None):
        """
        Initialize empty training buffer.

        Initializes:
            - samples: completed training samples
            - _pending: temporary storage for incomplete samples
        """

        environment_config = environment_config or EnvironmentConfig()
        self.min_temperature = environment_config.min_temperature
        self.max_temperature = environment_config.max_temperature
        self.max_food = environment_config.food_availability

        self.samples: list[TrainingSample] = []
        self._pending: dict[int, TrainingSample] = {}

    def record_birth(
            self,
            child_id: int,
            pre_mutation_genome: np.ndarray,
            mutation_deltas: np.ndarray,
            p1_fitness: float,
            p2_fitness: float,
            env_params: dict[str, float],
            env_delta: dict[str, float],
            pop_mean_genome: np.ndarray,
            pop_mean_fitness: float,
    ):
        """
        Record a new birth event (pre-fitness stage).

        Stores all information except the final fitness.

        Args:
            child_id: Unique identifier of the new individual.
            pre_mutation_genome: Genome before mutation.
            mutation_deltas: Applied mutation vector.
            p1_fitness: Fitness of parent 1.
            p2_fitness: Fitness of parent 2.
            env_params: Current environment parameters.
            env_delta: Change in environment since last step.
            pop_mean_genome: Mean genome of population.
            pop_mean_fitness: Mean population fitness.
        """

        temp_norm = (
            (env_params["temperature"] - self.min_temperature)
            / (self.max_temperature - self.min_temperature)
        )
        food_norm = env_params["food_availability"] / self.max_food
        env_vec = [
            temp_norm,
            food_norm,
            env_params["hazard_level"]
        ]

        delta_temp_norm = env_delta.get("temperature", 0.0) / (self.max_temperature - self.min_temperature)
        env_delta_vec = [
            delta_temp_norm,
            env_delta.get("food_availability", 0.0) / self.max_food,
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
        """
        Finalize a training sample with fitness information.

        Args:
            child_id: Unique identifier of the individual.
            child_fitness: Computed fitness value.
        """

        if child_id not in self._pending:
            return
        sample = self._pending.pop(child_id)
        sample.child_fitness = child_fitness
        sample.fitness_improvement = child_fitness - sample.parent_mean_fitness
        self.samples.append(sample)

    def save(self, path: str | Path):
        """
        Save collected samples to a JSON file.

        Args:
            path: Output file path.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in self.samples], f)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert collected samples into NumPy arrays for training.

        Returns:
            x: Feature matrix combining:
                - genome
                - environment
                - environment delta
                - population stats
            y: Target matrix (mutation deltas)
            weights: Normalized sample weights based on fitness improvement

        Raises:
            ValueError: If no completed samples are available.
        """

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
