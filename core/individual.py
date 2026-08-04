import mesa
import numpy as np

from config.sim_config import IndividualConfig
from .enums import Gender, DeathCause
from .fitness import fitness

DEFAULT_INDIVIDUAL_CONFIG = IndividualConfig()


class Individual(mesa.Agent):
    """
    Represents a single agent (organism) in the evolutionary simulation.

    Each individual is defined by a genome — a vector of continuous traits,
    which determine its survival and reproduction capabilities.
    """

    def __init__(self, model: mesa.Model, genome: np.ndarray, genome_labels: list[str], parent_ids: tuple | None,
                 generation: int = 0, age: int = 0):
        """
        Initialize an individual agent.

        Args:
            model: Reference to the simulation model.
            genome: Array of gene values (0–1 range).
            genome_labels: Names corresponding to genome entries.
            parent_ids: Tuple of parent IDs, or None if randomly initialized.
            generation: Generation index of the individual.
            age: Initial age.

        Initializes:
            - genes_map: Mapping from gene names to values
            - gender: Randomly assigned biological sex
            - food_need: Computed from genome traits
            - fitness: Initially None
            - is_alive: Alive state flag
        """

        super().__init__(model)
        model_config = getattr(self.model, "config", None)
        self.config = getattr(model_config, "individual", None) or DEFAULT_INDIVIDUAL_CONFIG
        self.genome = np.array(genome, dtype=float)
        self.genome_labels = genome_labels
        self.parent_ids = parent_ids
        self.generation = generation
        self.genes_map = dict(zip(self.genome_labels, self.genome))
        self.gender = self.model.rng.choice(list(Gender))
        self.death_cause = None

        self.age = age
        self.food_eaten = 0
        self.food_need = max(1, int(round((self.genes_map["size"] * self.config.food_need_size_coef +
                                           self.genes_map["resilience"] * self.config.food_need_resilience_coef +
                                           self.genes_map["speed"] * self.config.food_need_speed_coef +
                                           self.genes_map["aggressiveness"] * self.config.food_need_aggr_coef) * 10)))
        self.fitness = None
        self.is_alive = True

    def __getitem__(self, item: str) -> float:
        """
        Access gene value by name.

        Args:
            item: Gene label.

        Returns:
            Value of the corresponding gene.
        """

        return self.genes_map[item]

    @property
    def satiation(self) -> float:
        """
        Compute current satiation level.

        Returns:
        Ratio of consumed food to required food.
        """

        return self.food_eaten / self.food_need

    def compute_fitness(self) -> float:
        """
        Compute fitness score for the individual.

        Returns:
            Fitness score in range [score_floor, 1.0].
        """

        ind_params = self.genes_map.copy()
        ind_params["satiation"] = self.satiation
        model_config = getattr(self.model, "config", None)
        fitness_config = getattr(model_config, "fitness", None)
        environment_config = getattr(model_config, "environment", None)
        return fitness(
            ind_params,
            self.model.environment.current_params,
            fitness_config=fitness_config,
            environment_config=environment_config,
        )

    def step(self):
        """
        Advance the individual by one simulation step.

        Checks if the agent is dead.
        """

        self.fitness = self.compute_fitness()
        self.model.training_buffer.record_fitness(self.unique_id, self.fitness)
        if self.fitness < self.config.fitness_death_threshold:
            self.is_alive = False
            self.death_cause = DeathCause.THRESHOLD
            return

        self.age += 1
        age_death_prob = (1 - np.exp(-self.age / self.config.age_scale)) ** self.config.age_death_power
        fitness_death_prob = (1 - self.fitness) * self.config.fitness_death_prob_coef
        death_prob = 1 - (1 - age_death_prob) * (1 - fitness_death_prob)

        if self.model.rng.random() < death_prob:
            self.is_alive = False
            self.death_cause = DeathCause.FITNESS if fitness_death_prob > age_death_prob else DeathCause.AGE
            return

    @classmethod
    def random_init(cls, model: mesa.Model, genome_labels: list[str], generation: int = 0,
                    age: int = 2) -> "Individual":
        """
        Create a randomly initialized individual.

        Args:
            model: Simulation model.
            genome_labels: Names of genome traits.
            generation: Initial generation.
            age: Initial age.

        Returns:
            New randomly initialized Individual.
        """

        genome = model.rng.random(len(genome_labels))

        return cls(
            model=model,
            genome=genome,
            parent_ids=None,
            genome_labels=genome_labels,
            generation=generation,
            age=age
        )

    @classmethod
    def from_parents(cls, model: mesa.Model, p1: "Individual", p2: "Individual", genome: np.ndarray,
                     generation: int = 0) -> "Individual":
        """
        Create an individual from two parents.

        Args:
            model: Simulation model.
            p1: First parent.
            p2: Second parent.
            genome: Resulting genome after recombination and mutation.
            generation: Generation index.

        Returns:
            New Individual with inherited parent IDs.
        """

        return cls(
            model=model,
            genome=genome,
            genome_labels=p1.genome_labels,
            generation=generation,
            parent_ids=(p1.unique_id, p2.unique_id),
        )
