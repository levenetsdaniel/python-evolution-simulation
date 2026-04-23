import mesa
import numpy as np

from config.sim_config import IndividualConfig
from .enums import Gender, DeathCause
from .fitness import fitness


class Individual(mesa.Agent):
    def __init__(self, model: mesa.Model, genome: np.ndarray, genome_labels: list[str], parent_ids: tuple | None,
                 generation: int = 0, age: int = 0):
        super().__init__(model)
        self.config = IndividualConfig()
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
        return self.genes_map[item]

    @property
    def n_genes(self) -> int:
        return len(self.genome)

    @property
    def satiation(self) -> float:
        return self.food_eaten / self.food_need

    def compute_fitness(self) -> float:
        ind_params = self.genes_map.copy()
        ind_params["satiation"] = self.satiation
        return fitness(ind_params, self.model.environment.current_params)

    def step(self):
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
    def random_init(cls, model: mesa.Model, genome_labels: list[str], generation: int = 0, age: int = 2):
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
                     generation: int = 0):
        return cls(
            model=model,
            genome=genome,
            genome_labels=p1.genome_labels,
            generation=generation,
            parent_ids=(p1.unique_id, p2.unique_id),
        )
