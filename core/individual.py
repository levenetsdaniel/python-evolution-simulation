import mesa
import numpy as np
import random

from .enums import Gender, DeathCause
from .fitness import fitness

rng = np.random.default_rng()


class Individual(mesa.Agent):
    def __init__(self, model: mesa.Model, genome: np.ndarray, genome_labels: list[str], parent_ids: tuple | None,
                 generation: int = 0):
        super().__init__(model)
        self.genome = np.array(genome, dtype=float)
        self.genome_labels = list(genome_labels)
        self.parent_ids = parent_ids
        self.generation = generation
        self.genes_map = dict(zip(genome_labels, genome))
        self.gender = random.choice(list(Gender))
        self.death_cause = None

        self.age = 0
        self.fitness = None
        self.is_alive = True

    def __getitem__(self, item: str):
        return self.genes_map[item]

    @property
    def n_genes(self):
        return len(self.genome)

    def compute_fitness(self):
        return fitness(self.genes_map, self.model.environment.current_params)

    def step(self):
        self.fitness = self.compute_fitness()
        self.model.training_buffer.record_fitness(self.unique_id, self.fitness)
        if self.fitness < 0.1:
            self.is_alive = False
            self.death_cause = DeathCause.THRESHOLD
            return

        self.age += 1
        age_death_prob = (1 - np.exp(-self.age / 80)) ** 1.2
        fitness_death_prob = (1 - self.fitness) * 0.1
        death_prob = 1 - (1 - age_death_prob) * (1 - fitness_death_prob)

        if rng.random() < death_prob:
            self.is_alive = False
            self.death_cause = DeathCause.FITNESS if fitness_death_prob > age_death_prob else DeathCause.AGE
            return

    @classmethod
    def random_init(cls, model: mesa.Model, labels: list[str], generation: int = 0):
        genome = np.random.rand(5)
        return cls(
            model=model,
            genome=genome,
            genome_labels=labels,
            parent_ids=None,
            generation=generation
        )

    @classmethod
    def from_parents(cls, model: mesa.Model, p1, p2, genome: np.ndarray, generation: int = 0):
        return cls(
            model=model,
            genome=genome,
            genome_labels=p1.genome_labels,
            generation=generation,
            parent_ids=(p1.unique_id, p2.unique_id),
        )
