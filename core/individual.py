import mesa
import numpy as np
import random

from .enums import Gender, DeathCause
from .fitness import fitness

rng = np.random.default_rng()


class Individual(mesa.Agent):
    def __init__(self, model, genome: np.array, genome_labels: list[str], parent_ids, generation: int = 0):
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

    def __getitem__(self, item):
        return self.genes_map[item]

    @property
    def n_genes(self):
        return len(self.genome)

    def compute_fitness(self):
        return fitness(self.genes_map, self.model.environment.current_params)

    def step(self):
        params = self.model.environment.current_params
        self.fitness = self.compute_fitness()
        if self.fitness < 0.3:
            self.is_alive = False
            self.death_cause = DeathCause.THRESHOLD
            return

        self.age += 1
        age_death_prob = 1 - np.exp(-self.age / 100)
        fitness_death_prob = (1 - self.fitness) * 0.1
        death_prob = 1 - (1 - age_death_prob) * (1 - fitness_death_prob)

        if rng.random() < death_prob:
            self.is_alive = False
            self.death_cause = DeathCause.FITNESS if fitness_death_prob > age_death_prob else DeathCause.AGE
            return

    @classmethod
    def random_init(cls, model, labels, generation=0):
        genome = np.random.rand(5)
        return cls(
            model=model,
            genome=genome,
            genome_labels=labels,
            parent_ids=None,
            generation=generation
        )

    @classmethod
    def from_parents(cls, model, p1, p2, genome, generation=0):
        return cls(
            model=model,
            genome=genome,
            genome_labels=p1.genome_labels,
            generation=generation,
            parent_ids=[p1.unique_id, p2.unique_id],
        )
