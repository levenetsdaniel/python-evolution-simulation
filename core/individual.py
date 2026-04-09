import mesa
import numpy as np
import pandas as pd


class Individual(mesa.Agent):
    def __init__(self, model, genome: np.array, genome_labels: list[str], parent_ids, generation: int = 0):
        super().__init__(model)
        self.genome = np.array(genome, dtype=float)
        self.genome_labels = list(genome_labels)
        self.parent_ids = parent_ids
        self.generation = generation
        self.genes_map = pd.Series(genome, index=genome_labels)

        self.age = 0
        self.fitness = None
        self.is_alive = True

    def __getitem__(self, item):
        return self.genes_map[item]

    @property
    def n_genes(self):
        return len(self.genome)

    def compute_fitness(self, params):
        return float(np.mean(self.genome))  # заглушка, надо переделать

    def step(self):
        params = self.model.environment.current_params
        self.fitness = self.compute_fitness(params)
        self.age += 1

    @classmethod
    def random_init(cls, model, labels, ranges, generation=0):
        genome = [model.random.uniform(lo, hi) for lo, hi in ranges]
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
