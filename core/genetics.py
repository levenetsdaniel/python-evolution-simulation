"""Genetic operators shared by the classic and spatial life cycles."""

import numpy as np


def crossover_and_mutate(p1, p2, model):
    """Uniform crossover followed by the model's configured mutation strategy."""
    mask = model.rng.random(len(p1.genome)) > 0.5
    before_mutation = np.where(mask, p1.genome, p2.genome)
    genome, deltas = model.mutation_strategy.mutate(before_mutation, p1, p2, model)
    return genome, before_mutation, deltas
