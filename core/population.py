import numpy as np
import random
from .enums import DeathCause
from .individual import Individual

GENOM_LABELES = ["heat_resistance", "cold_resistance", "metabolic_rate", "resilience", "size"]


class Population:
    def __init__(self, model, size=100):
        self.model = model
        self.size = size
        self.generation = 0

    def initialize(self, labels=GENOM_LABELES):
        for _ in range(self.size):
            ind = Individual.random_init(self.model, labels)
            self.model.agents.add(ind)

    def remove_dead(self):
        dead = list(self.model.agents.select(lambda a: not a.is_alive))
        self.model.deaths_this_step = {
            DeathCause.AGE: sum(1 for a in dead if a.death_cause == DeathCause.AGE),
            DeathCause.FITNESS: sum(1 for a in dead if a.death_cause == DeathCause.FITNESS),
            DeathCause.THRESHOLD: sum(1 for a in dead if a.death_cause == DeathCause.THRESHOLD),
        }
        for agent in dead:
            agent.remove()

    def reproduce(self):
        for _ in range(len(self.model.agents) // 8):
            parents1 = list(s for s in self.model.agents if s.fitness is not None)
            p1_fitness = [s.fitness for s in parents1]

            ranks = np.argsort(np.argsort(p1_fitness))
            weights = ranks / ranks.sum()

            p1 = random.choices(parents1, weights=weights)[0]

            parents2 = [s for s in parents1 if s is not p1]
            p2_fitness = [s.fitness for s in parents2]

            if parents2:
                ranks = np.argsort(np.argsort(p2_fitness))
                weights = ranks / ranks.sum()

                p2 = random.choices(parents2, weights=weights)[0]
                parents1.remove(p1)
                parents1.remove(p2)

                mask = np.random.rand(len(p1.genome)) > 0.5
                genome = np.where(mask, p1.genome, p2.genome)
                genome += np.random.normal(0, 0.12, size=len(genome))
                genome = np.clip(genome, 0.0, 1.0)
                child = Individual.from_parents(
                    model=self.model,
                    p1=p1,
                    p2=p2,
                    genome=genome,
                    generation=self.generation,
                )
                self.model.agents.add(child)
                self.model.births_this_step += 1

    def step(self):
        self.model.agents.shuffle_do("step")
        self.remove_dead()
        self.generation += 1
        self.reproduce()
