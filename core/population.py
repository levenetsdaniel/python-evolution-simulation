import numpy as np
import random
from .enums import DeathCause, Gender
from .individual import Individual

GENOM_LABELES = ["heat_resistance", "cold_resistance", "metabolic_rate", "resilience", "size"]


class Population:
    def __init__(self, model, size: int = 100):
        self.model = model
        self.initial_size = size
        self.generation = 0

    def initialize(self, labels: list[str] = GENOM_LABELES):
        for _ in range(self.initial_size):
            ind = Individual.random_init(self.model, labels)
            self.model.agents.add(ind)

    @property
    def count_females(self):
        return len([a for a in self.model.agents if a.gender == Gender.FEMALE])

    @property
    def count_males(self):
        return len([a for a in self.model.agents if a.gender == Gender.MALE])

    @property
    def actual_size(self):
        return len([a for a in self.model.agents if a.is_alive])

    @property
    def avg_fitness(self):
        return np.mean([a.fitness for a in self.model.agents if a.fitness is not None and a.is_alive] or [0])

    @property
    def avg_age(self):
        return np.mean([a.age for a in self.model.agents if a.is_alive])

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
        females = list(a for a in self.model.agents if a.fitness is not None and a.gender == Gender.FEMALE)
        males = list(a for a in self.model.agents if a.fitness is not None and a.gender == Gender.MALE)
        n_offspring = len(females) // 4
        for _ in range(n_offspring):
            if males and females:
                p1_fitness = [a.fitness for a in females]

                ranks1 = np.argsort(np.argsort(p1_fitness)) + 1
                weights1 = ranks1 / ranks1.sum()

                p1 = random.choices(females, weights=weights1)[0]

                p2_fitness = [a.fitness for a in males]

                ranks2 = np.argsort(np.argsort(p2_fitness)) + 1
                weights2 = ranks2 / ranks2.sum()

                p2 = random.choices(males, weights=weights2)[0]
                females.remove(p1)

                mask = np.random.rand(len(p1.genome)) > 0.5
                genome = np.where(mask, p1.genome, p2.genome)
                pre_mutation_genome = genome.copy()
                genome += np.random.normal(0, 0.12, size=len(genome))
                genome = np.clip(genome, 0.0, 1.0)
                mutation_deltas = genome - pre_mutation_genome
                child = Individual.from_parents(
                    model=self.model,
                    p1=p1,
                    p2=p2,
                    genome=genome,
                    generation=self.generation,
                )

                self.model.agents.add(child)
                self.model.births_this_step += 1

                alive = [a for a in self.model.agents if a.is_alive and a.fitness is not None]
                self.model.training_buffer.record_birth(
                    child_id=child.unique_id,
                    pre_mutation_genome=pre_mutation_genome,
                    mutation_deltas=mutation_deltas,
                    p1_fitness=p1.fitness,
                    p2_fitness=p2.fitness,
                    env_params=self.model.environment.current_params,
                    env_delta={
                        k: self.model.environment.current_params[k] -
                           self.model.environment.prev_params.get(k, v)
                        for k, v in self.model.environment.current_params.items()
                    },
                    pop_mean_genome=np.mean([a.genome for a in alive], axis=0),
                    pop_mean_fitness=self.avg_fitness,
                )

    def step(self):
        self.model.agents.shuffle_do("step")
        self.remove_dead()
        self.generation += 1
        self.reproduce()
