import numpy as np
from .enums import DeathCause, Gender
from .individual import Individual

GENOM_LABELES = ["heat_resistance", "cold_resistance", "metabolic_rate", "resilience", "size", "speed",
                 "aggressiveness"]
WOUND_BASE = 0.4


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
    def count_females(self) -> int:
        return len([a for a in self.model.agents if a.gender == Gender.FEMALE])

    @property
    def count_males(self) -> int:
        return len([a for a in self.model.agents if a.gender == Gender.MALE])

    @property
    def actual_size(self) -> int:
        return len([a for a in self.model.agents if a.is_alive])

    @property
    def avg_fitness(self) -> np.floating:
        return np.mean([a.fitness for a in self.model.agents if a.fitness is not None and a.is_alive] or [0])

    @property
    def avg_age(self) -> np.floating:
        return np.mean([a.age for a in self.model.agents if a.is_alive])

    @property
    def avg_satiation(self) -> np.floating:
        return np.mean([a.satiation for a in self.model.agents if a.is_alive])

    def remove_dead(self):
        dead = list(self.model.agents.select(lambda a: not a.is_alive))
        self.model.deaths_this_step = {
            DeathCause.AGE: sum(1 for a in dead if a.death_cause == DeathCause.AGE),
            DeathCause.FITNESS: sum(1 for a in dead if a.death_cause == DeathCause.FITNESS),
            DeathCause.THRESHOLD: sum(1 for a in dead if a.death_cause == DeathCause.THRESHOLD),
            DeathCause.COMPETITION: sum(1 for a in dead if a.death_cause == DeathCause.COMPETITION)
        }
        for agent in dead:
            agent.remove()

    def _death_prob(self, loser, pover_dif):
        death_prob = WOUND_BASE * abs(pover_dif) * (1.0 - loser.genes_map["resilience"]) * loser.genes_map["aggressiveness"]
        if self.model.rng.random() < death_prob:
            loser.is_alive = False
            loser.death_cause = DeathCause.COMPETITION

    def compete(self, hungry: list, fed: list):
        for attacker in hungry:
            if not fed:
                break

            victim = self.model.rng.choice(fed)

            a_power = attacker.genes_map["size"] + attacker.genes_map["aggressiveness"]
            v_power = victim.genes_map["size"] + victim.genes_map["aggressiveness"]
            total = a_power + v_power

            win_prob = a_power / total
            power_diff = (a_power - v_power) / total

            if self.model.rng.random() < win_prob:
                steal_coef = np.clip(0.5 + 0.5 * power_diff, 0.1, 0.9)
                stolen = max(1, int(victim.food_eaten * steal_coef))
                stolen = min(stolen, victim.food_eaten)

                victim.food_eaten -= stolen
                attacker.food_eaten += stolen

                self._death_prob(victim, power_diff)
                if not victim.is_alive:
                    fed.remove(victim)

    def record_birth(self, child_id: int, p1: Individual, p2: Individual, pre_mutation_genome: np.ndarray,
                     mutation_deltas: np.ndarray):
        alive = [a for a in self.model.agents if a.is_alive and a.fitness is not None]
        self.model.training_buffer.record_birth(
            child_id=child_id,
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

    def reproduce(self):
        females = list(a for a in self.model.agents if a.fitness is not None and a.gender == Gender.FEMALE and a.age >= 2)
        males = list(a for a in self.model.agents if a.fitness is not None and a.gender == Gender.MALE and a.age >= 2)
        n_offspring = max(2, int(len(females) * self.avg_satiation * 0.4))
        for _ in range(n_offspring):
            if males and females:
                p1_fitness = [a.fitness for a in females]

                ranks1 = np.argsort(np.argsort(p1_fitness)) + 1
                weights1 = ranks1 / ranks1.sum()

                p1 = females[self.model.rng.choice(len(females), p=weights1)]

                p2_fitness = [a.fitness for a in males]

                ranks2 = np.argsort(np.argsort(p2_fitness)) + 1
                weights2 = ranks2 / ranks2.sum()

                p2 = males[self.model.rng.choice(len(males), p=weights2)]
                females.remove(p1)

                mask = self.model.rng.random(len(p1.genome)) > 0.5
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

                self.record_birth(child.unique_id, p1, p2, pre_mutation_genome, mutation_deltas)

    def step(self):
        alive = [a for a in self.model.agents if a.is_alive]

        # temporary diagnostic
        print(f"food_pool before distribute: {self.model.environment.current_params['food_availability']}")
        self.model.environment.food_distribution()

        sample = [a for a in alive[:3]]
        for a in sample:
            print(f"  agent {a.unique_id}: food_eaten={a.food_eaten}, size={a.genes_map['size']:.2f}")

        self.model.agents.shuffle_do("step")
        self.remove_dead()
        self.generation += 1
        self.reproduce()
