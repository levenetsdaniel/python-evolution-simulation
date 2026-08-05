from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

import numpy as np
from config.sim_config import PopulationConfig
from .enums import DeathCause, Gender
from .individual import Individual


class Population:
    """Manages the population of individuals and evolutionary dynamics."""

    def __init__(self, model: Model, config: PopulationConfig | None = None):
        """
        Initialize population manager.

        Args:
            model: Reference to the simulation model.
            config: Population configuration. Defaults are used if None.

        Initializes:
            - Initial population size
            - Generation counter
        """

        self.model = model
        self.config = config or PopulationConfig()
        self.initial_size = self.config.initial_size
        self.generation = 0

    def initialize(self):
        """
        Create the initial population.

        Generates individuals with random genomes.
        """

        for _ in range(self.initial_size):
            ind = Individual.random_init(self.model, self.config.genome_labels)
            self.model.agents.add(ind)

    @property
    def count_females(self) -> int:
        """Return number of female individuals in the population."""
        return len([a for a in self.model.agents if a.is_alive and a.gender == Gender.FEMALE])

    @property
    def count_males(self) -> int:
        """Return number of male individuals in the population."""
        return len([a for a in self.model.agents if a.is_alive and a.gender == Gender.MALE])

    @property
    def actual_pop_size(self) -> int:
        """Return number of alive individuals."""
        return len([a for a in self.model.agents if a.is_alive])

    @property
    def avg_fitness(self) -> np.floating:
        """Compute average fitness of alive individuals."""
        return self._mean_alive(lambda agent: agent.fitness, require_fitness=True)

    @property
    def avg_age(self) -> np.floating:
        """Compute average age of alive individuals."""
        return self._mean_alive(lambda agent: agent.age)

    @property
    def avg_satiation(self) -> np.floating:
        """Compute average normalized energy reserve of alive individuals."""
        return self._mean_alive(lambda agent: agent.satiation)

    @property
    def avg_energy(self) -> np.floating:
        """Compute average absolute energy reserve of alive individuals."""
        return self._mean_alive(lambda agent: agent.energy)

    @property
    def avg_heat_resistance(self) -> np.floating:
        """Compute average heat resistance trait."""
        return self._mean_alive(lambda agent: agent["heat_resistance"])

    @property
    def avg_cold_resistance(self) -> np.floating:
        """Compute average cold resistance trait."""
        return self._mean_alive(lambda agent: agent["cold_resistance"])

    @property
    def avg_metabolic_rate(self) -> np.floating:
        """Compute average metabolic rate trait."""
        return self._mean_alive(lambda agent: agent["metabolic_rate"])

    @property
    def avg_resilience(self) -> np.floating:
        """Compute average resilience trait."""
        return self._mean_alive(lambda agent: agent["resilience"])

    @property
    def avg_size(self) -> np.floating:
        """Compute average size trait."""
        return self._mean_alive(lambda agent: agent["size"])

    @property
    def avg_speed(self) -> np.floating:
        """Compute average speed trait."""
        return self._mean_alive(lambda agent: agent["speed"])

    @property
    def avg_aggressiveness(self) -> np.floating:
        """Compute average aggressiveness trait."""
        return self._mean_alive(lambda agent: agent["aggressiveness"])

    def _mean_alive(self, value, *, require_fitness: bool = False) -> np.floating:
        """Return a stable mean over alive agents, using zero after extinction."""
        agents = (
            agent
            for agent in self.model.agents
            if agent.is_alive and (not require_fitness or agent.fitness is not None)
        )
        values = [value(agent) for agent in agents]
        return np.mean(values) if values else np.float64(0.0)

    def remove_dead(self):
        """
        Remove dead individuals from the simulation.

        Also updates death statistics grouped by cause.
        """

        dead = list(self.model.agents.select(lambda a: not a.is_alive))
        self.model.deaths_this_step = {
            DeathCause.AGE: sum(1 for a in dead if a.death_cause == DeathCause.AGE),
            DeathCause.FITNESS: sum(1 for a in dead if a.death_cause == DeathCause.FITNESS),
            DeathCause.THRESHOLD: sum(1 for a in dead if a.death_cause == DeathCause.THRESHOLD),
            DeathCause.COMPETITION: sum(1 for a in dead if a.death_cause == DeathCause.COMPETITION)
        }
        for agent in dead:
            agent.remove()

    def remove_food(self):
        for a in self.model.agents:
            a.food_eaten = 0

    def _death_prob(self, loser: Individual, power_diff: float):
        """
        Determine an individual death probability after competition.

        Args:
            loser: Individual that lost the competition.
            power_diff: Relative difference in strength between agents.
        """

        death_prob = self.config.wound_base * abs(power_diff) * (1.0 - loser["resilience"]) * loser["aggressiveness"]
        if self.model.rng.random() < death_prob:
            loser.is_alive = False
            loser.death_cause = DeathCause.COMPETITION

    def compete(self, hungry: list[Individual], fed: list[Individual]):
        """
        Handle competition for food between agents.

        Hungry agents try to steal food from fed agents.

        Args:
            hungry: Agents that did not receive enough food.
            fed: Agents that successfully obtained food.
        """

        for attacker in hungry:
            while fed and attacker.food_eaten < attacker.food_need:
                victim = self.model.rng.choice(fed)
                self.model.attacks_this_step["considered"] += 1

                a_power = attacker["size"] + attacker["aggressiveness"]
                v_power = victim["size"] + victim["aggressiveness"]
                total = a_power + v_power

                if total == 0:
                    win_prob = 0.5
                    power_diff = 0.0
                else:
                    win_prob = a_power / total
                    power_diff = (a_power - v_power) / total

                if self.model.rng.random() >= self._attack_probability(attacker, victim, win_prob):
                    self.model.attacks_this_step["declined"] += 1
                    break

                energy_cost = min(attacker.energy, self.config.attack_energy_cost)
                attacker.spend_energy(energy_cost)
                self.model.attacks_this_step["started"] += 1
                self.model.attacks_this_step["energy_spent"] += energy_cost
                if self.model.rng.random() >= win_prob:
                    self.model.attacks_this_step["lost"] += 1
                    break

                steal_coef = np.clip(0.5 + 0.5 * power_diff, 0.1, 0.9)
                stolen = max(1, int(victim.food_eaten * steal_coef))
                remaining_need = attacker.food_need - attacker.food_eaten
                stolen = min(stolen, victim.food_eaten, remaining_need)

                if stolen <= 0:
                    fed.remove(victim)
                    break

                victim.food_eaten -= stolen
                attacker.food_eaten += stolen
                self.model.attacks_this_step["won"] += 1
                self.model.attacks_this_step["food_stolen"] += stolen

                self._death_prob(victim, power_diff)
                if not victim.is_alive or victim.food_eaten == 0:
                    fed.remove(victim)

    def record_birth(self, child_id: int, p1: Individual, p2: Individual, pre_mutation_genome: np.ndarray,
                     mutation_deltas: np.ndarray):
        """
        Record data about a newly created individual.

        Args:
            child_id: Unique ID of the child.
            p1: First parent.
            p2: Second parent.
            pre_mutation_genome: Genome before mutation.
            mutation_deltas: Changes applied during mutation.
        """

        recorder = getattr(self.model, "recorder", None)
        if recorder is None:
            return

        alive = [a for a in self.model.agents if a.is_alive and a.fitness is not None]

        recorder.record_birth(
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
            pop_mean_fitness=float(self.avg_fitness),
        )

    def reproduce(self):
        """
        Generate offspring using sexual reproduction.

        Process:
            - Select males and females
            - Choose parents based on fitness ranking
            - Perform crossover and mutation
            - Create new individuals
        """

        females = list(a for a in self.model.agents if
                       a.is_alive and a.fitness is not None and a.gender == Gender.FEMALE
                       and a.age >= self.config.min_reproduction_age
                       and a.energy >= self.config.reproduction_energy_cost)

        males = list(a for a in self.model.agents if
                     a.is_alive and a.fitness is not None and a.gender == Gender.MALE and a.age >= self.config.min_reproduction_age)

        if not females or not males:
            return

        male_weights = self._selection_weights(males)

        for p1 in females:
            birth_probability = np.clip(
                self.config.reproduction_rate * p1.satiation,
                0.0,
                1.0,
            )
            if self.model.rng.random() >= birth_probability:
                continue

            p2 = males[self.model.rng.choice(len(males), p=male_weights)]

            mask = self.model.rng.random(len(p1.genome)) > 0.5
            pre_mutation_genome = np.where(mask, p1.genome, p2.genome)

            genome, mutation_deltas = self.model.mutation_strategy.mutate(pre_mutation_genome, p1, p2, self.model)
            child = Individual.from_parents(
                model=self.model,
                p1=p1,
                p2=p2,
                genome=genome,
                generation=self.generation,
            )

            self.model.agents.add(child)
            self.model.births_this_step += 1
            p1.spend_energy(self.config.reproduction_energy_cost)

            self.record_birth(child.unique_id, p1, p2, pre_mutation_genome, mutation_deltas)

    def _attack_probability(self, attacker: Individual, victim: Individual, win_probability: float) -> float:
        """Estimate whether a hungry individual accepts a risky attack."""
        remaining_need = attacker.food_need - attacker.food_eaten
        expected_food = min(victim.food_eaten, remaining_need) / attacker.food_need
        hunger = 1.0 - attacker.satiation
        risk = (1.0 - win_probability) * (1.0 - attacker["resilience"])
        aggression_drive = self.config.attack_base_probability + (
            1.0 - self.config.attack_base_probability
        ) * attacker["aggressiveness"]

        probability = aggression_drive * (0.2 + 0.8 * hunger) * expected_food
        probability *= 1.0 - self.config.attack_risk_aversion * risk

        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def _selection_weights(candidates: list[Individual]) -> np.ndarray:
        """Return rank-based parent-selection weights with unbiased fitness ties."""
        fitness_values = np.asarray([candidate.fitness for candidate in candidates], dtype=float)
        _, rank_groups = np.unique(fitness_values, return_inverse=True)
        weights = rank_groups + 1
        return weights / weights.sum()

    def step(self):
        """Advance population state by one step."""

        self.model.agents.shuffle_do("step")
        self.remove_dead()
        self.generation += 1
        self.reproduce()
