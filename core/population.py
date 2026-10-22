from __future__ import annotations
from dataclasses import dataclass, replace
from math import cos, pi, sin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model
    from .spatial import SpatialWorld

import numpy as np
from config.sim_config import PopulationConfig, SpatialConfig
from .enums import DeathCause, Gender
from .fitness import batch_fitness
from .genetics import crossover_and_mutate
from .individual import Individual


@dataclass(frozen=True)
class PopulationMetrics:
    """Aggregate values calculated from one consistent population snapshot."""

    population_size: int
    female_count: int
    male_count: int
    avg_fitness: float
    avg_age: float
    avg_satiation: float
    avg_energy: float
    avg_heat_resistance: float
    avg_cold_resistance: float
    avg_metabolic_rate: float
    avg_resilience: float
    avg_size: float
    avg_speed: float
    avg_aggressiveness: float


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
            - Simulation time-step counter
        """

        self.model = model
        self.config = config or PopulationConfig()
        self.initial_size = self.config.initial_size
        self.time_step = 0
        self._metrics: PopulationMetrics | None = None
        self._batch_agents: list[Individual] | None = None
        self._batch_genomes: np.ndarray | None = None
        self._trait_indices = {label: index for index, label in enumerate(self.config.genome_labels)}

    def initialize(self):
        """
        Create the initial population.

        Generates individuals with random genomes.
        """

        for _ in range(self.initial_size):
            ind = Individual.random_init(self.model, self.config.genome_labels)
            self.model.agents.add(ind)
        self.invalidate_metrics()
        self.invalidate_batch_cache()

    def invalidate_metrics(self) -> None:
        """Discard the temporary aggregate snapshot."""
        self._metrics = None

    def invalidate_batch_cache(self) -> None:
        """Discard cached agent ordering and genome matrix after births or deaths."""
        self._batch_agents = None
        self._batch_genomes = None

    def _fitness_batch(self) -> tuple[list[Individual], np.ndarray]:
        """Return live agents and their immutable genomes, rebuilding after population changes."""
        if self._batch_agents is None or self._batch_genomes is None:
            self._batch_agents = [agent for agent in self.model.agents if agent.is_alive]
            self._batch_genomes = (
                np.stack([agent.genome for agent in self._batch_agents])
                if self._batch_agents
                else np.empty((0, len(self.config.genome_labels)), dtype=float)
            )
        return self._batch_agents, self._batch_genomes

    def metrics(self) -> PopulationMetrics:
        """Return the active aggregate snapshot, or calculate a fresh one."""
        return self._metrics or self._calculate_metrics()

    def cache_metrics(self) -> PopulationMetrics:
        """Calculate metrics once for a coordinated reporting operation."""
        self._metrics = self._calculate_metrics()
        return self._metrics

    def _calculate_metrics(self) -> PopulationMetrics:
        """Calculate aggregate values from the currently live agents."""
        alive_count = female_count = male_count = fitness_count = 0
        totals = {
            "fitness": 0.0,
            "age": 0.0,
            "satiation": 0.0,
            "energy": 0.0,
            "heat_resistance": 0.0,
            "cold_resistance": 0.0,
            "metabolic_rate": 0.0,
            "resilience": 0.0,
            "size": 0.0,
            "speed": 0.0,
            "aggressiveness": 0.0,
        }
        for agent in self.model.agents:
            if not agent.is_alive:
                continue
            alive_count += 1
            if agent.gender == Gender.FEMALE:
                female_count += 1
            else:
                male_count += 1
            totals["age"] += agent.age
            totals["satiation"] += agent.satiation
            totals["energy"] += agent.energy
            for trait in self.config.genome_labels:
                totals[trait] += agent[trait]
            if agent.fitness is not None:
                totals["fitness"] += agent.fitness
                fitness_count += 1

        mean = lambda name: totals[name] / alive_count if alive_count else 0.0
        return PopulationMetrics(
            population_size=alive_count,
            female_count=female_count,
            male_count=male_count,
            avg_fitness=totals["fitness"] / fitness_count if fitness_count else 0.0,
            avg_age=mean("age"),
            avg_satiation=mean("satiation"),
            avg_energy=mean("energy"),
            avg_heat_resistance=mean("heat_resistance"),
            avg_cold_resistance=mean("cold_resistance"),
            avg_metabolic_rate=mean("metabolic_rate"),
            avg_resilience=mean("resilience"),
            avg_size=mean("size"),
            avg_speed=mean("speed"),
            avg_aggressiveness=mean("aggressiveness"),
        )

    @property
    def count_females(self) -> int:
        """Return number of female individuals in the population."""
        return self.metrics().female_count

    @property
    def count_males(self) -> int:
        """Return number of male individuals in the population."""
        return self.metrics().male_count

    @property
    def actual_pop_size(self) -> int:
        """Return number of alive individuals."""
        return self.metrics().population_size

    @property
    def avg_fitness(self) -> np.floating:
        """Compute average fitness of alive individuals."""
        return self.metrics().avg_fitness

    @property
    def avg_age(self) -> np.floating:
        """Compute average age of alive individuals."""
        return self.metrics().avg_age

    @property
    def avg_satiation(self) -> np.floating:
        """Compute average normalized energy reserve of alive individuals."""
        return self.metrics().avg_satiation

    @property
    def avg_energy(self) -> np.floating:
        """Compute average absolute energy reserve of alive individuals."""
        return self.metrics().avg_energy

    @property
    def avg_heat_resistance(self) -> np.floating:
        """Compute average heat resistance trait."""
        return self.metrics().avg_heat_resistance

    @property
    def avg_cold_resistance(self) -> np.floating:
        """Compute average cold resistance trait."""
        return self.metrics().avg_cold_resistance

    @property
    def avg_metabolic_rate(self) -> np.floating:
        """Compute average metabolic rate trait."""
        return self.metrics().avg_metabolic_rate

    @property
    def avg_resilience(self) -> np.floating:
        """Compute average resilience trait."""
        return self.metrics().avg_resilience

    @property
    def avg_size(self) -> np.floating:
        """Compute average size trait."""
        return self.metrics().avg_size

    @property
    def avg_speed(self) -> np.floating:
        """Compute average speed trait."""
        return self.metrics().avg_speed

    @property
    def avg_aggressiveness(self) -> np.floating:
        """Compute average aggressiveness trait."""
        return self.metrics().avg_aggressiveness

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
        self.invalidate_metrics()
        if dead:
            self.invalidate_batch_cache()

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
            self.invalidate_batch_cache()

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

        self.invalidate_metrics()
        females = list(a for a in self.model.agents if
                       a.can_reproduce(self.config.min_reproduction_age, self.config.reproduction_energy_cost)
                       and a.fitness is not None and a.gender == Gender.FEMALE)

        males = list(a for a in self.model.agents if
                     a.can_reproduce(self.config.min_reproduction_age, 0)
                     and a.fitness is not None and a.gender == Gender.MALE)

        if not females or not males:
            return

        male_weights = self._selection_weights(males)

        for p1 in females:
            birth_probability = p1.reproduction_probability(self.config.reproduction_rate)
            if self.model.rng.random() >= birth_probability:
                continue

            p2 = males[self.model.rng.choice(len(males), p=male_weights)]

            genome, pre_mutation_genome, mutation_deltas = crossover_and_mutate(p1, p2, self.model)
            child = Individual.from_parents(
                model=self.model,
                p1=p1,
                p2=p2,
                genome=genome,
                birth_step=self.time_step,
            )

            self.model.agents.add(child)
            self.invalidate_batch_cache()
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

        self.invalidate_metrics()
        shuffled_agents = list(self.model.agents.shuffle())
        for agent in shuffled_agents:
            if agent.is_alive:
                agent._update_energy()

        batch_agents, genomes = self._fitness_batch()
        satiation = np.fromiter(
            (agent.satiation for agent in batch_agents),
            dtype=float,
            count=len(batch_agents),
        )
        scores = batch_fitness(
            genomes,
            satiation,
            self.model.environment.current_params,
            fitness_config=getattr(getattr(self.model, "config", None), "fitness", None),
            environment_config=getattr(getattr(self.model, "config", None), "environment", None),
            trait_indices=self._trait_indices,
        )
        score_by_agent = dict(zip(batch_agents, scores))
        for agent in shuffled_agents:
            if agent.is_alive:
                agent.fitness = float(score_by_agent[agent])
                agent.apply_fitness_outcome()
        self.remove_dead()
        self.time_step += 1
        self.reproduce()


class SpatialPopulation(Population):
    """Use baseline agents and genetic operators with local spatial parent choice."""

    def __init__(self, model: SpatialWorld, config: SpatialConfig,
                 population_config: PopulationConfig | None = None):
        baseline_config = replace(population_config or PopulationConfig(),
                                  genome_labels=list(model.genome_labels))
        super().__init__(model, baseline_config)
        self.spatial_config = config

    def can_reproduce(self, agent: Individual) -> bool:
        """Shared readiness signal for observations and actual parent selection."""
        cfg = self.spatial_config
        energy = cfg.reproduction_energy if agent.gender == Gender.FEMALE else 0
        return (len(self.model.agents) < cfg.max_population and cfg.reproduction_rate > 0
                and agent.energy > 0 and agent.can_reproduce(cfg.min_reproduction_age, energy))

    def remove_dead(self) -> dict[str, int]:
        organisms = self.model.organisms
        for agent in organisms.values():
            if not agent.is_alive:
                continue
            if agent.energy <= 0:
                agent.is_alive = False
                agent.death_cause = DeathCause.ENERGY
                agent.age += 1
            else:
                agent.fitness = agent.compute_fitness()
                agent.apply_fitness_outcome()
        dead = [agent for agent in organisms.values() if not agent.is_alive]
        deaths = {
            "age": sum(agent.death_cause == DeathCause.AGE for agent in dead),
            "fitness": sum(agent.death_cause == DeathCause.FITNESS for agent in dead),
            "threshold": sum(agent.death_cause == DeathCause.THRESHOLD for agent in dead),
            "energy": sum(agent.death_cause == DeathCause.ENERGY for agent in dead),
            "competition": sum(agent.death_cause == DeathCause.COMPETITION for agent in dead),
        }
        for agent in dead:
            self.model._dead_lifetime_sum += agent.age
            self.model._dead_count += 1
            self.model.remove_organism(agent.unique_id)
        self.invalidate_metrics()
        if dead:
            self.invalidate_batch_cache()
        return deaths

    def reproduce(self) -> int:
        cfg = self.spatial_config
        births = 0
        parents = list(self.model.organisms.values())
        parent_ids = {agent.unique_id for agent in parents}
        for mother in parents:
            if len(self.model.agents) >= cfg.max_population:
                break
            if mother.gender != Gender.FEMALE or not self.can_reproduce(mother):
                continue
            mates = sorted((father for father in self.model.space.get_neighbors(mother.pos, self.model.mating_distance)
                            if father.unique_id in parent_ids and father is not mother
                            and father.gender == Gender.MALE
                            and self.can_reproduce(father)),
                           key=lambda agent: agent.unique_id)
            if not mates or self.model.rng.random() >= mother.reproduction_probability(cfg.reproduction_rate):
                continue
            father = mates[int(self.model.rng.choice(len(mates), p=self._selection_weights(mates)))]
            genome, pre_mutation_genome, mutation_deltas = crossover_and_mutate(mother, father, self.model)
            direction = float(self.model.rng.uniform(-pi, pi))
            offset = float(self.model.rng.uniform(0, cfg.offspring_offset))
            x = float(np.clip(mother.state.x + offset * cos(direction), 0, cfg.width))
            y = float(np.clip(mother.state.y + offset * sin(direction), 0, cfg.height))
            child = self.model.add_organism(genome, x, y, energy=cfg.offspring_energy,
                                            parent_ids=(mother.unique_id, father.unique_id))
            mother.spend_energy(cfg.offspring_energy)
            self.record_birth(child.unique_id, mother, father, pre_mutation_genome, mutation_deltas)
            births += 1
        self.invalidate_metrics()
        if births:
            self.invalidate_batch_cache()
        return births
