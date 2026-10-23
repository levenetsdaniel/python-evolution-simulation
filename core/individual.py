import mesa
import numpy as np

from config.sim_config import IndividualConfig
from .enums import Gender, DeathCause
from .fitness import fitness

DEFAULT_INDIVIDUAL_CONFIG = IndividualConfig()


class Individual(mesa.Agent):
    """
    Represents a single agent (organism) in the evolutionary simulation.

    Each individual is defined by a genome — a vector of continuous traits,
    which determine its survival and reproduction capabilities.
    """

    def __init__(self, model: mesa.Model, genome: np.ndarray, genome_labels: list[str], parent_ids: tuple | None,
                 birth_step: int = 0, age: int = 0):
        """
        Initialize an individual agent.

        Args:
            model: Reference to the simulation model.
            genome: Array of gene values (0–1 range).
            genome_labels: Names corresponding to genome entries.
            parent_ids: Tuple of parent IDs, or None if randomly initialized.
            birth_step: Simulation step at which the individual was born.
            age: Initial age.

        Initializes:
            - genes_map: Mapping from gene names to values
            - gender: Randomly assigned biological sex
            - food_need: Computed from genome traits
            - fitness: Initially None
            - is_alive: Alive state flag
        """

        super().__init__(model)
        model_config = getattr(self.model, "config", None)
        self.config = (getattr(self.model, "individual_config", None)
                       or getattr(model_config, "individual", None)
                       or DEFAULT_INDIVIDUAL_CONFIG)
        self._fitness_config = (getattr(self.model, "fitness_config", None)
                                or getattr(model_config, "fitness", None))
        self._environment_config = (getattr(self.model, "environment_config", None)
                                    or getattr(model_config, "environment", None))
        self.genome = np.array(genome, dtype=float)
        self.genome_labels = genome_labels
        if self.genome.ndim != 1 or len(self.genome) != len(self.genome_labels):
            raise ValueError("genome must be one-dimensional and match genome_labels")
        if not np.isfinite(self.genome).all() or not ((0.0 <= self.genome) & (self.genome <= 1.0)).all():
            raise ValueError("genome values must be finite and within [0, 1]")
        self.parent_ids = parent_ids
        self.birth_step = birth_step
        self.genes_map = dict(zip(self.genome_labels, self.genome))
        self.gender = self.model.rng.choice(list(Gender))
        self.death_cause = None

        self.age = age
        self.food_eaten = 0
        self.food_need = max(1, int(round((self.genes_map["size"] * self.config.food_need_size_coef +
                                           self.genes_map["resilience"] * self.config.food_need_resilience_coef +
                                           self.genes_map["speed"] * self.config.food_need_speed_coef +
                                           self.genes_map["aggressiveness"] * self.config.food_need_aggr_coef) * 10)))
        self.energy = self.config.energy_initial
        self.fitness = None
        self.is_alive = True

    def __getitem__(self, item: str) -> float:
        """
        Access gene value by name.

        Args:
            item: Gene label.

        Returns:
            Value of the corresponding gene.
        """

        return self.genes_map[item]

    @property
    def satiation(self) -> float:
        """Return the normalized energy reserve used for survival and reproduction."""

        return self.energy / self.config.energy_capacity

    @property
    def food_satiation(self) -> float:
        """Return the fraction of this step's food requirement that was met."""

        return min(1.0, self.food_eaten / self.food_need)

    def spend_energy(self, amount: float) -> None:
        """Spend energy without letting the reserve become negative."""
        self.energy = max(0.0, self.energy - amount)

    def gain_energy(self, amount: float) -> None:
        """Add energy without exceeding the individual's configured capacity."""
        self.energy = min(self.config.energy_capacity, self.energy + amount)

    def eat(self, amount: float, energy_per_unit: float) -> None:
        """Record food intake and convert it to energy using this mode's units."""
        self.record_food(amount)
        self.gain_energy(amount * energy_per_unit)

    def record_food(self, amount: float) -> None:
        """Record food before energy conversion or competition transfers it."""
        self.food_eaten += amount

    def can_reproduce(self, min_age: int, min_energy: float) -> bool:
        return self.is_alive and self.age >= min_age and self.energy >= min_energy

    def reproduction_probability(self, rate: float) -> float:
        return float(np.clip(rate * self.satiation, 0.0, 1.0))

    def metabolic_cost(self) -> float:
        """Return the shared per-step cost of living for this genome."""
        return float(
            self.config.basal_energy_cost
            + self["speed"] * self.config.speed_energy_cost
            + self["aggressiveness"] * self.config.aggressiveness_energy_cost
        )

    def age_death_probability(self) -> float:
        """Return age-related mortality after this step's age increment."""
        return (1 - np.exp(-self.age / self.config.age_scale)) ** self.config.age_death_power

    def _update_energy(self) -> None:
        """Convert consumed food into energy and pay this step's metabolic costs."""
        self.gain_energy(self.food_eaten * self.config.food_energy_conversion)
        self.spend_energy(self.metabolic_cost())

    def compute_fitness(self) -> float:
        """
        Compute fitness score for the individual.

        Returns:
            Fitness score in range [score_floor, 1.0].
        """

        state = getattr(self, "state", None)
        environment = self.model.environment
        conditions = (environment.conditions_at(state.x, state.y) if state is not None
                      else environment.current_params)
        return fitness(
            self.genes_map,
            conditions,
            fitness_config=self._fitness_config,
            environment_config=self._environment_config,
            satiation=self.satiation,
        )

    def step(self):
        """
        Advance the individual by one simulation step.

        Checks if the agent is dead.
        """

        if not self.is_alive:
            return

        self._update_energy()
        self.fitness = self.compute_fitness()
        self.apply_fitness_outcome()

    def apply_fitness_outcome(self) -> None:
        """Record an assigned fitness value and apply its survival consequences."""
        recorder = getattr(self.model, "recorder", None)
        if recorder is not None:
            recorder.record_fitness(self.unique_id, self.fitness)
        if self.fitness < self.config.fitness_death_threshold:
            self.is_alive = False
            self.death_cause = DeathCause.THRESHOLD
            return

        self.age += 1
        age_death_prob = self.age_death_probability()
        fitness_death_prob = (1 - self.fitness) * self.config.fitness_death_prob_coef
        death_prob = 1 - (1 - age_death_prob) * (1 - fitness_death_prob)

        if self.model.rng.random() < death_prob:
            self.is_alive = False
            self.death_cause = DeathCause.FITNESS if fitness_death_prob > age_death_prob else DeathCause.AGE
            return

    @classmethod
    def random_init(cls, model: mesa.Model, genome_labels: list[str], birth_step: int = 0,
                    age: int = 2) -> "Individual":
        """
        Create a randomly initialized individual.

        Args:
            model: Simulation model.
            genome_labels: Names of genome traits.
            birth_step: Initial simulation step.
            age: Initial age.

        Returns:
            New randomly initialized Individual.
        """

        genome = model.rng.random(len(genome_labels))

        return cls(
            model=model,
            genome=genome,
            parent_ids=None,
            genome_labels=genome_labels,
            birth_step=birth_step,
            age=age
        )

    @classmethod
    def from_parents(cls, model: mesa.Model, p1: "Individual", p2: "Individual", genome: np.ndarray,
                     birth_step: int = 0) -> "Individual":
        """
        Create an individual from two parents.

        Args:
            model: Simulation model.
            p1: First parent.
            p2: Second parent.
            genome: Resulting genome after recombination and mutation.
            birth_step: Birth simulation step.

        Returns:
            New Individual with inherited parent IDs.
        """

        return cls(
            model=model,
            genome=genome,
            genome_labels=p1.genome_labels,
            birth_step=birth_step,
            parent_ids=(p1.unique_id, p2.unique_id),
        )
