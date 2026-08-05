from __future__ import annotations

from dataclasses import dataclass

import mesa
from config.sim_config import SimConfig, validate_sim_config

from .enums import DeathCause, RunStatus
from .environment import Environment
from .mutation_patterns import MutationStrategy, BaselineMutation
from .population import Population
from .recording import EvolutionRecorder


@dataclass(frozen=True)
class SimulationResult:
    """Summary of a completed or prematurely terminated simulation run."""

    status: RunStatus
    completed_steps: int


class Model(mesa.Model):
    """
    Main simulation model coordinating environment, population, and data collection.

    The model executes a step-based simulation loop and tracks key statistics.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        mutation_strategy: MutationStrategy | None = None,
        recorder: EvolutionRecorder | None = None,
    ):
        """
        Initialize the simulation model.

        Args:
            config: Simulation configuration. If None, default SimConfig is used.

        Initializes:
            - Random number generator
            - Environment and population subsystems
            - Optional event recorder for external analytics / ML pipelines
            - Counters for births and deaths
            - DataCollector for tracking simulation metrics
        """

        self.config = config or SimConfig()
        validate_sim_config(self.config)
        super().__init__(rng=self.config.seed)
        self.step_count = 0

        self.mutation_strategy = mutation_strategy or BaselineMutation(self.config.population.mutation_std)

        self.environment = Environment(self, config=self.config.environment)

        self.population = Population(self, config=self.config.population)
        self.recorder = recorder

        self.deaths_this_step = {DeathCause.AGE: 0, DeathCause.FITNESS: 0, DeathCause.THRESHOLD: 0,
                                 DeathCause.COMPETITION: 0}
        self.births_this_step = 0
        self.attacks_this_step = {
            "considered": 0,
            "declined": 0,
            "started": 0,
            "lost": 0,
            "won": 0,
            "food_stolen": 0,
            "energy_spent": 0.0,
        }
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": lambda m: m.step_count,
                "Generation": lambda m: m.population.generation,
                "PopulationSize": lambda m: m.population.actual_pop_size,
                "FemaleCount": lambda m: m.population.count_females,
                "MaleCount": lambda m: m.population.count_males,
                "FoodCapacity": lambda m: m.environment.food_capacity,
                "FoodAvailable": lambda m: m.environment.current_food,

                "AvgFitness": lambda m: m.population.avg_fitness,
                "AvgAge": lambda m: m.population.avg_age,
                "AvgSatiation": lambda m: m.population.avg_satiation,
                "AvgEnergy": lambda m: m.population.avg_energy,
                "AvgHeatResistance": lambda m: m.population.avg_heat_resistance,
                "AvgColdResistance": lambda m: m.population.avg_cold_resistance,
                "AvgMetabolicRate": lambda m: m.population.avg_metabolic_rate,
                "AvgResilience": lambda m: m.population.avg_resilience,
                "AvgSize": lambda m: m.population.avg_size,
                "AvgSpeed": lambda m: m.population.avg_speed,
                "AvgAggressiveness": lambda m: m.population.avg_aggressiveness,

                "BirthCount": lambda m: m.births_this_step,

                "Deaths_age": lambda m: m.deaths_this_step[DeathCause.AGE],
                "Deaths_fitness": lambda m: m.deaths_this_step[DeathCause.FITNESS],
                "Deaths_threshold": lambda m: m.deaths_this_step[DeathCause.THRESHOLD],
                "Deaths_competition": lambda m: m.deaths_this_step[DeathCause.COMPETITION],
                "Deaths_total": lambda m: sum(m.deaths_this_step.values()),
                "PopulationDelta": lambda m: m.births_this_step - sum(m.deaths_this_step.values()),

                "AttacksConsidered": lambda m: m.attacks_this_step["considered"],
                "AttacksDeclined": lambda m: m.attacks_this_step["declined"],
                "AttacksStarted": lambda m: m.attacks_this_step["started"],
                "AttacksLost": lambda m: m.attacks_this_step["lost"],
                "AttacksWon": lambda m: m.attacks_this_step["won"],
                "FoodStolen": lambda m: m.attacks_this_step["food_stolen"],
                "AttackEnergySpent": lambda m: m.attacks_this_step["energy_spent"],
            },

            agent_reporters={
                "Fitness": "fitness",
                "Age": "age",
                "Energy": "energy",
            }
        )

        self.population.initialize()
        self.datacollector.collect(self)

    def _display_info(self):
        print("step", self.step_count)

        print("Temperature", float(self.environment.current_params["temperature"]))

        print("HazardLevel", float(self.environment.current_params["hazard_level"]))

        print("FoodCapacity", float(self.environment.food_capacity))

        print("FoodAvailable", float(self.environment.current_food))

        print("AvgFitness", float(self.datacollector.model_reporters["AvgFitness"](self)))

        print("AvgEnergy", self.population.avg_energy)

        print("AvgHeatResistance", self.population.avg_heat_resistance)

        print("AvgColdResistance", self.population.avg_cold_resistance)

        print("AvgSize", self.population.avg_size)

        print("AvgSpeed", self.population.avg_speed)

        print("AvgResilience", self.population.avg_resilience)

        print("AvgAggressivness", self.population.avg_aggressiveness)

        print("Females", self.population.count_females)

        print("Males", self.population.count_males, "\n")

    def step(self):
        """Advance the simulation by one time step."""

        self.births_this_step = 0
        for key in self.deaths_this_step:
            self.deaths_this_step[key] = 0
        for key in self.attacks_this_step:
            self.attacks_this_step[key] = 0

        self.environment.step()
        self.population.step()
        self.step_count += 1
        self.datacollector.collect(self)

        if self.config.debug:
            self._display_info()

    def run(self, n_steps: int = 100) -> SimulationResult:
        """
        Run the simulation for a given number of steps.

        The simulation stops early if extinction conditions are met:
            - No agents remain
            - Only one gender remains (no reproduction possible)

        Args:
            n_steps: Maximum number of steps to simulate.
        """

        status = self._terminal_status()
        if status is not None:
            return SimulationResult(status=status, completed_steps=self.step_count)

        for _ in range(n_steps):
            self.step()

            status = self._terminal_status()
            if status is not None:
                return SimulationResult(status=status, completed_steps=self.step_count)

        return SimulationResult(status=RunStatus.COMPLETED, completed_steps=self.step_count)

    def _terminal_status(self) -> RunStatus | None:
        if len(self.agents) == 0:
            return RunStatus.EXTINCT
        if self.population.count_females == 0:
            return RunStatus.NO_FEMALES
        if self.population.count_males == 0:
            return RunStatus.NO_MALES
        return None
