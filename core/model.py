from __future__ import annotations

import mesa
from config.sim_config import SimConfig

from .enums import DeathCause
from .environment import Environment
from .mutation_patterns import MutationStrategy, BaselineMutation
from .population import Population
from .training_buffer import TrainingBuffer


class Model(mesa.Model):
    """
    Main simulation model coordinating environment, population, and data collection.

    The model executes a step-based simulation loop and tracks key statistics.
    """

    def __init__(self, config: SimConfig | None = None, mutation_strategy: MutationStrategy | None = None):
        """
        Initialize the simulation model.

        Args:
            config: Simulation configuration. If None, default SimConfig is used.

        Initializes:
            - Random number generator
            - Environment and population subsystems
            - Training buffer for data collection
            - Counters for births and deaths
            - DataCollector for tracking simulation metrics
        """

        self.config = config or SimConfig()
        super().__init__(rng=self.config.seed)
        self.step_count = 0

        self.mutation_strategy = mutation_strategy or BaselineMutation(self.config.population.mutation_std)

        self.environment = Environment(self, config=self.config.environment)

        self.population = Population(self, config=self.config.population)

        self.training_buffer = TrainingBuffer()

        self.deaths_this_step = {DeathCause.AGE: 0, DeathCause.FITNESS: 0, DeathCause.THRESHOLD: 0,
                                 DeathCause.COMPETITION: 0}
        self.births_this_step = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": lambda m: m.step_count,
                "Generation": lambda m: m.population.generation,
                "PopulationSize": lambda m: m.population.actual_pop_size,

                "AvgFitness": lambda m: m.population.avg_fitness,
                "AvgAge": lambda m: m.population.avg_age,
                "AvgSatiation": lambda m: m.population.avg_satiation,
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
            },

            agent_reporters={
                "Fitness": "fitness",
                "Age": "age",
            }
        )

        self.population.initialize()

    def _display_info(self):
        print("step", self.step_count)

        print("Temperature", float(self.environment.current_params["temperature"]))

        print("HazardLevel", float(self.environment.current_params["hazard_level"]))

        print("FoodCount", float(self.environment.current_params["food_availability"]))

        print("AvgFitness", float(self.datacollector.model_reporters["AvgFitness"](self)))

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

        self.environment.step()
        self.population.step()
        self.step_count += 1
        self.datacollector.collect(self)

        if self.config.debug:
            self._display_info()

    def run(self, n_steps: int = 100):
        """
        Run the simulation for a given number of steps.

        The simulation stops early if extinction conditions are met:
            - No agents remain
            - Only one gender remains (no reproduction possible)

        Args:
            n_steps: Maximum number of steps to simulate.
        """

        for _ in range(n_steps):
            self.step()

            if len(self.agents) == 0:
                print(f"Population extinct at step {self.step_count}")
                break

            if self.population.count_females == 0:
                print(f"Population extinct at step {self.step_count}, all males died")
                break

            if self.population.count_males == 0:
                print(f"Population extinct at step {self.step_count}, all females died")
                break

            if self.config.debug:
                self._display_info()

            if self.config.record:
                self.training_buffer.save(self.config.output_path)
                try:
                    x, y, weights = self.training_buffer.to_numpy()
                except ValueError:
                    x = y = weights = None

                if self.config.debug and x is not None and y is not None:
                    print(f"Training data: x={x.shape}, y={y.shape}")
