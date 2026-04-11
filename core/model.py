import mesa
import numpy as np
from dataclasses import dataclass

from .enums import DeathCause
from .environment import Environment
from .population import Population


@dataclass
class ModelConfig:
    n_individuals: int = 100
    n_genes: int = 5
    initial_temp: float = 20.0


class Model(mesa.Model):
    def __init__(self, config: ModelConfig = None):
        super().__init__(seed=42)
        self.config = config or ModelConfig()
        self.step_count = 0

        self.environment = Environment(self)

        self.population = Population(self, config.n_individuals)

        self.deaths_this_step = {DeathCause.AGE: 0, DeathCause.FITNESS: 0, DeathCause.THRESHOLD: 0}
        self.births_this_step = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": "step_count",
                "Generation": lambda m: m.population.generation,
                "PopulationSize": lambda m: len([a for a in m.agents if a.is_alive]),

                "AvgFitness": lambda m: float(
                    np.mean([a.fitness for a in m.agents if a.fitness is not None and a.is_alive] or [0])),
                "AvgAge": lambda m: float(np.mean([a.age for a in m.agents if a.is_alive])),

                "BirthCount": lambda m: m.births_this_step,

                "Deaths_age": lambda m: m.deaths_this_step[DeathCause.AGE],
                "Deaths_fitness": lambda m: m.deaths_this_step[DeathCause.FITNESS],
                "Deaths_threshold": lambda m: m.deaths_this_step[DeathCause.THRESHOLD],
                "Deaths_total": lambda m: sum(m.deaths_this_step.values()),
            },

            agent_reporters={
                "Fitness": "fitness",
                "Age": "age",
            }
        )

        self.population.initialize()

    def step(self):
        self.births_this_step = 0
        for key in self.deaths_this_step:
            self.deaths_this_step[key] = 0

        self.environment.step()
        self.population.step()
        self.step_count += 1
        self.datacollector.collect(self)

    def run(self, n_steps=100):
        for _ in range(n_steps):
            if len(self.agents) == 0:
                print(f"Population extinct at step {self.step_count}")
                break
            print("step", _)
            print("AvgFitnass", float(self.datacollector.model_reporters["AvgFitness"](self)))

            print("AvgHeatResistance", float(np.mean([
                a.genes_map["heat_resistance"]
                for a in self.agents if a.is_alive and a.fitness is not None
            ])))

            print("AvgColdResistance", float(np.mean([
                a.genes_map["cold_resistance"]
                for a in self.agents if a.is_alive and a.fitness is not None
            ])))

            print("AvgResilience", float(np.mean([
                a.genes_map["resilience"]
                for a in self.agents if a.is_alive and a.fitness is not None
            ])))

            self.step()
