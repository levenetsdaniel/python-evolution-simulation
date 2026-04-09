import mesa
import numpy as np
from dataclasses import dataclass
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

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": "step_count",
                "Generation": lambda m: m.population.generation,
                "PopulationSize": lambda m: len(m.agents),
                "AvgFitness": lambda m: np.mean([a.fitness for a in m.agents if a.fitness is not None] or [0]),
                "Temperature": lambda m: m.environment.current_params["temperature"],
            },

            agent_reporters={
                "Fitness": "fitness",
                "Age": "age",
                "IsAlive": "is_alive",
            }
        )

        labels = ["heat_resistance", "cold_resistance", "speed", "size", "humidity"]
        ranges = [(0.0, 1.0)] * 5
        self.population.initialize(labels, ranges)

    def step(self):
        self.environment.step()
        self.population.step()
        self.datacollector.collect(self)
        self.step_count += 1

    def run(self, n_steps=100):
        for _ in range(n_steps):
            if len(self.agents) == 0:
                print(f"Population extinct at step {self.step_count}")
                break
            self.step()
