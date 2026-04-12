import mesa
import numpy as np
from dataclasses import dataclass

from .enums import DeathCause, Gender
from .environment import Environment
from .population import Population
from .training_buffer import TrainingBuffer


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

        self.training_buffer = TrainingBuffer()

        self.deaths_this_step = {DeathCause.AGE: 0, DeathCause.FITNESS: 0, DeathCause.THRESHOLD: 0}
        self.births_this_step = 0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": "step_count",
                "Generation": lambda m: m.population.generation,
                "PopulationSize": lambda m: m.population.actual_size,

                "AvgFitness": lambda m: m.population.avg_fitness,
                "AvgAge": lambda m: m.population.avg_age,

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

            if self.population.count_females == 0:
                print(f"Population extinct at step {self.step_count}, all males died")
                break

            if self.population.count_males == 0:
                print(f"Population extinct at step {self.step_count}, all females died")
                break

            print("step", _)
            print("AvgFitness", float(self.datacollector.model_reporters["AvgFitness"](self)))

            print("FoodCount", float(self.environment.current_params["food_availability"]))

            print("Temperature", float(self.environment.current_params["temperature"]))

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

            print("Females", len([s for s in self.agents if s.gender == Gender.FEMALE]))

            self.step()

            if _ > 0:
                self.training_buffer.save("data/training_samples.json")
                x, y, weights = self.training_buffer.to_numpy()
                print(f"Training data: x={x.shape}, y={y.shape}")
