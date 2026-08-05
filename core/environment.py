from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

import numpy as np
from config.sim_config import EnvironmentConfig


class Environment:
    """
    Represents the environment in which the population evolves.

    The environment maintains dynamic parameters such as temperature,
    food availability, and hazard level. These parameters evolve over time
    and directly influence agent survival, fitness, and interactions.
    """

    def __init__(self, model: Model, config: EnvironmentConfig | None = None):
        """
        Initialize the environment with initial parameters.

        Args:
            model: Reference to the parent simulation model.
            config: Environment configuration. If None, defaults are used.

        Initializes:
            time: Current simulation time.
            current_params: Active environmental parameters.
            prev_params: Parameters from the previous step (used for deltas).
        """

        self.model = model
        self.config = config or EnvironmentConfig()

        self.time = 0
        self.current_params = {
            "temperature": self.config.temp_start,
            "food_availability": self.config.food_availability,
            "hazard_level": self.config.hazard_level_start,
        }
        self.food_capacity = self.config.food_availability
        self.current_food = self.food_capacity
        self.prev_params = self.current_params.copy()
        self._temperature_step = self.config.temp_step
        self._hazard_step = self.config.hazard_step

    def food_distribution(self):
        """
        Distribute available food among alive agents.

        Agents are processed in weighted random order.

        Agents with enough available food are fully fed.
        Remaining agents become "hungry".
        Hungry agents may compete with fed agents for resources.
        """

        self.model.population.remove_food()
        alive = [a for a in self.model.agents if a.is_alive]

        if not alive:
            self.current_food = 0.0
            return

        raw_weight = np.array([a.genes_map["speed"] * 2.0 / a.food_need for a in alive])
        raw_weight = np.clip(raw_weight, 1e-6, None)
        weights = raw_weight / raw_weight.sum()

        ordered = list(self.model.rng.choice(alive, size=len(alive), replace=False, p=weights))

        fed = []
        hungry = []

        for a in ordered:
            if a.food_need <= self.current_food:
                a.food_eaten = a.food_need
                self.current_food -= a.food_eaten
                fed.append(a)
            else:
                hungry.append(a)

        if hungry:
            self.model.population.compete(hungry, fed)

    def step(self):
        """
        Advance the environment by one step.

        Updates environmental parameters according to configuration:

        Triggers food distribution among agents.
        """

        self.prev_params = self.current_params.copy()
        self.time += 1

        self._regenerate_food()

        self.current_params["temperature"] = self._next_temperature()

        self.current_params["hazard_level"] = self._next_hazard_level()
        self.food_distribution()

    def _regenerate_food(self) -> None:
        """Restore a configured share of the missing food up to the capacity."""
        missing_food = max(0.0, self.food_capacity - self.current_food)
        regenerated = self.config.food_regeneration_rate * missing_food
        self.current_food = min(self.food_capacity, self.current_food + regenerated)

    def _next_temperature(self) -> float:
        """Advance temperature and reflect it at configured physical bounds."""
        temperature = self.current_params["temperature"] + self._temperature_step
        minimum = self.config.min_temperature
        maximum = self.config.max_temperature

        while temperature > maximum or temperature < minimum:
            if temperature > maximum:
                temperature = maximum - (temperature - maximum)
                self._temperature_step = -abs(self._temperature_step)
            else:
                temperature = minimum + (minimum - temperature)
                self._temperature_step = abs(self._temperature_step)

        return temperature

    def _next_hazard_level(self) -> float:
        """Advance hazard level and reflect it at the valid probability bounds."""
        hazard_level = self.current_params["hazard_level"] + self._hazard_step

        while hazard_level > 1.0 or hazard_level < 0.0:
            if hazard_level > 1.0:
                hazard_level = 1.0 - (hazard_level - 1.0)
                self._hazard_step = -abs(self._hazard_step)
            else:
                hazard_level = -hazard_level
                self._hazard_step = abs(self._hazard_step)

        return hazard_level
