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
        self.current_food = self.current_params["food_availability"]
        self.prev_params = self.current_params.copy()

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

        raw_weight = np.array([a.genes_map["speed"] * 2.0 / a.food_need for a in alive])
        raw_weight = np.clip(raw_weight, 1e-6, None)
        weights = raw_weight / raw_weight.sum()

        ordered = list(self.model.rng.choice(alive, size=len(alive), replace=False, p=weights))

        fed = []
        hungry = []

        for a in ordered:
            if a.food_need < self.current_food:
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

        self.current_params["food_availability"] += self.config.food_step
        if self.current_params["food_availability"] < 0:
            self.current_params["food_availability"] = self.config.food_step
        self.current_food = self.current_params["food_availability"]

        self.current_params["temperature"] += self.config.temp_step
        if self.current_params["temperature"] >= self.config.max_temperature:
            self.current_params["temperature"] = self.config.temp_reset

        if self.current_params["temperature"] <= self.config.min_temperature:
            self.current_params["temperature"] = self.config.temp_reset

        self.current_params["hazard_level"] += self.config.hazard_step
        if self.current_params["hazard_level"] < 0 or self.current_params["hazard_level"] > 1:
            self.current_params["hazard_level"] = self.config.hazard_level_start



        self.food_distribution()
