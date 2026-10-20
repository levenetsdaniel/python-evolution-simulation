from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from math import hypot

if TYPE_CHECKING:
    from .model import Model
    from .spatial import SpatialWorld

import numpy as np
from config.sim_config import EnvironmentConfig, SpatialConfig


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
                a.record_food(a.food_need)
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

        self.advance_conditions()
        self._regenerate_food()
        self.food_distribution()

    def advance_conditions(self) -> None:
        """Advance climate and hazard before agents are evaluated this step."""
        self.prev_params = self.current_params.copy()
        self.time += 1
        self.current_params["temperature"] = self._next_temperature()
        self.current_params["hazard_level"] = self._next_hazard_level()

    def conditions_at(self, x: float, y: float) -> dict[str, float]:
        """Return conditions at a point; currently uniform across the world."""
        return self.current_params

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


@dataclass
class FoodSource:
    x: float
    y: float
    stock: float
    capacity: float
    regeneration: float
    radius: float

    def restore(self) -> None:
        self.stock = min(self.capacity, self.stock + self.regeneration)


class SpatialEnvironment(Environment):
    """Spatial food supply, consumption, disappearance and relocation."""

    def __init__(self, model: SpatialWorld, config: SpatialConfig,
                 environment_config: EnvironmentConfig | None = None):
        super().__init__(model, environment_config)
        self.spatial_config = config
        self.food_sources: list[FoodSource] = []
        self.disappeared_this_step = 0
        self.spawned_this_step = 0
        self._generate_food()
        self.food_capacity = config.cluster_count * config.sources_per_cluster * config.food_capacity
        self._sync_food_supply()

    def _sync_food_supply(self) -> None:
        """Expose spatial food totals through the shared environment interface."""
        self.current_food = sum(source.stock for source in self.food_sources)
        self.current_params["food_availability"] = self.current_food

    def _new_source(self, x: float, y: float) -> FoodSource:
        cfg = self.spatial_config
        return FoodSource(float(x), float(y), cfg.food_initial, cfg.food_capacity,
                          cfg.food_regeneration, cfg.food_radius)

    def _generate_food(self) -> None:
        cfg = self.spatial_config
        for _ in range(cfg.cluster_count):
            cx = self.model.rng.uniform(0, cfg.width)
            cy = self.model.rng.uniform(0, cfg.height)
            for _ in range(cfg.sources_per_cluster):
                x = np.clip(self.model.rng.normal(cx, cfg.cluster_spread), 0, cfg.width)
                y = np.clip(self.model.rng.normal(cy, cfg.cluster_spread), 0, cfg.height)
                self.food_sources.append(self._new_source(x, y))

    def nearest_food(self, x: float, y: float, radius: float) -> FoodSource | None:
        candidates = ((hypot(food.x - x, food.y - y), index, food)
                      for index, food in enumerate(self.food_sources) if food.stock > 0)
        return next((food for distance, _, food in sorted(candidates) if distance <= radius), None)

    def food_distribution(self) -> None:
        """Allocate by ID, then apply baseline competition within each source."""
        self.model.population.remove_food()
        requests = {}
        for organism in self.model.organisms.values():
            x, y = organism.state.x, organism.state.y
            reachable = [(hypot(food.x - x, food.y - y), index)
                         for index, food in enumerate(self.food_sources)
                         if food.stock > 0 and hypot(food.x - x, food.y - y) <= food.radius]
            if reachable and organism.energy < self.spatial_config.max_energy:
                _, index = min(reachable)
                requests.setdefault(index, []).append(organism)
        for index in sorted(requests):
            source = self.food_sources[index]
            organisms = requests[index]
            offset = (self.model.step_count + index) % len(organisms)
            organisms = organisms[offset:] + organisms[:offset]
            remaining = source.stock
            fed = []
            hungry = []
            for organism in organisms:
                amount = min(remaining, self.spatial_config.food_bite, organism.food_need)
                if amount > 1e-12:
                    organism.record_food(amount)
                    remaining = max(0.0, remaining - amount)
                    fed.append(organism)
                else:
                    hungry.append(organism)
            source.stock = remaining
            if hungry and fed:
                self.model.population.compete(hungry, fed)
            for organism in organisms:
                if organism.is_alive:
                    organism.gain_energy(organism.food_eaten * self.spatial_config.food_energy)
        self._sync_food_supply()

    def step(self) -> None:
        """Advance shared climate and hazard; food regrows after reproduction."""
        self.advance_conditions()

    def update_resources(self) -> None:
        """Remove exhausted sources by chance, restore survivors, then fill vacancies elsewhere.

        Sources are checked in list order. Each vacant slot has an independent
        spawn trial; new positions are uniform across the whole world.
        """
        self.disappeared_this_step = 0
        self.spawned_this_step = 0
        survivors = []
        for source in self.food_sources:
            if source.stock <= 1e-12 and self.model.rng.random() < self.spatial_config.food_disappearance_probability:
                self.disappeared_this_step += 1
            else:
                source.restore()
                survivors.append(source)
        self.food_sources[:] = survivors
        vacancies = self.spatial_config.cluster_count * self.spatial_config.sources_per_cluster - len(self.food_sources)
        for _ in range(max(0, vacancies)):
            if self.model.rng.random() < self.spatial_config.food_spawn_probability:
                x = self.model.rng.uniform(0, self.spatial_config.width)
                y = self.model.rng.uniform(0, self.spatial_config.height)
                self.food_sources.append(self._new_source(x, y))
                self.spawned_this_step += 1
        self._sync_food_supply()
