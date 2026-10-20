"""Continuous 2D evolution using Mesa agents and a headless spatial model."""

from __future__ import annotations

import csv
from dataclasses import replace
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from typing import Mapping

import numpy as np
import mesa
from mesa.space import ContinuousSpace

from config.sim_config import (EnvironmentConfig, FitnessConfig, IndividualConfig,
                               PopulationConfig, SimConfig,
                               SpatialConfig, validate_sim_config, validate_spatial_config)
from .environment import FoodSource, SpatialEnvironment
from .mutation_patterns import BaselineMutation, MutationStrategy
from .population import SpatialPopulation
from .recording import EvolutionRecorder
from .spatial_individual import SpatialIndividual, SpatialState
from .spatial_policy import (FoodObservation, HeuristicPolicy, MoveAction,
                             SpatialObservation, SpatialObservationProvider)


class SpatialWorld(mesa.Model):
    """Owns spatial state and executes simultaneous actions in a fixed step order.

    Organisms are processed by ascending ID. Each source rotates this fixed
    order by step and source index; those without food may compete locally.
    """

    def __init__(self, config: SpatialConfig | None = None, seed: int = 42,
                 genome_labels: list[str] | None = None,
                 individual_config: IndividualConfig | None = None,
                 environment_config: EnvironmentConfig | None = None,
                 fitness_config: FitnessConfig | None = None,
                 mutation_strategy: MutationStrategy | None = None,
                 population_config: PopulationConfig | None = None,
                 recorder: EvolutionRecorder | None = None):
        self.config = config or SpatialConfig()
        validate_spatial_config(self.config)
        super().__init__(rng=seed)
        self.individual_config = replace(individual_config or IndividualConfig(),
                                         energy_initial=self.config.initial_energy,
                                         energy_capacity=self.config.max_energy)
        self.environment_config = environment_config or EnvironmentConfig()
        self.fitness_config = fitness_config or FitnessConfig()
        self.population_config = population_config or PopulationConfig()
        validate_sim_config(SimConfig(individual=self.individual_config, spatial=self.config,
                                      environment=self.environment_config, fitness=self.fitness_config,
                                      population=self.population_config))
        self.mutation_strategy = mutation_strategy or BaselineMutation(self.config.mutation_std)
        self.recorder = recorder
        self.genome_labels = tuple(genome_labels or self.population_config.genome_labels)
        if set(self.genome_labels) != set(PopulationConfig().genome_labels) or len(self.genome_labels) != 7:
            raise ValueError("spatial genome_labels must contain the seven classic traits")
        self.width = self.config.width
        self.height = self.config.height
        self.space = ContinuousSpace(float(np.nextafter(self.width, np.inf)),
                                     float(np.nextafter(self.height, np.inf)), torus=False)
        self.step_count = 0
        self._pending_actions: Mapping[int, MoveAction] | None = None
        self._dead_count = 0
        self._dead_lifetime_sum = 0
        self.history: list[dict[str, float | int]] = []
        report_names = ("TimeStep", "PopulationSize", "AvgEnergy", "BirthCount",
                        "Deaths_age", "Deaths_fitness", "Deaths_threshold",
                        "Deaths_energy", "Deaths_competition", "Deaths_total", "AvgFitness",
                        "AttacksConsidered", "AttacksDeclined", "AttacksStarted",
                        "AttacksLost", "AttacksWon", "FoodStolen", "AttackEnergySpent",
                        "Temperature", "HazardLevel",
                        "FoodAvailable", "FoodSourceCount", "FoodSourcesDisappeared",
                        "FoodSourcesSpawned", "AvgAge", "AvgLifespan")
        report_names += tuple("Avg" + "".join(part.title() for part in trait.split("_"))
                              for trait in self.genome_labels)
        self.datacollector = mesa.DataCollector(model_reporters={
            name: (lambda model, column=name: model.history[-1][column]) for name in report_names
        })
        self.observation_provider = SpatialObservationProvider()
        self.policy = HeuristicPolicy()
        self.attacks_this_step = {
            "considered": 0, "declined": 0, "started": 0, "lost": 0,
            "won": 0, "food_stolen": 0, "energy_spent": 0.0,
        }
        self.population = SpatialPopulation(self, self.config, self.population_config)
        self.environment = SpatialEnvironment(self, self.config, self.environment_config)
        for _ in range(self.config.initial_population):
            self.add_organism(
                self.rng.random(len(self.genome_labels)),
                float(self.rng.uniform(0, self.width)),
                float(self.rng.uniform(0, self.height)),
            )
        self._record(0, {cause: 0 for cause in ("age", "fitness", "threshold", "energy", "competition")})

    @classmethod
    def from_sim_config(cls, config: SimConfig, *,
                        mutation_strategy: MutationStrategy | None = None,
                        recorder: EvolutionRecorder | None = None) -> SpatialWorld:
        """Bind the shared simulation settings to a spatial world."""
        return cls(config.spatial, seed=config.seed,
                   genome_labels=config.population.genome_labels,
                   individual_config=config.individual,
                   environment_config=config.environment,
                   fitness_config=config.fitness,
                   population_config=config.population,
                   mutation_strategy=mutation_strategy, recorder=recorder)

    @property
    def organisms(self) -> dict[int, SpatialIndividual]:
        """ID lookup backed by Mesa's registered agents."""
        return {agent.unique_id: agent for agent in sorted(self.agents, key=lambda item: item.unique_id)}

    @property
    def food_sources(self) -> list[FoodSource]:
        return self.environment.food_sources

    def add_organism(self, genome: np.ndarray, x: float, y: float, *, energy: float | None = None,
                     heading: float | None = None, parent_ids: tuple[int, int] | None = None) -> SpatialIndividual:
        if len(self.agents) >= self.config.max_population:
            raise ValueError("population limit reached")
        genes = np.asarray(genome, dtype=float).copy()
        if genes.shape != (len(self.genome_labels),) or not np.isfinite(genes).all() or np.any((genes < 0) | (genes > 1)):
            raise ValueError("genome must have one finite [0, 1] value per trait")
        if not np.isfinite((x, y)).all() or not (0 <= x <= self.width and 0 <= y <= self.height):
            raise ValueError("coordinates outside world")
        reserve = self.config.initial_energy if energy is None else energy
        if not np.isfinite(reserve) or not 0 <= reserve <= self.config.max_energy:
            raise ValueError("invalid energy")
        direction = float(self.rng.uniform(-pi, pi)) if heading is None else heading
        if not np.isfinite(direction):
            raise ValueError("invalid heading")
        individual = SpatialIndividual(self, genes, self.genome_labels,
                                       SpatialState(float(x), float(y), float(direction)),
                                       parent_ids, self.step_count)
        individual.energy = float(reserve)
        self.space.place_agent(individual, (float(x), float(y)))
        return individual

    def remove_organism(self, organism_id: int) -> None:
        organism = self.organisms[organism_id]
        organism.is_alive = False
        self.space.remove_agent(organism)
        organism.remove()

    def speed(self, organism: SpatialIndividual) -> float:
        return self.config.base_speed + self.config.speed_trait_scale * organism["speed"]

    def perception(self, organism: SpatialIndividual) -> float:
        return self.config.base_perception + self.config.perception_trait_scale * organism["resilience"]

    def nearest_food(self, x: float, y: float, radius: float) -> FoodSource | None:
        return self.environment.nearest_food(x, y, radius)

    def update_resources(self) -> None:
        self.environment.update_resources()

    def _actions(self, supplied: Mapping[int, MoveAction] | None) -> dict[int, MoveAction]:
        organisms = self.organisms
        if supplied is not None:
            unknown = set(supplied) - organisms.keys()
            if unknown:
                raise ValueError(f"unknown organism IDs: {sorted(unknown)}")
        result = {}
        for organism_id in sorted(organisms):
            organism = organisms[organism_id]
            result[organism_id] = (supplied[organism_id] if supplied is not None and organism_id in supplied
                                   else self.policy.choose(self.observation_provider.observe(self, organism),
                                                           self.rng, self.config.exploration_turn))
            action = result[organism_id]
            if not np.isfinite((action.dx, action.dy)).all():
                raise ValueError("action direction must be finite")
        return result

    def _move(self, organism: SpatialIndividual, action: MoveAction) -> None:
        state = organism.state
        length = hypot(action.dx, action.dy)
        if length == 0 or organism.energy <= 0:
            return
        direction = atan2(action.dy, action.dx)
        distance = min(self.speed(organism), length)
        if self.config.movement_energy_cost > 0:
            distance = min(distance, max(0.0, organism.energy - organism.metabolic_cost())
                           / self.config.movement_energy_cost)
        nx = float(np.clip(state.x + distance * cos(direction), 0, self.width))
        ny = float(np.clip(state.y + distance * sin(direction), 0, self.height))
        travelled = hypot(nx - state.x, ny - state.y)
        state.x, state.y, state.heading = nx, ny, direction
        self.space.move_agent(organism, (nx, ny))
        organism.spend_energy(travelled * self.config.movement_energy_cost)

    def _wrapped_step(self, actions: Mapping[int, MoveAction] | None = None) -> dict[str, float | int]:
        """Accept external actions while Mesa advances its own time and schedule."""
        if actions is not None:
            unknown = set(actions) - self.organisms.keys()
            if unknown:
                raise ValueError(f"unknown organism IDs: {sorted(unknown)}")
            if any(not np.isfinite((action.dx, action.dy)).all() for action in actions.values()):
                raise ValueError("action direction must be finite")
        self._pending_actions = actions
        try:
            super()._wrapped_step()
        finally:
            self._pending_actions = None
        return self.history[-1]

    def step(self) -> None:
        """Climate, actions, movement, costs, food, fitness/deaths, births, resources, statistics."""
        for key in self.attacks_this_step:
            self.attacks_this_step[key] = 0
        self.environment.step()
        chosen = self._actions(self._pending_actions)
        organisms = self.organisms
        for organism_id in sorted(organisms):
            self._move(organisms[organism_id], chosen[organism_id])
        for organism_id in sorted(organisms):
            organism = organisms[organism_id]
            organism.spend_energy(organism.metabolic_cost())
        self.environment.food_distribution()
        deaths = self.population.remove_dead()
        births = self.population.reproduce()
        self.update_resources()
        self.step_count += 1
        self._record(births, deaths)

    def _record(self, births: int, deaths: dict[str, int]) -> dict[str, float | int]:
        organisms = list(self.organisms.values())
        count = len(organisms)
        evaluated = [o.fitness for o in organisms if o.fitness is not None]
        row: dict[str, float | int] = {
            "TimeStep": self.step_count,
            "PopulationSize": count,
            "AvgEnergy": sum(o.energy for o in organisms) / count if count else 0.0,
            "AvgFitness": float(sum(evaluated) / len(evaluated)) if evaluated else 0.0,
            "Temperature": self.environment.current_params["temperature"],
            "HazardLevel": self.environment.current_params["hazard_level"],
            "BirthCount": births,
            "Deaths_age": deaths["age"],
            "Deaths_fitness": deaths["fitness"],
            "Deaths_threshold": deaths["threshold"],
            "Deaths_energy": deaths["energy"],
            "Deaths_competition": deaths["competition"],
            "Deaths_total": sum(deaths.values()),
            "AttacksConsidered": self.attacks_this_step["considered"],
            "AttacksDeclined": self.attacks_this_step["declined"],
            "AttacksStarted": self.attacks_this_step["started"],
            "AttacksLost": self.attacks_this_step["lost"],
            "AttacksWon": self.attacks_this_step["won"],
            "FoodStolen": self.attacks_this_step["food_stolen"],
            "AttackEnergySpent": self.attacks_this_step["energy_spent"],
            "FoodAvailable": sum(food.stock for food in self.food_sources),
            "FoodSourceCount": len(self.food_sources),
            "FoodSourcesDisappeared": self.environment.disappeared_this_step,
            "FoodSourcesSpawned": self.environment.spawned_this_step,
            "AvgAge": sum(o.age for o in organisms) / count if count else 0.0,
            "AvgLifespan": self._dead_lifetime_sum / self._dead_count if self._dead_count else 0.0,
        }
        for index, label in enumerate(self.genome_labels):
            column = "Avg" + "".join(part.title() for part in label.split("_"))
            row[column] = sum(float(o.genome[index]) for o in organisms) / count if count else 0.0
        self.history.append(row)
        self.datacollector.collect(self)
        return row

    def run(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self.step()

    def export_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
