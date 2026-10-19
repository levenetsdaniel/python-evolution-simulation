"""Continuous 2D evolution using Mesa agents and a headless spatial model."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from typing import Mapping

import numpy as np
import mesa
from mesa.space import ContinuousSpace

from config.sim_config import (IndividualConfig, PopulationConfig, SimConfig,
                               SpatialConfig, validate_sim_config, validate_spatial_config)
from .enums import DeathCause, Gender
from .genetics import crossover_and_mutate
from .mutation_patterns import BaselineMutation
from .spatial_individual import SpatialIndividual, SpatialState
from .spatial_policy import (FoodObservation, HeuristicPolicy, MoveAction,
                             SpatialObservation, SpatialObservationProvider)


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


class SpatialWorld(mesa.Model):
    """Owns spatial state and executes simultaneous actions in a fixed step order.

    Organisms are processed by ascending ID. Food requests are simultaneous:
    each source divides insufficient stock in proportion to requested bites.
    """

    def __init__(self, config: SpatialConfig | None = None, seed: int = 42,
                 genome_labels: list[str] | None = None,
                 individual_config: IndividualConfig | None = None):
        self.config = config or SpatialConfig()
        validate_spatial_config(self.config)
        super().__init__(rng=seed)
        self.individual_config = replace(individual_config or IndividualConfig(),
                                         energy_initial=self.config.initial_energy,
                                         energy_capacity=self.config.max_energy)
        validate_sim_config(SimConfig(individual=self.individual_config, spatial=self.config))
        self.mutation_strategy = BaselineMutation(self.config.mutation_std)
        self.genome_labels = tuple(genome_labels or PopulationConfig().genome_labels)
        if set(self.genome_labels) != set(PopulationConfig().genome_labels) or len(self.genome_labels) != 7:
            raise ValueError("spatial genome_labels must contain the seven classic traits")
        self.width = self.config.width
        self.height = self.config.height
        # Mesa treats max bounds as exclusive; nextafter includes our closed wall.
        self.space = ContinuousSpace(float(np.nextafter(self.width, np.inf)),
                                     float(np.nextafter(self.height, np.inf)), torus=False)
        self.food_sources: list[FoodSource] = []
        self.step_count = 0
        self._pending_actions: Mapping[int, MoveAction] | None = None
        self._dead_count = 0
        self._dead_lifetime_sum = 0
        self.history: list[dict[str, float | int]] = []
        report_names = ("TimeStep", "PopulationSize", "AvgEnergy", "BirthCount",
                        "Deaths_age", "Deaths_energy", "Deaths_total",
                        "FoodAvailable", "AvgAge", "AvgLifespan")
        report_names += tuple("Avg" + "".join(part.title() for part in trait.split("_"))
                              for trait in self.genome_labels)
        self.datacollector = mesa.DataCollector(model_reporters={
            name: (lambda model, column=name: model.history[-1][column]) for name in report_names
        })
        self.observation_provider = SpatialObservationProvider()
        self.policy = HeuristicPolicy()
        self._generate_food()
        for _ in range(self.config.initial_population):
            self.add_organism(
                self.rng.random(len(self.genome_labels)),
                float(self.rng.uniform(0, self.width)),
                float(self.rng.uniform(0, self.height)),
            )
        self._record(0, 0, 0)

    @property
    def organisms(self) -> dict[int, SpatialIndividual]:
        """ID lookup backed by Mesa's registered agents."""
        return {agent.unique_id: agent for agent in sorted(self.agents, key=lambda item: item.unique_id)}

    def _generate_food(self) -> None:
        cfg = self.config
        for _ in range(cfg.cluster_count):
            cx, cy = self.rng.uniform(0, self.width), self.rng.uniform(0, self.height)
            for _ in range(cfg.sources_per_cluster):
                x = float(np.clip(self.rng.normal(cx, cfg.cluster_spread), 0, self.width))
                y = float(np.clip(self.rng.normal(cy, cfg.cluster_spread), 0, self.height))
                self.food_sources.append(FoodSource(x, y, cfg.food_initial, cfg.food_capacity,
                                                    cfg.food_regeneration, cfg.food_radius))

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
        candidates = ((hypot(food.x - x, food.y - y), index, food)
                      for index, food in enumerate(self.food_sources) if food.stock > 0)
        return next((food for distance, _, food in sorted(candidates) if distance <= radius), None)

    def update_resources(self) -> None:
        for food in self.food_sources:
            food.restore()

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

    def _feed(self) -> None:
        requests: dict[int, list[SpatialIndividual]] = {}
        organisms = self.organisms
        for organism_id in sorted(organisms):
            organism = organisms[organism_id]
            state = organism.state
            reachable = [(hypot(food.x - state.x, food.y - state.y), index)
                         for index, food in enumerate(self.food_sources)
                         if food.stock > 0 and hypot(food.x - state.x, food.y - state.y) <= food.radius]
            if reachable and organism.energy < self.config.max_energy:
                _, index = min(reachable)
                requests.setdefault(index, []).append(organism)
        for index in sorted(requests):
            food = self.food_sources[index]
            participants = requests[index]
            share = min(self.config.food_bite, food.stock / len(participants))
            for organism in participants:
                organism.gain_energy(share * self.config.food_energy)
            food.stock = max(0.0, food.stock - share * len(participants))

    def _reproduce(self) -> int:
        cfg = self.config
        births = 0
        organisms = self.organisms
        parents = [organisms[key] for key in sorted(organisms)]
        parent_ids = set(organisms)
        for p1 in parents:
            if len(self.agents) >= cfg.max_population:
                break
            if p1.gender != Gender.FEMALE or p1.age < cfg.min_reproduction_age or p1.energy < cfg.reproduction_energy:
                continue
            mates = sorted((p2 for p2 in self.space.get_neighbors(p1.pos, cfg.mating_radius)
                            if p2.unique_id in parent_ids and p2 is not p1
                            and p2.gender == Gender.MALE
                            and p2.age >= cfg.min_reproduction_age and p2.energy > 0),
                           key=lambda agent: agent.unique_id)
            if not mates or self.rng.random() >= cfg.reproduction_rate * p1.satiation:
                continue
            p2 = mates[int(self.rng.integers(len(mates)))]
            genome, _, _ = crossover_and_mutate(p1, p2, self)
            direction = float(self.rng.uniform(-pi, pi))
            offset = float(self.rng.uniform(0, cfg.offspring_offset))
            x = float(np.clip(p1.state.x + offset * cos(direction), 0, self.width))
            y = float(np.clip(p1.state.y + offset * sin(direction), 0, self.height))
            self.add_organism(genome, x, y, energy=cfg.offspring_energy,
                              parent_ids=(p1.unique_id, p2.unique_id))
            p1.spend_energy(cfg.offspring_energy)
            births += 1
        return births

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
        """Actions, movement, costs, food, deaths, births, regrowth, statistics."""
        chosen = self._actions(self._pending_actions)
        organisms = self.organisms
        for organism_id in sorted(organisms):
            self._move(organisms[organism_id], chosen[organism_id])
        for organism_id in sorted(organisms):
            organism = organisms[organism_id]
            organism.spend_energy(organism.metabolic_cost())
            organism.age += 1
        self._feed()
        for organism_id in sorted(organisms):
            organism = organisms[organism_id]
            if organism.energy <= 0:
                organism.is_alive = False
                organism.death_cause = DeathCause.ENERGY
            elif self.rng.random() < organism.age_death_probability():
                organism.is_alive = False
                organism.death_cause = DeathCause.AGE
        dead = [organism_id for organism_id in sorted(organisms) if not organisms[organism_id].is_alive]
        deaths_age = sum(organisms[key].death_cause == DeathCause.AGE for key in dead)
        deaths_energy = len(dead) - deaths_age
        for organism_id in dead:
            self._dead_lifetime_sum += organisms[organism_id].age
            self._dead_count += 1
            self.remove_organism(organism_id)
        births = self._reproduce()
        self.update_resources()
        self.step_count += 1
        self._record(births, deaths_age, deaths_energy)

    def _record(self, births: int, deaths_age: int, deaths_energy: int) -> dict[str, float | int]:
        organisms = list(self.organisms.values())
        count = len(organisms)
        row: dict[str, float | int] = {
            "TimeStep": self.step_count,
            "PopulationSize": count,
            "AvgEnergy": sum(o.energy for o in organisms) / count if count else 0.0,
            "BirthCount": births,
            "Deaths_age": deaths_age,
            "Deaths_energy": deaths_energy,
            "Deaths_total": deaths_age + deaths_energy,
            "FoodAvailable": sum(food.stock for food in self.food_sources),
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
