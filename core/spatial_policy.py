"""Observation and action interfaces for spatial agents."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import TYPE_CHECKING

import numpy as np

from .enums import Gender

if TYPE_CHECKING:
    from .spatial import SpatialWorld
    from .spatial_individual import SpatialIndividual


LOCAL_MAP_CHANNELS = ("food", "organisms", "boundary", "visibility", "mates")


@dataclass(frozen=True)
class MoveAction:
    """Desired displacement; speed and available energy limit its magnitude."""

    dx: float
    dy: float


@dataclass(frozen=True)
class SpatialTarget:
    """A selected goal in coordinates relative to the observing organism."""

    kind: str
    dx: float = 0.0
    dy: float = 0.0
    cell_size: float = 0.0


class TargetController:
    """Translate a perceived target into a bounded displacement."""

    def choose(self, target: SpatialTarget, observation: SpatialObservation,
               rng: np.random.Generator, turn: float) -> MoveAction:
        if target.kind == "rest" or observation.speed <= 0 or observation.energy <= 0:
            return MoveAction(0, 0)
        if target.kind == "explore":
            heading = observation.heading + rng.uniform(-turn, turn)
            return MoveAction(cos(heading) * observation.speed, sin(heading) * observation.speed)
        distance = hypot(target.dx, target.dy)
        # A point anywhere in the selected cell may be half its diagonal from
        # its center. Keep this uncertainty inside the physical mating radius.
        uncertainty = target.cell_size / 2 ** 0.5
        stop_distance = (max(0.0, observation.mating_distance - uncertainty) * 0.8
                         if target.kind == "mate" else uncertainty)
        if distance <= stop_distance + 1e-9:
            return MoveAction(0, 0)
        travel = min(observation.speed, distance - stop_distance)
        return MoveAction(target.dx * travel / distance, target.dy * travel / distance)


@dataclass(frozen=True)
class FoodObservation:
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class MateObservation:
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class SpatialObservation:
    x: float
    y: float
    heading: float
    energy: float
    nearest_food: FoodObservation | None
    local_map: np.ndarray | None = None
    satiation: float = 0.0
    genome: tuple[float, ...] = ()
    speed: float = 1.0
    nearest_mate: MateObservation | None = None
    ready_to_reproduce: bool = False
    foraging: bool = False
    is_female: bool = False
    mating_distance: float = 0.0
    nearby_map: np.ndarray | None = None


class SpatialObservationProvider:
    """Projects model state into information available to a policy."""

    def local_map(self, world: SpatialWorld, individual: SpatialIndividual, *,
                  span: float | None = None) -> np.ndarray:
        """Return a detached float32 [channel, row, column] map in world axes.

        Rows increase along +y, columns along +x. The agent is at the center
        of the fixed physical window, independent of heading and perception.
        Visibility includes cell centers within perception; boundary marks
        visible cells outside the world. Point sources use their stock fraction
        (maximum for shared cells), other live agents use binary occupancy.
        The final channel marks ready opposite-sex partners. Objects outside
        perception are excluded even if they share a visible cell.
        """
        size = world.config.local_map_size
        span = world.config.local_map_span if span is None else span
        x, y = individual.state.x, individual.state.y
        radius = world.perception(individual)
        offsets = (np.arange(size) + 0.5) * (span / size) - span / 2
        visible = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius ** 2
        outside = ((x + offsets[None, :] < 0) | (x + offsets[None, :] > world.width)
                   | (y + offsets[:, None] < 0) | (y + offsets[:, None] > world.height))
        result = np.zeros((len(LOCAL_MAP_CHANNELS), size, size), dtype=np.float32)
        result[2] = outside & visible
        result[3] = visible

        def deposit(channel: int, points: list[tuple[float, float, float]]) -> None:
            if not points:
                return
            values = np.asarray(points, dtype=float)
            dx, dy = values[:, 0] - x, values[:, 1] - y
            columns = np.floor((dx + span / 2) * size / span).astype(int)
            rows = np.floor((dy + span / 2) * size / span).astype(int)
            valid = ((dx * dx + dy * dy <= radius * radius)
                     & (columns >= 0) & (columns < size) & (rows >= 0) & (rows < size))
            rows, columns = rows[valid], columns[valid]
            weights = np.clip(values[valid, 2], 0, 1)
            weights *= visible[rows, columns] & ~outside[rows, columns]
            np.maximum.at(result[channel], (rows, columns), weights)

        deposit(0, [(food.x, food.y, food.stock / food.capacity)
                    for food in world.food_sources if food.stock > 0])
        deposit(1, [(agent.state.x, agent.state.y, 1.0)
                    for agent in world.agents if agent is not individual and agent.is_alive])
        deposit(4, [(agent.state.x, agent.state.y, 1.0)
                    for agent in world.agents if agent is not individual
                    and agent.gender != individual.gender and world.population.can_reproduce(agent)])
        result.setflags(write=False)
        return result

    def observe(self, world: SpatialWorld, individual: SpatialIndividual) -> SpatialObservation:
        state = individual.state
        nearest = world.nearest_food(state.x, state.y, world.perception(individual))
        food = None
        if nearest is not None:
            food = FoodObservation(nearest.x, nearest.y, hypot(nearest.x - state.x, nearest.y - state.y))
        mates = [(hypot(agent.state.x - state.x, agent.state.y - state.y), agent.unique_id, agent)
                 for agent in world.space.get_neighbors(individual.pos, world.perception(individual))
                 if agent is not individual and agent.gender != individual.gender
                 and world.population.can_reproduce(agent)]
        mate = None
        if mates:
            distance, _, partner = min(mates, key=lambda item: (item[0], item[1]))
            mate = MateObservation(partner.state.x, partner.state.y, distance)
        return SpatialObservation(state.x, state.y, state.heading, individual.energy, food,
                                  self.local_map(world, individual), individual.satiation,
                                  tuple(float(gene) for gene in individual.genome), world.speed(individual),
                                  mate, world.population.can_reproduce(individual), individual.foraging,
                                  individual.gender == Gender.FEMALE, world.mating_distance,
                                  self.local_map(world, individual, span=world.config.nearby_map_span))


class HeuristicPolicy:
    """Refuel when hungry, approach a mate when ready, otherwise explore or rest."""

    @staticmethod
    def goal(observation: SpatialObservation) -> str:
        if observation.foraging:
            return "forage" if observation.nearest_food is not None else "explore"
        if observation.ready_to_reproduce:
            if observation.nearest_mate is not None:
                return "mate"
            return "rendezvous" if observation.nearest_food is not None else "explore"
        return "rest"

    @staticmethod
    def mating_action(observation: SpatialObservation) -> MoveAction:
        """Approach the visible partner with a margin inside the interaction radius."""
        mate = observation.nearest_mate
        if mate.distance <= observation.mating_distance:
            return MoveAction(0, 0)
        distance = max(0.0, mate.distance - observation.mating_distance * 0.8)
        return MoveAction((mate.x - observation.x) * distance / mate.distance,
                          (mate.y - observation.y) * distance / mate.distance)

    def choose(self, observation: SpatialObservation, rng: np.random.Generator,
               turn: float) -> MoveAction:
        goal = self.goal(observation)
        if goal in ("forage", "rendezvous"):
            food = observation.nearest_food
            return MoveAction(food.x - observation.x, food.y - observation.y)
        if goal == "mate":
            return self.mating_action(observation)
        if goal == "rest":
            return MoveAction(0, 0)
        heading = observation.heading + rng.uniform(-turn, turn)
        return MoveAction(cos(heading) * observation.speed, sin(heading) * observation.speed)
