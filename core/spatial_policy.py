"""Observation and action interfaces for spatial agents."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .spatial import SpatialWorld
    from .spatial_individual import SpatialIndividual


LOCAL_MAP_CHANNELS = ("food", "organisms", "boundary", "visibility")


@dataclass(frozen=True)
class MoveAction:
    """Desired displacement; speed and available energy limit its magnitude."""

    dx: float
    dy: float


@dataclass(frozen=True)
class FoodObservation:
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


class SpatialObservationProvider:
    """Projects model state into information available to a policy."""

    def local_map(self, world: SpatialWorld, individual: SpatialIndividual) -> np.ndarray:
        """Return a detached float32 [channel, row, column] map in world axes.

        Rows increase along +y, columns along +x. The agent is at the center
        of the fixed physical window, independent of heading and perception.
        Visibility includes cell centers within perception; boundary marks
        visible cells outside the world. Point sources use their stock fraction
        (maximum for shared cells), other live agents use binary occupancy.
        Objects outside perception are excluded even if they share a visible cell.
        """
        size = world.config.local_map_size
        span = world.config.local_map_span
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
        result.setflags(write=False)
        return result

    def observe(self, world: SpatialWorld, individual: SpatialIndividual) -> SpatialObservation:
        state = individual.state
        nearest = world.nearest_food(state.x, state.y, world.perception(individual))
        food = None
        if nearest is not None:
            food = FoodObservation(nearest.x, nearest.y, hypot(nearest.x - state.x, nearest.y - state.y))
        return SpatialObservation(state.x, state.y, state.heading, individual.energy, food,
                                  self.local_map(world, individual), individual.satiation,
                                  tuple(float(gene) for gene in individual.genome), world.speed(individual))


class HeuristicPolicy:
    """Seek visible food; otherwise turn a bounded random angle to explore."""

    def choose(self, observation: SpatialObservation, rng: np.random.Generator,
               turn: float) -> MoveAction:
        if observation.nearest_food is not None:
            food = observation.nearest_food
            return MoveAction(food.x - observation.x, food.y - observation.y)
        heading = observation.heading + rng.uniform(-turn, turn)
        return MoveAction(cos(heading), sin(heading))
