"""Observation and action interfaces for spatial agents."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .spatial import SpatialWorld
    from .spatial_individual import SpatialIndividual


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


class SpatialObservationProvider:
    """Projects model state into information available to a policy."""

    def observe(self, world: SpatialWorld, individual: SpatialIndividual) -> SpatialObservation:
        state = individual.state
        nearest = world.nearest_food(state.x, state.y, world.perception(individual))
        food = None
        if nearest is not None:
            food = FoodObservation(nearest.x, nearest.y, hypot(nearest.x - state.x, nearest.y - state.y))
        return SpatialObservation(state.x, state.y, state.heading, individual.energy, food)


class HeuristicPolicy:
    """Seek visible food; otherwise turn a bounded random angle to explore."""

    def choose(self, observation: SpatialObservation, rng: np.random.Generator,
               turn: float) -> MoveAction:
        if observation.nearest_food is not None:
            food = observation.nearest_food
            return MoveAction(food.x - observation.x, food.y - observation.y)
        heading = observation.heading + rng.uniform(-turn, turn)
        return MoveAction(cos(heading), sin(heading))
