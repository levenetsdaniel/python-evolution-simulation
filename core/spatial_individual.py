"""Mesa agent and mutable state for the spatial simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .individual import Individual

if TYPE_CHECKING:
    from .spatial import SpatialWorld


@dataclass
class SpatialState:
    x: float
    y: float
    heading: float


class SpatialIndividual(Individual):
    """Baseline individual with coordinates and heading for spatial behavior."""

    def __init__(self, model: SpatialWorld, genome: np.ndarray,
                 genome_labels: tuple[str, ...], state: SpatialState,
                 parent_ids: tuple[int, int] | None = None, birth_step: int = 0):
        super().__init__(model, genome, list(genome_labels), parent_ids, birth_step)
        self.state = state
