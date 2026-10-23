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
    foraging: bool = False


class SpatialIndividual(Individual):
    """Baseline individual with coordinates and heading for spatial behavior."""

    def __init__(self, model: SpatialWorld, genome: np.ndarray,
                 genome_labels: tuple[str, ...], state: SpatialState,
                 parent_ids: tuple[int, int] | None = None, birth_step: int = 0):
        super().__init__(model, genome, list(genome_labels), parent_ids, birth_step)
        self.state = state

    @property
    def local_map(self) -> np.ndarray:
        """Build a current observation; maps are state snapshots, not inherited traits."""
        return self.model.observation_provider.local_map(self.model, self)

    @property
    def foraging(self) -> bool:
        """Hunger hysteresis; reading observations never mutates behavior memory."""
        if self.satiation <= self.model.config.foraging_threshold:
            return True
        if self.satiation >= self.model.config.satiation_threshold:
            return False
        return self.state.foraging
