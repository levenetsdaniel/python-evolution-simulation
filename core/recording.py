from __future__ import annotations

from typing import Protocol

import numpy as np


class EvolutionRecorder(Protocol):
    """Optional sink for offspring and fitness events emitted by the simulation."""

    def record_birth(
        self,
        *,
        child_id: int,
        pre_mutation_genome: np.ndarray,
        mutation_deltas: np.ndarray,
        p1_fitness: float,
        p2_fitness: float,
        env_params: dict[str, float],
        env_delta: dict[str, float],
        pop_mean_genome: np.ndarray,
        pop_mean_fitness: float,
    ) -> None:
        ...

    def record_fitness(self, child_id: int, child_fitness: float) -> None:
        ...
