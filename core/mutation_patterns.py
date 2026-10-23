from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

from core.individual import Individual

if TYPE_CHECKING:
    from .model import Model


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[
        np.ndarray, np.ndarray]:
        pass


class BaselineMutation(MutationStrategy):
    """Classic mutation, using gor baseline."""

    def __init__(self, std: float):
        self._std = std

    def mutate(self, pre_genome: np.ndarray, p1: Individual, p2: Individual, model: "Model") -> tuple[np.ndarray, np.ndarray]:
        noise = model.rng.normal(0, self._std, size=len(pre_genome))
        genome = np.clip(pre_genome + noise, 0.0, 1.0)
        mutations_deltas = genome - pre_genome
        return genome, mutations_deltas
