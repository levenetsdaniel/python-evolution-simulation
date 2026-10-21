"""Shared convolutional movement policy; world physics stays in core."""

from __future__ import annotations

from math import cos, hypot, sin
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.sim_config import SimConfig
from core.spatial_policy import LOCAL_MAP_CHANNELS, MoveAction, SpatialObservation


STATE_SIZE = 10


def encode_observation(observation: SpatialObservation) -> tuple[np.ndarray, np.ndarray]:
    """Use sensory pixels and own state, without target coordinates or global state."""
    pixels = observation.local_map
    if pixels is None or pixels.ndim != 3 or pixels.shape[0] != len(LOCAL_MAP_CHANNELS):
        raise ValueError("CNN requires a four-channel local map")
    state = np.asarray([observation.satiation, cos(observation.heading), sin(observation.heading),
                        *observation.genome], dtype=np.float32)
    if state.shape != (STATE_SIZE,) or not np.isfinite(state).all():
        raise ValueError("CNN requires finite own state and seven genome traits")
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("CNN local map values must be finite and in [0, 1]")
    return np.rint(pixels * 255).astype(np.uint8), state


def normalize_action(action: MoveAction, speed: float) -> np.ndarray:
    """Express teacher displacement as a fraction of the agent's movement budget."""
    length = hypot(action.dx, action.dy)
    if length == 0 or speed <= 0:
        return np.zeros(2, dtype=np.float32)
    return (np.asarray([action.dx, action.dy], dtype=np.float32) / max(length, float(speed))).astype(np.float32)


class MovementCNN(nn.Module):
    """Retain spatial layout when combining visual features with own state."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(len(LOCAL_MAP_CHANNELS), 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)), nn.Flatten(),
        )
        self.nearby = nn.Sequential(nn.Conv2d(len(LOCAL_MAP_CHANNELS), 8, 3, padding=1),
                                    nn.ReLU(), nn.Flatten())
        self.movement = nn.Sequential(nn.Linear(32 * 8 * 8 + 8 * 8 * 8 + STATE_SIZE, 128), nn.ReLU(),
                                      nn.Linear(128, 2), nn.Tanh())

    def forward(self, pixels: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        row, column = pixels.shape[2] // 2, pixels.shape[3] // 2
        nearby = self.nearby(pixels[:, :, row - 4:row + 4, column - 4:column + 4])
        vector = self.movement(torch.cat((self.visual(pixels), nearby, state), dim=1))
        return vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1)


class CNNPolicy:
    """Deterministic CPU inference for one shared network across all organisms."""

    def __init__(self, network: MovementCNN, config: SimConfig,
                 training_report: dict | None = None):
        self.network = network.cpu().eval()
        self.metadata = {
            "version": 1,
            "channels": list(LOCAL_MAP_CHANNELS),
            "map_size": config.spatial.local_map_size,
            "map_span": config.spatial.local_map_span,
            "genome_labels": list(config.population.genome_labels),
        }
        self.training_report = training_report or {}

    def choose(self, observation: SpatialObservation, rng: np.random.Generator,
               turn: float) -> MoveAction:
        return self.choose_batch([observation], rng, turn)[0]

    def choose_batch(self, observations: list[SpatialObservation], rng: np.random.Generator,
                     turn: float) -> list[MoveAction]:
        if not observations:
            return []
        encoded = [encode_observation(observation) for observation in observations]
        pixels = np.stack([item[0] for item in encoded])
        if pixels.shape[2:] != (self.metadata["map_size"], self.metadata["map_size"]):
            raise ValueError("CNN observation map size does not match the checkpoint")
        state = np.stack([item[1] for item in encoded])
        with torch.inference_mode():
            vectors = self.network(torch.from_numpy(pixels).float() / 255,
                                   torch.from_numpy(state)).numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("CNN produced a non-finite movement")
        return [MoveAction(float(vector[0] * observation.speed),
                           float(vector[1] * observation.speed))
                for vector, observation in zip(vectors, observations)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": self.metadata, "state_dict": self.network.state_dict(),
                    "training_report": self.training_report}, path)

    @classmethod
    def load(cls, path: str | Path, config: SimConfig) -> CNNPolicy:
        if not Path(path).is_file():
            raise FileNotFoundError(f"CNN model file does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            policy = cls(MovementCNN(), config)
        if checkpoint.get("metadata") != policy.metadata:
            raise ValueError("CNN checkpoint is incompatible with map geometry, channels or genome order")
        policy.network.load_state_dict(checkpoint["state_dict"])
        policy.training_report = checkpoint.get("training_report", {})
        return policy
