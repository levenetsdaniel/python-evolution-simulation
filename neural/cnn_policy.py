"""Visual goal and target selection, with a separate geometric controller."""

from __future__ import annotations

from math import cos, sin
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.sim_config import SimConfig
from core.spatial_policy import (LOCAL_MAP_CHANNELS, MoveAction, SpatialObservation,
                                 SpatialTarget, TargetController)


STATE_SIZE = 13
GOAL_NAMES = ("forage", "mate", "rendezvous", "explore", "rest")
IGNORE_TARGET = -100


def encode_observation(observation: SpatialObservation) -> tuple[np.ndarray, np.ndarray]:
    """Use sensory pixels and own state, without object or absolute coordinates."""
    maps = (observation.local_map, observation.nearby_map)
    for sensory_map in maps:
        if (sensory_map is None or sensory_map.ndim != 3
                or sensory_map.shape[0] != len(LOCAL_MAP_CHANNELS)):
            raise ValueError("CNN requires five-channel local and nearby maps")
    if maps[0].shape != maps[1].shape:
        raise ValueError("CNN local and nearby map shapes must match")
    pixels = np.concatenate(maps, axis=0)
    state = np.asarray([observation.satiation, cos(observation.heading), sin(observation.heading),
                        *observation.genome, observation.is_female,
                        observation.ready_to_reproduce, observation.foraging], dtype=np.float32)
    if state.shape != (STATE_SIZE,) or not np.isfinite(state).all():
        raise ValueError("CNN requires finite own state and seven genome traits")
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
        raise ValueError("CNN map values must be finite and in [0, 1]")
    return np.rint(pixels * 255).astype(np.uint8), state


class TargetCNN(nn.Module):
    """Classify goals and score occupied cells without discarding their positions.

    The target branches share weights across scales. Their coordinate channels
    describe pixel centers in the observer's frame; they contain no world state.
    """

    def __init__(self, nearby_scale: float = 0.125):
        super().__init__()
        self.nearby_scale = nearby_scale
        self.visual = nn.Sequential(nn.Conv2d(2 * len(LOCAL_MAP_CHANNELS), 16, 3, padding=1),
                                    nn.ReLU(), nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())
        self.goal = nn.Sequential(nn.Linear(16 + STATE_SIZE, 64), nn.ReLU(),
                                  nn.Linear(64, len(GOAL_NAMES)))
        self.state_embedding = nn.Sequential(nn.Linear(STATE_SIZE, 8), nn.ReLU())
        self.targets = nn.Sequential(nn.Conv2d(len(LOCAL_MAP_CHANNELS) + 4 + 8, 16, 1),
                                     nn.ReLU(), nn.Conv2d(16, 2, 1))

    @staticmethod
    def candidate_mask(pixels: torch.Tensor) -> torch.Tensor:
        """[batch, kind(food/mate), scale, row, column] visible occupied cells."""
        channels = len(LOCAL_MAP_CHANNELS)
        masks = []
        for offset in (0, channels):
            visible = (pixels[:, offset + 3] > 0) & (pixels[:, offset + 2] == 0)
            masks.append(torch.stack(((pixels[:, offset] > 0) & visible,
                                      (pixels[:, offset + 4] > 0) & visible), dim=1))
        return torch.stack(masks, dim=2)

    def forward(self, pixels: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        goal_logits = self.goal(torch.cat((self.visual(pixels), state), dim=1))
        batch, _, size, _ = pixels.shape
        offsets = (torch.arange(size, device=pixels.device, dtype=pixels.dtype) + 0.5) / size - 0.5
        y, x = torch.meshgrid(offsets, offsets, indexing="ij")
        own_state = self.state_embedding(state)[:, :, None, None].expand(-1, -1, size, size)
        scores = []
        channels = len(LOCAL_MAP_CHANNELS)
        for offset, scale in ((0, 1.0), (channels, self.nearby_scale)):
            geometry = torch.stack((x * scale, y * scale,
                                    torch.sqrt(x.square() + y.square()) * scale,
                                    torch.full_like(x, scale)))
            features = torch.cat((pixels[:, offset:offset + channels],
                                  geometry[None].expand(batch, -1, -1, -1), own_state), dim=1)
            scores.append(self.targets(features))
        target_logits = torch.stack(scores, dim=2)
        candidates = self.candidate_mask(pixels)
        available = candidates.flatten(2).any(dim=2)
        valid_goals = torch.stack((available[:, 0], available[:, 1] & (state[:, -2] > 0),
                                   available[:, 0], torch.ones_like(available[:, 0]),
                                   torch.ones_like(available[:, 0])), dim=1)
        return (goal_logits.masked_fill(~valid_goals, -1e9),
                target_logits.masked_fill(~candidates, -1e9))


class CNNPolicy:
    """One visual target selector shared by all organisms; no world lookup at inference."""

    def __init__(self, network: TargetCNN, config: SimConfig, training_report: dict | None = None):
        self.network = network.cpu().eval()
        self.controller = TargetController()
        self.metadata = {
            "version": 4,
            "state_size": STATE_SIZE,
            "teacher": "visual_targets",
            "goals": list(GOAL_NAMES),
            "channels": list(LOCAL_MAP_CHANNELS),
            "map_size": config.spatial.local_map_size,
            "map_span": config.spatial.local_map_span,
            "nearby_map_span": config.spatial.nearby_map_span,
            "mating_radius": config.spatial.mating_radius,
            "genome_labels": list(config.population.genome_labels),
        }
        expected_scale = config.spatial.nearby_map_span / config.spatial.local_map_span
        if network.nearby_scale != expected_scale:
            raise ValueError("CNN target geometry does not match the configured map scales")
        self.training_report = training_report or {}

    def select_targets(self, observations: list[SpatialObservation]) -> list[SpatialTarget]:
        if not observations:
            return []
        encoded = [encode_observation(observation) for observation in observations]
        pixels = np.stack([item[0] for item in encoded])
        size = self.metadata["map_size"]
        if pixels.shape[2:] != (size, size):
            raise ValueError("CNN observation map size does not match the checkpoint")
        state = np.stack([item[1] for item in encoded])
        with torch.inference_mode():
            tensor = torch.from_numpy(pixels).float() / 255
            goal_logits, target_logits = self.network(tensor, torch.from_numpy(state))
            if not torch.isfinite(goal_logits).all() or not torch.isfinite(target_logits).all():
                raise ValueError("CNN produced a non-finite target prediction")
            goals = goal_logits.argmax(dim=1).tolist()
            cells = target_logits.flatten(2).argmax(dim=2).tolist()
        return self.decode_targets(goals, cells)

    def decode_targets(self, goals: list[int], cells: list[list[int]]) -> list[SpatialTarget]:
        """Decode goal/cell decisions for imitation and reinforcement policies."""
        size = self.metadata["map_size"]
        selected = []
        for index, goal in enumerate(goals):
            kind = GOAL_NAMES[goal]
            if kind in ("rest", "explore"):
                selected.append(SpatialTarget(kind))
                continue
            cell = cells[index][int(kind == "mate")]
            scale, cell = divmod(cell, size * size)
            row, column = divmod(cell, size)
            span = self.metadata["nearby_map_span" if scale else "map_span"]
            cell_size = span / size
            selected.append(SpatialTarget(kind, (column + 0.5) * cell_size - span / 2,
                                           (row + 0.5) * cell_size - span / 2, cell_size))
        return selected

    def choose(self, observation: SpatialObservation, rng: np.random.Generator, turn: float) -> MoveAction:
        return self.choose_batch([observation], rng, turn)[0]

    def choose_batch(self, observations: list[SpatialObservation], rng: np.random.Generator,
                     turn: float) -> list[MoveAction]:
        targets = self.select_targets(observations)
        return [self.controller.choose(target, observation, rng, turn)
                for target, observation in zip(targets, observations)]

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
        if checkpoint.get("algorithm") == "ppo":
            from .ppo_policy import PPOPolicy
            return PPOPolicy.load(path, config)
        with torch.random.fork_rng(devices=[]):
            policy = cls(TargetCNN(config.spatial.nearby_map_span / config.spatial.local_map_span), config)
        if checkpoint.get("metadata") != policy.metadata:
            raise ValueError("CNN checkpoint is incompatible with visual target selection or map geometry; "
                             "retrain the policy for the current target format")
        policy.network.load_state_dict(checkpoint["state_dict"])
        policy.training_report = checkpoint.get("training_report", {})
        return policy
