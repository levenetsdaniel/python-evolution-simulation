"""Imitation learning from heuristic spatial trajectories, with held-out worlds."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config.sim_config import SimConfig, validate_sim_config
from core.spatial import SpatialWorld
from core.spatial_policy import HeuristicPolicy

from .cnn_policy import CNNPolicy, MovementCNN, encode_observation, normalize_action


class MovementBuffer:
    """Bounded reservoir of examples; its RNG is independent of world physics."""

    def __init__(self, max_samples: int, seed: int):
        self.max_samples = max_samples
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def record(self, observation, action) -> None:
        self.seen += 1
        index = len(self.samples) if len(self.samples) < self.max_samples else int(self.rng.integers(self.seen))
        if index >= self.max_samples:
            return
        pixels, state = encode_observation(observation)
        sample = pixels, state, normalize_action(action, observation.speed)
        if index == len(self.samples):
            self.samples.append(sample)
        else:
            self.samples[index] = sample

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise ValueError("CNN training produced no observations; check initial population and training steps")
        return tuple(torch.from_numpy(np.stack(items)) for items in zip(*self.samples))


class RecordingHeuristicPolicy(HeuristicPolicy):
    def __init__(self, buffer: MovementBuffer):
        self.buffer = buffer

    def choose(self, observation, rng, turn):
        action = super().choose(observation, rng, turn)
        self.buffer.record(observation, action)
        return action


def collect_movement_samples(config: SimConfig, seeds: list[int], steps: int,
                             max_samples: int) -> MovementBuffer:
    buffer = MovementBuffer(max_samples, config.seed)
    for seed in seeds:
        world = SpatialWorld.from_sim_config(replace(config, seed=seed))
        world.policy = RecordingHeuristicPolicy(buffer)
        world.run(steps)
    return buffer


def _evaluate(network: MovementCNN, data, batch_size: int) -> float:
    pixels, state, targets = data
    total = 0.0
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            end = start + batch_size
            predictions = network(pixels[start:end].float() / 255, state[start:end])
            total += float(torch.sum((predictions - targets[start:end]) ** 2))
    return total / targets.numel()


def train_cnn_policy(config: SimConfig) -> CNNPolicy:
    settings = config.spatial_cnn
    training_seeds = [config.seed + index for index in range(settings.training_runs)]
    validation_seed = config.seed + settings.training_runs
    training = collect_movement_samples(config, training_seeds, settings.training_steps, settings.max_samples)
    validation = collect_movement_samples(config, [validation_seed], settings.validation_steps,
                                         min(settings.max_samples, 1024))
    train_data, validation_data = training.tensors(), validation.tensors()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        network = MovementCNN()
        optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        initial_loss = _evaluate(network, validation_data, settings.batch_size)
        best_loss, best_epoch = initial_loss, 0
        best_weights = {key: value.detach().clone() for key, value in network.state_dict().items()}
        losses = []
        pixels, state, targets = train_data
        for epoch in range(settings.epochs):
            network.train()
            order = torch.randperm(len(targets), generator=generator)
            total = 0.0
            for indices in order.split(settings.batch_size):
                optimizer.zero_grad()
                prediction = network(pixels[indices].float() / 255, state[indices])
                loss = nn.functional.mse_loss(prediction, targets[indices])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            validation_loss = _evaluate(network, validation_data, settings.batch_size)
            losses.append({"epoch": epoch + 1, "training_mse": total / len(targets),
                           "validation_mse": validation_loss})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch + 1
                best_weights = {key: value.detach().clone() for key, value in network.state_dict().items()}
        network.load_state_dict(best_weights)
    report = {"training_seeds": training_seeds, "validation_seed": validation_seed,
              "training_samples": len(training.samples), "validation_samples": len(validation.samples),
              "initial_validation_mse": initial_loss, "validation_mse": best_loss,
              "stationary_validation_mse": float(validation_data[2].square().mean()),
              "best_epoch": best_epoch, "epochs": losses}
    return CNNPolicy(network, config, report)


def build_cnn_policy(config: SimConfig) -> CNNPolicy:
    validate_sim_config(config)
    if config.mode != "spatial" or not config.spatial_cnn.enabled:
        raise ValueError("CNN policy requires mode=spatial and spatial_cnn.enabled=true")
    torch.set_num_threads(config.spatial_cnn.threads)
    if config.spatial_cnn.model_path:
        return CNNPolicy.load(config.spatial_cnn.model_path, config)
    policy = train_cnn_policy(config)
    if config.spatial_cnn.save_path:
        path = Path(config.spatial_cnn.save_path)
        policy.save(path)
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(policy.training_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return policy
