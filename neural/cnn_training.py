"""Offline imitation of heuristic goals and visible target cells."""

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
from .cnn_policy import CNNPolicy, GOAL_NAMES, IGNORE_TARGET, TargetCNN, encode_observation


def encode_target(observation, map_span: float, nearby_map_span: float) -> tuple[int, int]:
    """Teacher coordinates are labels only; prefer the detailed map when visible."""
    goal = HeuristicPolicy.goal(observation)
    target = (observation.nearest_food if goal in ("forage", "rendezvous")
              else observation.nearest_mate if goal == "mate" else None)
    if target is None:
        return GOAL_NAMES.index(goal), IGNORE_TARGET
    size = observation.local_map.shape[-1]
    channel = 4 if goal == "mate" else 0
    for scale, span, pixels in ((1, nearby_map_span, observation.nearby_map),
                                (0, map_span, observation.local_map)):
        column = int(np.floor((target.x - observation.x + span / 2) * size / span))
        row = int(np.floor((target.y - observation.y + span / 2) * size / span))
        if (0 <= row < size and 0 <= column < size and pixels[3, row, column] > 0
                and pixels[2, row, column] == 0 and round(float(pixels[channel, row, column]) * 255) > 0):
            return GOAL_NAMES.index(goal), scale * size * size + row * size + column
    # A teacher target excluded by sensory rasterization cannot be supervised
    # as if its exact coordinates were visible to the network.
    return GOAL_NAMES.index("explore"), IGNORE_TARGET


class TargetBuffer:
    """Bounded reservoir with a private RNG; observation recording leaves physics unchanged."""

    def __init__(self, max_samples: int, seed: int, map_span: float = 96.0, nearby_map_span: float = 12.0):
        self.max_samples, self.map_span, self.nearby_map_span = max_samples, map_span, nearby_map_span
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.goal_counts = dict.fromkeys(GOAL_NAMES, 0)
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def record(self, observation) -> None:
        self.seen += 1
        goal, target = encode_target(observation, self.map_span, self.nearby_map_span)
        self.goal_counts[GOAL_NAMES[goal]] += 1
        index = len(self.samples) if len(self.samples) < self.max_samples else int(self.rng.integers(self.seen))
        if index >= self.max_samples:
            return
        pixels, state = encode_observation(observation)
        sample = pixels, state, np.asarray(goal, dtype=np.int64), np.asarray(target, dtype=np.int64)
        if index == len(self.samples):
            self.samples.append(sample)
        else:
            self.samples[index] = sample

    def tensors(self) -> tuple[torch.Tensor, ...]:
        if not self.samples:
            raise ValueError("CNN training produced no observations; check initial population and training steps")
        return tuple(torch.from_numpy(np.stack(items)) for items in zip(*self.samples))


class RecordingHeuristicPolicy(HeuristicPolicy):
    def __init__(self, buffer: TargetBuffer):
        self.buffer = buffer

    def choose(self, observation, rng, turn):
        action = super().choose(observation, rng, turn)
        self.buffer.record(observation)
        return action


def collect_target_samples(config: SimConfig, seeds: list[int], steps: int, max_samples: int) -> TargetBuffer:
    buffer = TargetBuffer(max_samples, config.seed, config.spatial.local_map_span, config.spatial.nearby_map_span)
    for seed in seeds:
        world = SpatialWorld.from_sim_config(replace(config, seed=seed))
        world.policy = RecordingHeuristicPolicy(buffer)
        world.run(steps)
    return buffer


def rotate_examples(pixels: torch.Tensor, state: torch.Tensor, targets: torch.Tensor,
                    turns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate maps, heading and flattened target cells together, preserving scale."""
    pixels, state, targets = pixels.clone(), state.clone(), targets.clone()
    size = pixels.shape[-1]
    for turn in (1, 2, 3):
        mask = turns == turn
        pixels[mask] = torch.rot90(pixels[mask], turn, dims=(-2, -1))
        valid = mask & (targets != IGNORE_TARGET)
        scale = targets[valid] // (size * size)
        rows, columns = (targets[valid] % (size * size)) // size, targets[valid] % size
        for _ in range(turn):
            heading = state[mask, 1:3].clone()
            state[mask, 1], state[mask, 2] = -heading[:, 1], heading[:, 0]
            rows, columns = columns, size - 1 - rows
        targets[valid] = scale * size * size + rows * size + columns
    return pixels, state, targets


def target_scores(logits: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Select the food or mate map for each supervised goal."""
    kinds = (goals == GOAL_NAMES.index("mate")).long()
    return logits[torch.arange(len(goals), device=logits.device), kinds].flatten(1)


def target_loss(network: TargetCNN, data, class_weights, target_weight: float):
    pixels, state, goals, targets = data
    goal_logits, target_logits = network(pixels.float() / 255, state)
    goal_loss = nn.functional.cross_entropy(goal_logits, goals, weight=class_weights)
    valid = targets != IGNORE_TARGET
    cell_loss = (nn.functional.cross_entropy(target_scores(target_logits, goals)[valid], targets[valid])
                 if valid.any() else target_logits.sum() * 0)
    return goal_loss + target_weight * cell_loss, goal_logits, target_logits


def evaluate_targets(network: TargetCNN, data, batch_size: int, class_weights, target_weight: float) -> dict:
    total = correct_goals = correct_targets = target_count = 0
    per_goal = {goal: {"samples": 0, "correct_goals": 0, "targets": 0, "correct_targets": 0}
                for goal in GOAL_NAMES}
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(data[2]), batch_size):
            batch = tuple(item[start:start + batch_size] for item in data)
            loss, goal_logits, target_logits = target_loss(network, batch, class_weights, target_weight)
            goals, targets = batch[2:]
            valid = targets != IGNORE_TARGET
            selected = target_scores(target_logits, goals).argmax(dim=1)
            goal_correct = goal_logits.argmax(dim=1) == goals
            cell_correct = (selected == targets) & valid
            total += float(loss) * len(goals)
            correct_goals += int(goal_correct.sum())
            correct_targets += int(cell_correct.sum())
            target_count += int(valid.sum())
            for index, name in enumerate(GOAL_NAMES):
                mask = goals == index
                record = per_goal[name]
                record["samples"] += int(mask.sum())
                record["correct_goals"] += int((goal_correct & mask).sum())
                record["targets"] += int((valid & mask).sum())
                record["correct_targets"] += int((cell_correct & mask).sum())
    by_goal = {name: {"samples": r["samples"], "target_samples": r["targets"],
                       "goal_accuracy": r["correct_goals"] / r["samples"],
                       "target_accuracy": r["correct_targets"] / max(1, r["targets"])}
               for name, r in per_goal.items() if r["samples"]}
    return {"validation_loss": total / len(data[2]), "goal_accuracy": correct_goals / len(data[2]),
            "target_accuracy": correct_targets / max(1, target_count), "target_samples": target_count,
            "validation_by_goal": by_goal}


def train_cnn_policy(config: SimConfig) -> CNNPolicy:
    settings = config.spatial_cnn
    training_seeds = [config.seed + index for index in range(settings.training_runs)]
    validation_seed = config.seed + settings.training_runs
    training = collect_target_samples(config, training_seeds, settings.training_steps, settings.max_samples)
    validation = collect_target_samples(config, [validation_seed], settings.validation_steps,
                                         min(settings.max_samples, 1024))
    train_data, validation_data = training.tensors(), validation.tensors()
    counts = torch.bincount(train_data[2], minlength=len(GOAL_NAMES)).float()
    class_weights = torch.where(counts > 0, len(train_data[2]) / (len(GOAL_NAMES) * counts.clamp_min(1)), 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        network = TargetCNN(config.spatial.nearby_map_span / config.spatial.local_map_span)
        optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        initial = evaluate_targets(network, validation_data, settings.batch_size, class_weights, settings.target_loss_weight)
        best_loss, best_epoch = initial["validation_loss"], 0
        best_weights = {key: value.detach().clone() for key, value in network.state_dict().items()}
        losses = []
        for epoch in range(settings.epochs):
            network.train()
            order = torch.randperm(len(train_data[2]), generator=generator)
            total = 0.0
            for indices in order.split(settings.batch_size):
                pixels, state, goals, targets = (item[indices] for item in train_data)
                pixels, state, targets = rotate_examples(pixels, state, targets,
                                                         torch.randint(4, (len(indices),), generator=generator))
                optimizer.zero_grad()
                loss, _, _ = target_loss(network, (pixels, state, goals, targets), class_weights,
                                          settings.target_loss_weight)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            metrics = evaluate_targets(network, validation_data, settings.batch_size, class_weights,
                                       settings.target_loss_weight)
            losses.append({"epoch": epoch + 1, "training_loss": total / len(train_data[2]),
                           **{key: metrics[key] for key in ("validation_loss", "goal_accuracy", "target_accuracy")}})
            if metrics["validation_loss"] < best_loss:
                best_loss, best_epoch = metrics["validation_loss"], epoch + 1
                best_weights = {key: value.detach().clone() for key, value in network.state_dict().items()}
        network.load_state_dict(best_weights)
    metrics = evaluate_targets(network, validation_data, settings.batch_size, class_weights, settings.target_loss_weight)
    report = {"training_seeds": training_seeds, "validation_seed": validation_seed,
              "training_samples": len(training.samples), "validation_samples": len(validation.samples),
              "initial_validation_loss": initial["validation_loss"], **metrics,
              "training_goals": training.goal_counts, "validation_goals": validation.goal_counts,
              "learning_rate": settings.learning_rate, "batch_size": settings.batch_size,
              "target_loss_weight": settings.target_loss_weight, "rotation_augmentation": True,
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
