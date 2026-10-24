"""Offline shared-policy PPO with per-organism trajectories and behavioral tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from math import isfinite
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from config.sim_config import PPOTrainingConfig, validate_sim_config
from core.spatial_policy import TargetController
from .cnn_policy import CNNPolicy, TargetCNN
from .cnn_training import select_training_device
from .ppo_policy import ActorCritic, PPOPolicy, action_statistics, encode_batch, sample_actions
from .training_scenarios import TRAINING_SCENARIOS, build_training_world


def validate_ppo_config(config):
    sim = config.simulation
    validate_sim_config(replace(sim, spatial_cnn=replace(sim.spatial_cnn, enabled=True)))
    if sim.mode != "spatial" or sim.spatial_guidance.enabled:
        raise ValueError("PPO requires spatial mode with CatBoost disabled")
    for name in (
        "threads",
        "iterations",
        "rollout_steps",
        "episode_steps",
        "validation_steps",
        "validation_runs",
        "validation_interval",
        "epochs",
        "batch_size",
        "local_population_target",
        "offspring_survival_steps",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("training_scenarios", "validation_scenarios"):
        values = getattr(config, name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or value not in TRAINING_SCENARIOS for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"{name} must contain unique behavioral scenario names")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(config, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "entropy_coef",
        "value_coef",
        "survival_reward",
        "energy_reward",
        "birth_reward",
        "low_density_bonus",
        "offspring_survival_reward",
        "death_penalty",
    ):
        if not isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0 < config.gamma <= 1 or not 0 <= config.gae_lambda <= 1 or not 0 < config.clip_ratio < 1:
        raise ValueError("Invalid PPO discount, GAE lambda or clip ratio")
    curriculum = config.curriculum
    for name in ("group_count", "group_size"):
        value = getattr(curriculum, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"curriculum.{name} must be a positive integer")
    if (
        curriculum.group_size < 2
        or curriculum.group_count * curriculum.group_size > sim.spatial.max_population
    ):
        raise ValueError("Curriculum needs partners and must fit spatial.max_population")
    if not isfinite(curriculum.partner_distance) or curriculum.partner_distance <= 0:
        raise ValueError("curriculum.partner_distance must be finite and positive")
    if not 0 <= curriculum.depleted_stock_fraction <= 1:
        raise ValueError("curriculum.depleted_stock_fraction must be in [0, 1]")
    if not str(config.save_path).strip():
        raise ValueError("save_path is required")
    if (
        config.initial_model_path
        and Path(config.initial_model_path).resolve() == Path(config.save_path).resolve()
    ):
        raise ValueError("save_path must differ from initial_model_path")


def generalized_advantages(rewards, values, dones, bootstrap, gamma, gae_lambda):
    """One organism, one world, one rollout; never cross a birth/death boundary."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    following = float(bootstrap)
    accumulated = 0.0
    for index in reversed(range(len(rewards))):
        continuation = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * following * continuation - values[index]
        accumulated = delta + gamma * gae_lambda * continuation * accumulated
        advantages[index] = accumulated
        following = values[index]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def transition_rewards(before, observations, world, config, matured):
    """Credit actual births to both parents; density is measured before the birth."""
    current = world.organisms
    observations_by_id = dict(zip(before, observations))
    transferred = dict.fromkeys(before, 0.0)
    birth_bonus = dict.fromkeys(before, 0.0)
    births = [child for key, child in current.items() if key not in before and child.parent_ids]
    for child in births:
        mother, father = child.parent_ids
        if mother in transferred:
            transferred[mother] += child.energy
        for parent in (mother, father):
            if parent in birth_bonus:
                population = observations_by_id[parent].visible_population
                scarcity = max(0.0, 1.0 - population / config.local_population_target)
                birth_bonus[parent] += config.birth_reward * 0.5 * (1 + config.low_density_bonus * scarcity)
    rewards = {}
    for key, agent in before.items():
        alive = key in current
        energy_delta = agent.energy + transferred[key] - observations_by_id[key].energy
        reward = config.survival_reward if alive else -config.death_penalty
        reward += config.energy_reward * energy_delta / world.config.max_energy + birth_bonus[key]
        # The offspring receives a one-time reward for reaching a viable age.
        if alive and agent.parent_ids and key not in matured and agent.age >= config.offspring_survival_steps:
            reward += config.offspring_survival_reward
            matured.add(key)
        rewards[key] = float(reward)
    return rewards


def collect_rollout(world, network, decoder, config, generator, matured):
    device = next(network.parameters()).device
    controller = TargetController()
    trajectories = {}
    network.eval()
    for _ in range(config.rollout_steps):
        before = world.organisms
        if not before or world.step_count >= config.episode_steps:
            break
        observations = [world.observation_provider.observe(world, agent) for agent in before.values()]
        pixels, state, local = encode_batch(observations)
        with torch.no_grad():
            goal_logits, target_logits, values = network(
                pixels.to(device).float() / 255, state.to(device), local.to(device)
            )
            goals, cells = sample_actions(goal_logits, target_logits, generator)
            log_probabilities, _ = action_statistics(goal_logits, target_logits, goals, cells)
        goals, cells = goals.cpu(), cells.cpu()
        # Transfer vectors once instead of synchronizing the accelerator per organism.
        log_probabilities, values = log_probabilities.cpu(), values.cpu()
        targets = decoder.decode_targets(goals.tolist(), [[int(cell), int(cell)] for cell in cells])
        actions = {
            key: controller.choose(target, observation, world.rng, world.config.exploration_turn)
            for key, target, observation in zip(before, targets, observations)
        }
        world.step(actions)
        rewards = transition_rewards(before, observations, world, config, matured)
        current = world.organisms
        for index, key in enumerate(before):
            trajectories.setdefault(key, []).append(
                {
                    "pixels": pixels[index],
                    "state": state[index],
                    "local": local[index],
                    "goal": goals[index],
                    "cell": cells[index],
                    "log_probability": float(log_probabilities[index]),
                    "value": float(values[index]),
                    "reward": rewards[key],
                    "done": key not in current,
                }
            )
    if not trajectories:
        return []
    bootstrap = {}
    alive = world.organisms
    if alive:
        observations = [world.observation_provider.observe(world, agent) for agent in alive.values()]
        pixels, state, local = encode_batch(observations)
        with torch.no_grad():
            _, _, values = network(pixels.to(device).float() / 255, state.to(device), local.to(device))
        bootstrap = dict(zip(alive, values.cpu().tolist()))
    samples = []
    for key, trajectory in trajectories.items():
        advantages, returns = generalized_advantages(
            [item["reward"] for item in trajectory],
            [item["value"] for item in trajectory],
            [item["done"] for item in trajectory],
            bootstrap.get(key, 0.0),
            config.gamma,
            config.gae_lambda,
        )
        for item, advantage, value_return in zip(trajectory, advantages, returns):
            samples.append({**item, "advantage": float(advantage), "return": float(value_return)})
    return samples


def ppo_loss(log_probabilities, old_log_probabilities, advantages, values, returns, entropy, config):
    ratios = (log_probabilities - old_log_probabilities).exp()
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()
    return loss, policy_loss, value_loss


def update_policy(network, optimizer, samples, config, generator):
    device = next(network.parameters()).device
    data = {
        name: torch.stack([item[name] for item in samples])
        for name in ("pixels", "state", "local", "goal", "cell")
    }
    for name in ("log_probability", "advantage", "return"):
        data[name] = torch.tensor([item[name] for item in samples], dtype=torch.float32)
    advantages = data["advantage"]
    data["advantage"] = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    losses = []
    network.train()
    for _ in range(config.epochs):
        order = torch.randperm(len(samples), generator=generator)
        for indices in order.split(config.batch_size):
            batch = {name: value[indices].to(device) for name, value in data.items()}
            goals, targets, values = network(batch["pixels"].float() / 255, batch["state"], batch["local"])
            log_probability, entropy = action_statistics(goals, targets, batch["goal"], batch["cell"])
            loss, policy_loss, value_loss = ppo_loss(
                log_probability,
                batch["log_probability"],
                batch["advantage"],
                values,
                batch["return"],
                entropy,
                config,
            )
            if not torch.isfinite(loss):
                raise ValueError("PPO produced a non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append((float(loss.detach()), float(policy_loss.detach()), float(value_loss.detach())))
    return {
        "loss": float(np.mean([item[0] for item in losses])),
        "policy_loss": float(np.mean([item[1] for item in losses])),
        "value_loss": float(np.mean([item[2] for item in losses])),
    }


def evaluate_policy(network, config, seeds):
    policy = PPOPolicy(deepcopy(network).cpu(), config.simulation)
    rows = []
    for scenario in config.validation_scenarios:
        for seed in seeds:
            world = build_training_world(config.simulation, scenario, seed, config.curriculum)
            world.policy = policy
            initial = len(world.agents)
            matured = set()
            for _ in range(config.validation_steps):
                if not world.agents:
                    break
                world.step()
                matured.update(
                    agent.unique_id
                    for agent in world.agents
                    if agent.parent_ids and agent.age >= config.offspring_survival_steps
                )
            final = len(world.agents)
            area = sum(row["PopulationSize"] for row in world.history) / (config.validation_steps + 1)
            score = (final + area + len(matured)) / max(1, initial)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "initial_population": initial,
                    "final_population": final,
                    "births": sum(row["BirthCount"] for row in world.history),
                    "surviving_offspring": len(matured),
                    "steps": world.step_count,
                    "score": float(score),
                }
            )
    return {"score": float(np.mean([row["score"] for row in rows])), "runs": rows}


def train_ppo_policy(config: PPOTrainingConfig):
    validate_ppo_config(config)
    device = select_training_device(config.device)
    torch.set_num_threads(config.threads)
    print(f"PPO training device: {device}", flush=True)
    initial_hash = (
        hashlib.sha256(Path(config.initial_model_path).read_bytes()).hexdigest()
        if config.initial_model_path
        else None
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.simulation.seed)
        decoder = (
            CNNPolicy.load(config.initial_model_path, config.simulation)
            if config.initial_model_path
            else CNNPolicy(
                TargetCNN(
                    config.simulation.spatial.nearby_map_span / config.simulation.spatial.local_map_span
                ),
                config.simulation,
            )
        )
        network = (
            deepcopy(decoder.network)
            if isinstance(decoder, PPOPolicy)
            else ActorCritic(deepcopy(decoder.network))
        ).to(device)
    original = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    generator = torch.Generator().manual_seed(config.simulation.seed)
    validation_seeds = [config.simulation.seed + 1_000_000 + index for index in range(config.validation_runs)]
    training_seeds = []
    next_seed = config.simulation.seed

    def create_world(scenario):
        nonlocal next_seed
        while next_seed in validation_seeds:
            next_seed += 1
        seed = next_seed
        next_seed += 1
        training_seeds.append(seed)
        return build_training_world(config.simulation, scenario, seed, config.curriculum)

    worlds = {scenario: create_world(scenario) for scenario in config.training_scenarios}
    matured = {scenario: set() for scenario in worlds}
    initial = evaluate_policy(network, config, validation_seeds)
    best, best_iteration = initial, 0
    best_weights = original
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    history = []
    decisions = steps = 0
    for iteration in range(1, config.iterations + 1):
        samples = []
        for scenario in config.training_scenarios:
            world = worlds[scenario]
            if not world.agents or world.step_count >= config.episode_steps:
                world = worlds[scenario] = create_world(scenario)
                matured[scenario] = set()
            before_steps = world.step_count
            samples.extend(collect_rollout(world, network, decoder, config, generator, matured[scenario]))
            steps += world.step_count - before_steps
        if not samples:
            raise ValueError("PPO collected no organism decisions")
        metrics = update_policy(network, optimizer, samples, config, generator)
        decisions += len(samples)
        row = {
            "iteration": iteration,
            "decisions": len(samples),
            "mean_reward": float(np.mean([s["reward"] for s in samples])),
            **metrics,
        }
        if iteration % config.validation_interval == 0 or iteration == config.iterations:
            validation = evaluate_policy(network, config, validation_seeds)
            row["validation_score"] = validation["score"]
            if validation["score"] > best["score"]:
                best, best_iteration = validation, iteration
                best_weights = {
                    name: value.detach().cpu().clone() for name, value in network.state_dict().items()
                }
        history.append(row)
        print(
            f"PPO iteration {iteration}/{config.iterations}: decisions={len(samples)}, loss={metrics['loss']:.4f}",
            flush=True,
        )
    actor_updated = any(
        not torch.equal(value.detach().cpu(), original[name])
        for name, value in network.state_dict().items()
        if name.startswith("actor.")
    )
    network.load_state_dict(best_weights)
    report = {
        "algorithm": "ppo",
        "training_device": str(device),
        "updates": config.iterations,
        "world_steps": steps,
        "decisions": decisions,
        "training_seeds": training_seeds,
        "validation_seeds": validation_seeds,
        "training_scenarios": config.training_scenarios,
        "initial_validation": initial,
        "best_validation": best,
        "best_iteration": best_iteration,
        "actor_updated_during_training": actor_updated,
        "history": history,
        "initial_checkpoint_sha256": initial_hash,
        "config": asdict(config),
    }
    if config.initial_model_path:
        assert hashlib.sha256(Path(config.initial_model_path).read_bytes()).hexdigest() == initial_hash
    policy = PPOPolicy(network.cpu(), config.simulation, report)
    policy.save(config.save_path)
    Path(config.save_path + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return policy


@hydra.main(version_base=None, config_path="../config", config_name="ppo_config")
def main(cfg: PPOTrainingConfig):
    config = OmegaConf.to_object(cfg)
    policy = train_ppo_policy(config)
    print(
        f"PPO policy saved: {config.save_path}; best validation: {policy.training_report['best_validation']['score']:.3f}"
    )


if __name__ == "__main__":
    from config.registry import register_configs

    register_configs()
    main()
