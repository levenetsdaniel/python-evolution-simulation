"""CNN actor, local population context and observation-only PPO value estimates."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .cnn_policy import CNNPolicy, GOAL_NAMES, STATE_SIZE, TargetCNN, encode_observation


LOCAL_STATE_SIZE = 2


def encode_local_state(observation) -> np.ndarray:
    return np.log1p(
        np.asarray([observation.visible_population, observation.visible_mates], dtype=np.float32)
    ) / np.log(21.0)


def encode_batch(observations):
    encoded = [encode_observation(observation) for observation in observations]
    return (
        torch.from_numpy(np.stack([item[0] for item in encoded])),
        torch.from_numpy(np.stack([item[1] for item in encoded])),
        torch.from_numpy(np.stack([encode_local_state(observation) for observation in observations])).float(),
    )


class ActorCritic(nn.Module):
    def __init__(self, actor: TargetCNN):
        super().__init__()
        self.actor = actor
        self.nearby_scale = actor.nearby_scale
        self.context = nn.Sequential(
            nn.Linear(LOCAL_STATE_SIZE, 16), nn.Tanh(), nn.Linear(16, len(GOAL_NAMES))
        )
        # Warm start preserves the pretrained actor's goal logits exactly.
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        self.critic = nn.Sequential(
            nn.Linear(STATE_SIZE + LOCAL_STATE_SIZE + 10, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, pixels, state, local):
        goals, targets = self.actor(pixels, state)
        valid = goals > -1e8
        goals = (goals + self.context(local)).masked_fill(~valid, -1e9)
        value = self.critic(torch.cat((state, local, pixels.mean(dim=(-2, -1))), dim=1)).squeeze(-1)
        return goals, targets, value


def action_statistics(goal_logits, target_logits, goals, cells):
    """Joint likelihood of a goal and its conditional food/mate target cell."""
    goal_distribution = Categorical(logits=goal_logits)
    kind = (goals == GOAL_NAMES.index("mate")).long()
    scores = target_logits[torch.arange(len(goals), device=goals.device), kind].flatten(1)
    target_distribution = Categorical(logits=scores)
    uses_target = (goals == 0) | (goals == 1) | (goals == 2)
    log_probability = goal_distribution.log_prob(goals) + torch.where(
        uses_target, target_distribution.log_prob(cells), 0.0
    )
    entropy = goal_distribution.entropy() + torch.where(uses_target, target_distribution.entropy(), 0.0)
    return log_probability, entropy


def sample_actions(goal_logits, target_logits, generator):
    # Private CPU RNG keeps sampling reproducible without changing world RNGs.
    goals = torch.multinomial(goal_logits.softmax(-1).cpu(), 1, generator=generator).squeeze(-1)
    kind = (goals == GOAL_NAMES.index("mate")).long()
    scores = target_logits[
        torch.arange(len(goals), device=target_logits.device), kind.to(target_logits.device)
    ].flatten(1)
    cells = torch.multinomial(scores.softmax(-1).cpu(), 1, generator=generator).squeeze(-1)
    return goals.to(goal_logits.device), cells.to(goal_logits.device)


class PPOPolicy(CNNPolicy):
    def select_targets(self, observations):
        if not observations:
            return []
        pixels, state, local = encode_batch(observations)
        if pixels.shape[-1] != self.metadata["map_size"]:
            raise ValueError("PPO observation map size does not match the checkpoint")
        with torch.inference_mode():
            goals, targets, _ = self.network(pixels.float() / 255, state, local)
            if not torch.isfinite(goals).all() or not torch.isfinite(targets).all():
                raise ValueError("PPO produced a non-finite prediction")
        return self.decode_targets(goals.argmax(-1).tolist(), targets.flatten(2).argmax(-1).tolist())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.metadata,
                "algorithm": "ppo",
                "ppo_version": 1,
                "local_state_size": LOCAL_STATE_SIZE,
                "state_dict": self.network.state_dict(),
                "training_report": self.training_report,
            },
            path,
        )

    @classmethod
    def load(cls, path, config):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            policy = cls(
                ActorCritic(TargetCNN(config.spatial.nearby_map_span / config.spatial.local_map_span)), config
            )
        if (
            checkpoint.get("metadata") != policy.metadata
            or checkpoint.get("ppo_version") != 1
            or checkpoint.get("local_state_size") != LOCAL_STATE_SIZE
            or checkpoint.get("algorithm") != "ppo"
        ):
            raise ValueError("PPO checkpoint is incompatible with the observation or map geometry")
        policy.network.load_state_dict(checkpoint["state_dict"])
        policy.training_report = checkpoint.get("training_report", {})
        return policy
