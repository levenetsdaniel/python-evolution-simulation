"""Train or load a CatBoost advisor for spatial mutation guidance."""

from __future__ import annotations

from config.sim_config import SimConfig, validate_sim_config
from core.spatial import SpatialWorld

from .advisor import CatBoostAdvisor
from .guided_mutation import GuidedMutation
from .trainer import train_advisor
from .training_buffer import TrainingBuffer


class GuidedSpatialWorld(SpatialWorld):
    """Collect guided offspring outcomes and periodically extend the advisor."""

    @classmethod
    def from_guidance_config(cls, config: SimConfig, advisor: CatBoostAdvisor) -> GuidedSpatialWorld:
        food_capacity = (config.spatial.cluster_count * config.spatial.sources_per_cluster
                         * config.spatial.food_capacity)
        buffer = TrainingBuffer(config.environment, max_food=max(1.0, food_capacity))
        strategy = GuidedMutation(advisor, shift_strength=config.spatial_guidance.shift_strength)
        world = cls.from_sim_config(config, mutation_strategy=strategy, recorder=buffer)
        world.advisor = advisor
        world.retrain_steps = config.spatial_guidance.retrain_steps
        world.save_path = config.spatial_guidance.save_path
        world.retraining_steps: list[int] = []
        return world

    def step(self) -> None:
        super().step()
        if self.step_count % self.retrain_steps != 0 or not self.recorder.samples:
            return
        self.mutation_strategy.retrain(self.recorder, continue_training=True)
        self.retraining_steps.append(self.step_count)
        self.recorder.samples.clear()
        if self.save_path:
            self.advisor.save(self.save_path)


def build_guided_spatial_world(config: SimConfig) -> GuidedSpatialWorld:
    """Prepare the advisor on a separate baseline run, then start a guided world.

    The training and guided worlds use the same seed and configuration. A
    supplied advisor_path skips training. A new advisor can be saved for reuse.
    """
    validate_sim_config(config)
    if config.mode != "spatial" or not config.spatial_guidance.enabled:
        raise ValueError("spatial guidance requires mode=spatial and spatial_guidance.enabled=true")

    guidance = config.spatial_guidance
    advisor = CatBoostAdvisor(iterations=guidance.iterations,
                              random_seed=config.seed, verbose=False)
    if guidance.advisor_path:
        advisor.load(guidance.advisor_path)
    else:
        food_capacity = (config.spatial.cluster_count * config.spatial.sources_per_cluster
                         * config.spatial.food_capacity)
        buffer = TrainingBuffer(config.environment, max_food=max(1.0, food_capacity))
        baseline = SpatialWorld.from_sim_config(config, recorder=buffer)
        baseline.run(guidance.training_steps)
        if not buffer.samples:
            raise ValueError("spatial training produced no evaluated offspring; increase "
                             "spatial_guidance.training_steps or check reproduction settings")
        train_advisor(buffer, advisor)

    if guidance.save_path:
        advisor.save(guidance.save_path)

    return GuidedSpatialWorld.from_guidance_config(config, advisor)
