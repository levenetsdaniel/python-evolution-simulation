"""Attach a pretrained CatBoost advisor to the spatial mutation strategy."""

from config.sim_config import SimConfig, validate_sim_config
from core.spatial import SpatialWorld

from .advisor_training import load_pretrained_advisor
from .guided_mutation import GuidedMutation


def build_guided_spatial_world(config: SimConfig) -> SpatialWorld:
    """Load fixed advisor weights; simulation does not collect training samples."""
    validate_sim_config(config)
    if config.mode != "spatial" or not config.spatial_guidance.enabled:
        raise ValueError("spatial guidance requires mode=spatial and spatial_guidance.enabled=true")

    advisor = load_pretrained_advisor(config.spatial_guidance.advisor_path, config)
    strategy = GuidedMutation(advisor, shift_strength=config.spatial_guidance.shift_strength)
    return SpatialWorld.from_sim_config(config, mutation_strategy=strategy)
