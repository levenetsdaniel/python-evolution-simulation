"""
Command-line entry point for running the simulation with baseline and neural guided line.

Initializes configuration from CLI, runs two models.
"""

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import SimConfig
from core.engine import Engine
from vis.dashboard import build_dashboard

register_configs()


@hydra.main(version_base=None, config_path="config", config_name="sim_config")
def main(cfg: SimConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    if cfg.view:
        build_dashboard(cfg)
        return

    Engine(cfg).run(cfg.n_steps)


if __name__ == "__main__":
    main()
