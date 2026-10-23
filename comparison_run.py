"""Command-line entry point for the ML-guided comparison pipeline."""

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import ComparisonConfig
from neural.comparison_runner import ComparisonRunner
from vis.dashboard import build_dashboard

register_configs()


@hydra.main(version_base=None, config_path="config", config_name="compare_config")
def main(cfg: ComparisonConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    if cfg.view:
        build_dashboard(cfg)
        return

    ComparisonRunner(cfg).run(cfg.simulation.n_steps)


if __name__ == "__main__":
    main()
