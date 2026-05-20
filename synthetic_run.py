"""
Command-line entry point for running the simulation.

Initializes configuration from CLI, runs the model,
and optionally prints collected statistics.
"""

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import SimConfig
from core.model import Model

register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: SimConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    model = Model(cfg)
    model.run(cfg.n_steps)

    model_df = model.datacollector.get_model_vars_dataframe()

    if cfg.model_info:
        print("Model stats:")
        print(model_df.tail(cfg.steps_info))

    if cfg.population_info:
        print("Population stats:")
        print(model_df.describe())

    if cfg.individual_info:
        agent_df = model.datacollector.get_agent_vars_dataframe()
        print("\nAgent stats:")
        print(agent_df.describe())


if __name__ == "__main__":
    main()
