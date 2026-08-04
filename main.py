"""Command-line entry point for the core simulation."""

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import SimConfig
from core.enums import RunStatus
from core.model import Model
from vis.history import collect_history
from vis.plots import build_all_figures, save_figures

register_configs()


@hydra.main(version_base=None, config_path="config", config_name="sim_config")
def main(cfg: SimConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    if cfg.view:
        history = collect_history(Model(cfg), cfg.n_steps)
        save_figures(build_all_figures(history))
        return

    model = Model(cfg)
    result = model.run(cfg.n_steps)

    if result.status != RunStatus.COMPLETED:
        print(f"Simulation stopped at step {result.completed_steps}: {result.status.value}")

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
