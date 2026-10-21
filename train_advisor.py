"""Train and save CatBoost separately from simulation and comparison runs."""

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import AdvisorTrainingConfig
from neural.advisor_training import train_advisor_model

register_configs()


@hydra.main(version_base=None, config_path="config", config_name="advisor_config")
def main(cfg: AdvisorTrainingConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    _, report = train_advisor_model(cfg)
    print(f"Advisor saved: {report['model_path']}")
    print(report)


if __name__ == "__main__":
    main()
