import pandas as pd

from config.sim_config import ComparisonConfig, validate_sim_config
from core.model import Model
from neural.advisor_training import load_pretrained_advisor
from neural.guided_mutation import GuidedMutation
from vis.history import is_extinct


class ComparisonRunner:
    """Run baseline and ML-guided branches side by side."""

    def __init__(self, config: ComparisonConfig | None = None):
        self.config = config or ComparisonConfig()
        validate_sim_config(self.config.simulation)
        if self.config.simulation.mode != "classic":
            raise ValueError("comparison requires simulation.mode=classic")
        advisor = load_pretrained_advisor(self.config.advisor_path, self.config.simulation)
        self.baseline = Model(self.config.simulation)

        neural_strategy = GuidedMutation(
            advisor=advisor,
            shift_strength=self.config.shift_strength,
        )
        self.neural_branch = Model(self.config.simulation, mutation_strategy=neural_strategy)

    def step(self) -> None:
        if not is_extinct(self.baseline):
            self.baseline.step()

        if not is_extinct(self.neural_branch):
            self.neural_branch.step()

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            if is_extinct(self.baseline) and is_extinct(self.neural_branch):
                break
            self.step()

    def history(self) -> pd.DataFrame:
        baseline_df = self.baseline.datacollector.get_model_vars_dataframe()
        neural_df = self.neural_branch.datacollector.get_model_vars_dataframe()
        baseline_df["branch"] = "baseline"
        neural_df["branch"] = "neural"
        return pd.concat([baseline_df, neural_df], ignore_index=True)
