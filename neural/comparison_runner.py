import pandas as pd

from config.sim_config import ComparisonConfig
from core.model import Model
from neural.advisor import CatBoostAdvisor
from neural.guided_mutation import GuidedMutation
from neural.trainer import train_advisor
from neural.training_buffer import TrainingBuffer
from vis.history import is_extinct


class ComparisonRunner:
    """Run baseline and ML-guided branches side by side."""

    def __init__(self, config: ComparisonConfig | None = None):
        self.config = config or ComparisonConfig()
        self.baseline_buffer = TrainingBuffer(self.config.simulation.environment)
        self.baseline = Model(self.config.simulation, recorder=self.baseline_buffer)

        advisor = CatBoostAdvisor(iterations=600, verbose=False)
        train_advisor(self._train(), advisor)

        neural_strategy = GuidedMutation(
            advisor=advisor,
            shift_strength=self.config.shift_strength,
        )
        self.neural_branch = Model(self.config.simulation, mutation_strategy=neural_strategy)

    def _train(self) -> TrainingBuffer:
        training_buffer = TrainingBuffer(self.config.simulation.environment)
        training_model = Model(self.config.simulation, recorder=training_buffer)
        training_model.run(self.config.simulation.n_steps)
        return training_buffer

    def step(self) -> None:
        if not is_extinct(self.baseline):
            self.baseline.step()

        if not is_extinct(self.neural_branch):
            if self.neural_branch.step_count != 0 and self.neural_branch.step_count % self.config.retrain_steps == 0:
                self.neural_branch.mutation_strategy.retrain(self.baseline_buffer)

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
