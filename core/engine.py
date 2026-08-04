import pandas as pd
from config.sim_config import SimConfig
from core.model import Model
from core.mutation_patterns import NeuralMutation
from core.training_buffer import TrainingBuffer
from neural.advisor import CatBoostAdvisor
from neural.trainer import train_advisor
from vis.history import is_extinct


class Engine:
    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

        self.baseline = Model(self.config)

        advisor = CatBoostAdvisor(iterations=600, verbose=False)
        train_advisor(self._train(), advisor)

        neural_strategy = NeuralMutation(
            advisor=advisor,
            shift_strength=self.config.shift_strength,
        )
        self.neuralline = Model(self.config, mutation_strategy=neural_strategy)

    def _train(self) -> TrainingBuffer:
        training_model = Model(self.config)
        training_model.run(self.config.n_steps)
        return training_model.training_buffer

    def step(self):
        if not is_extinct(self.baseline):
            self.baseline.step()

        if not is_extinct(self.neuralline):
            if self.neuralline.step_count != 0 and self.neuralline.step_count % self.config.retrain_steps == 0:
                self.neuralline.mutation_strategy.retrain(self.baseline.training_buffer)

            self.neuralline.step()

    def run(self, n_steps: int):
        for _ in range(n_steps):
            if is_extinct(self.baseline) and is_extinct(self.neuralline):
                break
            self.step()

    def history(self):
        baseline_df = self.baseline.datacollector.get_model_vars_dataframe()
        neuralline_df = self.neuralline.datacollector.get_model_vars_dataframe()
        baseline_df["branch"] = "baseline"
        neuralline_df["branch"] = "neural"
        return pd.concat([baseline_df, neuralline_df], ignore_index=True)
