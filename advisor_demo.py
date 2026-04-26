import tempfile
from pathlib import Path

import numpy as np

from config.sim_config import SimConfig
from core.model import Model
from neural.advisor import CatBoostAdvisor
from neural.mutation import neural_mutate
from neural.trainer import train_advisor


def run_demo(n_steps: int = 30, population_size: int = 60, seed: int = 0) -> None:
    """
    The demo runs a baseline simulation, trains the advisor, saves and loads
    the model, and verifies that neural_mutate applies the expected 30% shift.
    """
    config = SimConfig(seed=seed, n_steps=n_steps)
    config.population.initial_size = population_size

    model = Model(config)
    model.run(n_steps)

    buffer = model.training_buffer
    if not buffer.samples:
        raise RuntimeError("Training buffer is empty.")

    advisor = CatBoostAdvisor(iterations=100, verbose=False)
    X, _, _ = train_advisor(buffer, advisor)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "advisor.cbm"
        advisor.save(model_path)

        loaded = CatBoostAdvisor()
        loaded.load(model_path)

        sample = buffer.samples[0]
        pre = np.asarray(sample.pre_mutation_genome, dtype=np.float32)

        prediction = loaded.predict(X[0]).astype(np.float32)
        prediction = np.clip(prediction, 0.0, 1.0)

        mutated, deltas = neural_mutate(
            pre_mutation_genome=pre,
            advisor=loaded,
            env_params=sample.env_params,
            env_delta=sample.env_delta,
            pop_mean_genome=sample.population_mean_genome,
            pop_mean_fitness=sample.population_mean_fitness,
            parent_mean_fitness=sample.parent_mean_fitness,
            shift_strength=0.3,
        )

        expected_mutated = pre + 0.3 * (prediction - pre)
        expected_mutated = np.clip(expected_mutated, 0.0, 1.0)

        assert loaded.is_trained
        assert mutated.shape == pre.shape
        assert np.all((mutated >= 0.0) & (mutated <= 1.0))
        assert not np.allclose(mutated, pre), "Genome did not change."
        assert np.allclose(mutated, expected_mutated), "30% shift rule is broken."
        assert np.allclose(deltas, mutated - pre), "deltas do not match mutated - pre."

    print("demo OK")
    print("samples:", len(buffer.samples))
    print("genome changed:", not np.allclose(mutated, pre))
    print("30% shift correct:", np.allclose(mutated, expected_mutated))
    print("deltas correct:", np.allclose(deltas, mutated - pre))


if __name__ == "__main__":
    run_demo()