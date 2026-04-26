from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor


class CatBoostAdvisor:
    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="MultiRMSE",
            random_seed=random_seed,
            verbose=verbose,
        )
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)

        self.model.fit(X, y, sample_weight=weights)
        self._is_trained = True

    def predict(self, feature_vector: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("CatBoostAdvisor is not trained yet.")

        feature_vector = np.asarray(feature_vector, dtype=np.float32)

        if feature_vector.ndim != 1:
            raise ValueError(
                f"feature_vector must be 1-dimensional, but got shape {feature_vector.shape}"
            )

        prediction = self.model.predict(feature_vector.reshape(1, -1))
        return np.asarray(prediction, dtype=np.float32).reshape(-1)

    def save(self, path: str | Path) -> None:
        if not self._is_trained:
            raise RuntimeError("Can't save advisor before training.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        self.model.load_model(str(path))
        self._is_trained = True