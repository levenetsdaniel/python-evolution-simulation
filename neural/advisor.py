from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor


class CatBoostAdvisor:
    """
    A small wrapper around CatBoost for genome target prediction.

    The model is trained on simulation features and predicts a target genome
    vector for one offspring candidate.
    """

    def __init__(
        self,
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Create the underlying CatBoost regressor and initialize training state.

        Parameters
        - iterations : int
            Number of boosting iterations.
        - learning_rate : float
            Step size used during boosting.
        - depth : int
            Tree depth for CatBoost.
        - random_seed : int
            Seed for reproducible training.
        - verbose : bool
            Whether CatBoost should print training logs.
        """
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
        """
        Indicate whether the advisor is ready for inference.

        Returns
        - bool
            True if the model was trained or loaded from disk, otherwise False.
        """
        return self._is_trained

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """
        Train the advisor on feature vectors and target genomes.

        Parameters
        - X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        - y : np.ndarray
            Target genome matrix of shape (n_samples, genome_size).
        - weights : np.ndarray | None
            Optional sample weights used during training.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)

        self.model.fit(X, y, sample_weight=weights)
        self._is_trained = True

    def predict(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Predict a target genome for a single feature vector.

        Parameters
        - feature_vector : np.ndarray
            One input sample as a 1D feature vector.

        Returns
        - np.ndarray
            Predicted target genome as a 1D array.

        Raises
        - RuntimeError
            If the model has not been trained or loaded yet.
        - ValueError
            If the input is not one-dimensional.
        """
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
        """
        Save the trained model to disk.

        Parameters
        - path : str | Path
            Output file path for the saved CatBoost model.

        Raises
        - RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save advisor before training.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        """
        Load a trained model from disk and mark the advisor as ready.

        Parameters
        - path : str | Path
            Path to a saved CatBoost model file.

        Raises
        - FileNotFoundError
            If the model file does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")

        self.model.load_model(str(path))
        self._is_trained = True