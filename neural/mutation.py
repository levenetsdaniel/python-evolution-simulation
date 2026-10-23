import numpy as np

from neural.advisor import CatBoostAdvisor
from neural.training_buffer import TrainingBuffer


def _build_features(
    pre_mutation_genome: np.ndarray,
    env_params,
    env_delta,
    pop_mean_genome: np.ndarray,
    pop_mean_fitness: float,
    parent_mean_fitness: float,
) -> np.ndarray:
    """
    Build the advisor input vector from the offspring genome, environment,
    environment change, and population statistics.

    The feature order must stay consistent with the order used during advisor
    training.
    """
    temp_range = TrainingBuffer.MAX_TEMPERATURE - TrainingBuffer.MIN_TEMPERATURE

    if isinstance(env_params, dict):
        temp_norm = (env_params["temperature"] - TrainingBuffer.MIN_TEMPERATURE) / temp_range
        food_norm = env_params["food_availability"] / TrainingBuffer.MAX_FOOD
        env_vec = np.asarray(
            [temp_norm, food_norm, env_params["hazard_level"]],
            dtype=np.float32,
        )
    else:
        env_vec = np.asarray(env_params, dtype=np.float32)

    if isinstance(env_delta, dict):
        delta_temp_norm = env_delta.get("temperature", 0.0) / temp_range
        delta_food_norm = env_delta.get("food_availability", 0.0) / TrainingBuffer.MAX_FOOD
        env_delta_vec = np.asarray(
            [delta_temp_norm, delta_food_norm, env_delta.get("hazard_level", 0.0)],
            dtype=np.float32,
        )
    else:
        env_delta_vec = np.asarray(env_delta, dtype=np.float32)

    return np.concatenate([
        np.asarray(pre_mutation_genome, dtype=np.float32),
        env_vec,
        env_delta_vec,
        np.asarray(pop_mean_genome, dtype=np.float32),
        np.asarray([pop_mean_fitness, parent_mean_fitness], dtype=np.float32),
    ])


def _fallback_random_mutate(
    pre_mutation_genome: np.ndarray,
    mutation_std: float = 0.12,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a baseline-style random mutation when the advisor is not trained.

    The mutation is generated from a normal distribution and then clipped
    to keep genome values inside the valid [0.0, 1.0] range.
    """
    pre = np.asarray(pre_mutation_genome, dtype=np.float32)

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=mutation_std,
        size=pre.shape,
    ).astype(np.float32)

    mutated = np.clip(pre + noise, 0.0, 1.0).astype(np.float32)
    deltas = mutated - pre

    return mutated, deltas


def neural_mutate(
    pre_mutation_genome: np.ndarray,
    advisor: CatBoostAdvisor,
    env_params,
    env_delta,
    pop_mean_genome: np.ndarray,
    pop_mean_fitness: float,
    parent_mean_fitness: float,
    shift_strength: float = 0.3,
    rng=None,
    mutation_std: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a target genome with the advisor and move the current genome
    toward it by the given shift strength.

    Returns the mutated genome and the applied mutation deltas.

    If the advisor is not trained, a baseline-style random fallback mutation
    is applied instead.
    """
    pre = np.asarray(pre_mutation_genome, dtype=np.float32)

    if not advisor.is_trained:
        return _fallback_random_mutate(
            pre,
            mutation_std=mutation_std,
            rng=rng,
        )

    features = _build_features(
        pre_mutation_genome=pre,
        env_params=env_params,
        env_delta=env_delta,
        pop_mean_genome=pop_mean_genome,
        pop_mean_fitness=pop_mean_fitness,
        parent_mean_fitness=parent_mean_fitness,
    )

    target = advisor.predict(features).astype(np.float32)
    target = np.clip(target, 0.0, 1.0)

    if target.shape != pre.shape:
        raise ValueError(
            f"advisor target shape {target.shape} != genome shape {pre.shape}"
        )

    mutated = pre + shift_strength * (target - pre)
    mutated = np.clip(mutated, 0.0, 1.0).astype(np.float32)
    deltas = mutated - pre

    return mutated, deltas
