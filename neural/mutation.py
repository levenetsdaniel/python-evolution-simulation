import numpy as np
from core.training_buffer import TrainingBuffer
from neural.advisor import CatBoostAdvisor


def _build_features(
    pre_mutation_genome: np.ndarray,
    env_params: dict,
    env_delta: dict,
    pop_mean_genome: np.ndarray,
    pop_mean_fitness: float,
    parent_mean_fitness: float,
) -> np.ndarray:
    temp_range = TrainingBuffer.MAX_TEMPERATURE - TrainingBuffer.MIN_TEMPERATURE

    temp_norm = (env_params["temperature"] - TrainingBuffer.MIN_TEMPERATURE) / temp_range
    food_norm = env_params["food_availability"] / TrainingBuffer.MAX_FOOD
    env_vec = [
        temp_norm,
        food_norm,
        env_params["hazard_level"],
    ]

    delta_temp_norm = env_delta.get("temperature", 0.0) / temp_range
    delta_food_norm = env_delta.get("food_availability", 0.0) / TrainingBuffer.MAX_FOOD
    env_delta_vec = [
        delta_temp_norm,
        delta_food_norm,
        env_delta.get("hazard_level", 0.0),
    ]

    return np.concatenate([
        np.asarray(pre_mutation_genome, dtype=float),
        np.asarray(env_vec, dtype=float),
        np.asarray(env_delta_vec, dtype=float),
        np.asarray(pop_mean_genome, dtype=float),
        np.asarray([pop_mean_fitness, parent_mean_fitness], dtype=float),
    ])


def neural_mutate(
    pre_mutation_genome: np.ndarray,
    advisor: CatBoostAdvisor,
    env_params: dict,
    env_delta: dict,
    pop_mean_genome: np.ndarray,
    pop_mean_fitness: float,
    parent_mean_fitness: float,
    shift_strength: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    pre = np.asarray(pre_mutation_genome, dtype=float)

    if not advisor.is_trained:
        return pre.copy(), np.zeros_like(pre)

    features = _build_features(
        pre_mutation_genome=pre,
        env_params=env_params,
        env_delta=env_delta,
        pop_mean_genome=pop_mean_genome,
        pop_mean_fitness=pop_mean_fitness,
        parent_mean_fitness=parent_mean_fitness,
    )

    target = advisor.predict(features).astype(float)
    target = np.clip(target, 0.0, 1.0)

    if target.shape != pre.shape:
        raise ValueError(
            f"advisor target shape {target.shape} != genome shape {pre.shape}"
        )

    mutated = pre + shift_strength * (target - pre)
    mutated = np.clip(mutated, 0.0, 1.0)
    deltas = mutated - pre
    return mutated, deltas