import numpy as np
from config.sim_config import FitnessConfig, EnvironmentConfig

DEFAULT_FITNESS_CONFIG = FitnessConfig()
DEFAULT_ENV_CONFIG = EnvironmentConfig()


def _temp_score(
    heat_res: float,
    cold_res: float,
    temp: float,
    fitness_config: FitnessConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
) -> float:
    """
    Calculates the temperature score given an agent's heat resistance, cold resistance and environment's current temperature.

    Args:
        heat_res: Individual resistance to high temperatures.
        cold_res: Individual resistance to low temperatures.
        temp: Current environmental temperature.

    Returns:
        A value in the range [score_floor, 1.0] representing temperature fitness.
    """

    fitness_config = fitness_config or DEFAULT_FITNESS_CONFIG
    environment_config = environment_config or DEFAULT_ENV_CONFIG

    temp_norm = (
        (temp - environment_config.min_temperature)
        / (environment_config.max_temperature - environment_config.min_temperature)
    )
    delta = temp_norm - fitness_config.temp_20_norm
    adaptation = heat_res if delta >= 0 else cold_res

    pressure = abs(delta) * (1.0 - adaptation)
    score = np.exp(-pressure ** 2 / fitness_config.temp_score_sharpness)

    return float(np.clip(score, fitness_config.score_floor, 1.0))


def _hazard_score(
    resilience: float,
    hazard: float,
    fitness_config: FitnessConfig | None = None,
) -> float:
    """
    Calculates the hazard score given an agent's resilience and environment's hazard level

    Args:
        resilience: Individual resilience trait.
        hazard: Current environmental hazard level.

    Returns:
        A value in the range [score_floor, 1.0] representing hazard fitness.
    """

    fitness_config = fitness_config or DEFAULT_FITNESS_CONFIG

    fragility = hazard * (1.0 - resilience)
    hazard_score = np.exp(-fragility ** 2)

    return np.clip(hazard_score, fitness_config.score_floor, 1.0)


def _energy_score(
    satiation: float,
    resilience: float,
    metabolic_rate: float,
    aggressiveness: float,
    fitness_config: FitnessConfig | None = None,
) -> float:
    """
    Calculates the energy score given an agent's satiation, resilience, aggressiveness and metabolic rate.

    Args:
        satiation: Ratio of consumed food to required food.
        resilience: Individual resilience trait.
        metabolic_rate: Energy consumption rate.
        aggressiveness: Individual aggression level.

    Returns:
        A value in the range [score_floor, 1.0] representing energy fitness.
    """

    fitness_config = fitness_config or DEFAULT_FITNESS_CONFIG

    efficiency = (
        1.0
        - metabolic_rate * fitness_config.metabolic_rate_efficiency_penalty
        - resilience * fitness_config.resilience_efficiency_penalty
    )

    aggression_excess = max(0.0, aggressiveness - metabolic_rate)
    metabolic_penalty = aggression_excess * fitness_config.aggression_metabolic_penalty

    energy_score = satiation * efficiency - metabolic_penalty

    return np.clip(energy_score, fitness_config.score_floor, 1.0)


def _proportion_score(
    speed: float,
    size: float,
    metabolic_rate: float,
    fitness_config: FitnessConfig | None = None,
) -> float:
    """
        Penalize agents whose speed or size is disproportionate to their metabolic rate.

        Args:
            speed: Individual speed trait.
            size: Individual size trait.
            metabolic_rate: Energy consumption rate.

        Returns:
            A value in the range [score_floor, 1.0] representing body-proportion fitness.
        """

    fitness_config = fitness_config or DEFAULT_FITNESS_CONFIG

    speed_excess = max(0.0, speed - fitness_config.speed_metabolic_ratio * metabolic_rate)
    size_excess = max(0.0, size - fitness_config.size_metabolic_ratio * metabolic_rate)
    penalty = (speed_excess + size_excess) * fitness_config.proportion_penalty

    return np.clip(1.0 - penalty, fitness_config.score_floor, 1.0)


def fitness(
    ind_params: dict[str, float],
    env_params: dict[str, float],
    fitness_config: FitnessConfig | None = None,
    environment_config: EnvironmentConfig | None = None,
) -> float:
    """
    Calculates the fitness given an individual's and environment's parameters

    Fitness is calculated as a multiplicative product of three independent components:

        - Temperature adaptation (_temp_score)
        - Energy efficiency (_energy_score)
        - Hazard resistance (_hazard_score)
        - Body proportions (_proportion_score)

    Args:
        ind_params: Dictionary of individual traits.
        env_params: Dictionary of environmental parameters.

    Returns:
        A scalar fitness score in the range [score_floor, 1.0].
    """

    fitness_config = fitness_config or DEFAULT_FITNESS_CONFIG
    environment_config = environment_config or DEFAULT_ENV_CONFIG

    temp_score = _temp_score(
        ind_params["heat_resistance"],
        ind_params["cold_resistance"],
        env_params["temperature"],
        fitness_config=fitness_config,
        environment_config=environment_config,
    )

    energy_score = _energy_score(
        ind_params["satiation"],
        ind_params["resilience"],
        ind_params["metabolic_rate"],
        ind_params["aggressiveness"],
        fitness_config=fitness_config,
    )

    hazard_score = _hazard_score(
        ind_params["resilience"],
        env_params["hazard_level"],
        fitness_config=fitness_config,
    )

    proportion_score = _proportion_score(
        ind_params["speed"],
        ind_params["size"],
        ind_params["metabolic_rate"],
        fitness_config=fitness_config,
    )

    fitness_score = temp_score * energy_score * hazard_score * proportion_score

    return fitness_score
