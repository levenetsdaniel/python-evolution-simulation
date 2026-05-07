import numpy as np
from config.sim_config import FitnessConfig, EnvironmentConfig

config = FitnessConfig()
env_config = EnvironmentConfig()


def _temp_score(heat_res: float, cold_res: float, temp: float, optimum: float) -> float:
    """
    Calculates the temperature score given an agent's heat resistance, cold resistance and environment's current temperature.

    Args:
        heat_res: Individual resistance to high temperatures.
        cold_res: Individual resistance to low temperatures.
        temp: Current environmental temperature.
        optimum: Target (optimal) normalized temperature for fitness peak.

    Returns:
        A value in the range [score_floor, 1.0] representing temperature fitness.
    """

    temp_norm = (temp - env_config.min_temperature) / (env_config.max_temperature - env_config.min_temperature)
    delta = temp_norm - config.temp_20_norm
    temp_intensity = abs(delta / config.temp_20_norm)
    if delta >= 0:
        temp_res = 1 - temp_intensity * (1 - heat_res) * cold_res
    else:
        temp_res = 1 - temp_intensity * (1 - cold_res) * heat_res

    temp_score = np.exp(- ((temp_res - optimum) ** 2) / config.temp_score_sharpness)

    return np.clip(temp_score, config.score_floor, 1.0)


def _hazard_score(resilience: float, hazard: float) -> float:
    """
    Calculates the hazard score given an agent's resilience and environment's hazard level

    Args:
        resilience: Individual resilience trait.
        hazard: Current environmental hazard level.

    Returns:
        A value in the range [score_floor, 1.0] representing hazard fitness.
    """

    fragility = hazard * (1.0 - resilience)
    hazard_score = np.exp(-fragility ** 2)

    return np.clip(hazard_score, config.score_floor, 1.0)


def _energy_score(satiation: float, resilience: float, metabolic_rate: float, aggressiveness: float) -> float:
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

    efficiency = 1.0 - metabolic_rate * config.metabolic_rate_efficiency_penalty - resilience * config.resilience_efficiency_penalty

    aggression_excess = max(0.0, aggressiveness - metabolic_rate)
    metabolic_penalty = aggression_excess * config.aggression_metabolic_penalty

    energy_score = satiation * efficiency - metabolic_penalty

    return np.clip(energy_score, config.score_floor, 1.0)


def _proportion_score(speed: float, size: float, metabolic_rate:float) -> float:
    """
        Penalize agents whose speed or size is disproportionate to their metabolic rate.

        Args:
            speed: Individual speed trait.
            size: Individual size trait.
            metabolic_rate: Energy consumption rate.

        Returns:
            A value in the range [score_floor, 1.0] representing body-proportion fitness.
        """

    speed_excess = max(0.0, speed - config.speed_metabolic_ratio * metabolic_rate)
    size_excess = max(0.0, size - config.size_metabolic_ratio * metabolic_rate)
    penalty = (speed_excess + size_excess) * config.proportion_penalty

    return np.clip(1.0 - penalty, config.score_floor, 1.0)


def fitness(ind_params: dict[str, float], env_params: dict[str, float]) -> float:
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

    temp_score = _temp_score(ind_params["heat_resistance"], ind_params["cold_resistance"], env_params["temperature"],
                             env_params["optimum_temperature"])

    energy_score = _energy_score(ind_params["satiation"], ind_params["resilience"], ind_params["metabolic_rate"],
                                 ind_params["aggressiveness"])

    hazard_score = _hazard_score(ind_params["resilience"], env_params["hazard_level"])

    proportion_score = _proportion_score(ind_params["speed"], ind_params["size"], ind_params["metabolic_rate"])

    fitness_score = temp_score * energy_score * hazard_score * proportion_score

    return fitness_score
