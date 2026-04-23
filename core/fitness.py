import numpy as np
from config.sim_config import FitnessConfig, EnvironmentConfig

config = FitnessConfig()
env_config = EnvironmentConfig()


def _temp_score(heat_res: float, cold_res: float, temp: float, optimum: float) -> float:
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
    fragility = hazard * (1.0 - resilience)
    hazard_score = np.exp(-fragility ** 2)

    return np.clip(hazard_score, config.score_floor, 1.0)


def _energy_score(satiation: float, resilience: float, metabolic_rate: float, aggressiveness: float) -> float:
    efficiency = 1.0 - metabolic_rate * config.metabolic_rate_efficiency_penalty - resilience * config.resilience_efficiency_penalty

    aggression_excess = max(0.0, aggressiveness - metabolic_rate)
    metabolic_penalty = aggression_excess * config.aggression_metabolic_penalty

    energy_score = satiation * efficiency - metabolic_penalty

    return np.clip(energy_score, config.score_floor, 1.0)


def fitness(ind_params: dict[str, float], env_params: dict[str, float]) -> float:
    temp_score = _temp_score(ind_params["heat_resistance"], ind_params["cold_resistance"], env_params["temperature"],
                             env_params["optimum_temperature"])

    energy_score = _energy_score(ind_params["satiation"], ind_params["resilience"], ind_params["metabolic_rate"],
                                 ind_params["aggressiveness"])

    hazard_score = _hazard_score(ind_params["resilience"], env_params["hazard_level"])

    fitness_score = temp_score * energy_score * hazard_score

    return fitness_score
