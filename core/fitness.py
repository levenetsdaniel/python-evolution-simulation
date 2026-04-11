import numpy as np

MIN_TEMPERATURE = -30.0
MAX_TEMPERATURE = 50.0


def fitness(ind_params: dict, env_params: dict):
    temperature_norm = (env_params["temperature"] - MIN_TEMPERATURE) / (MAX_TEMPERATURE - MIN_TEMPERATURE)
    delta = temperature_norm - 0.625
    temperature_intensity = abs(delta / 0.625)
    if delta > 0.0:
        temperature_score = 1 - temperature_intensity * (1 - ind_params["heat_resistance"])
    elif delta < 0.0:
        temperature_score = 1 - temperature_intensity * (1 - ind_params["cold_resistance"])
    else:
        temperature_score = ind_params["heat_resistance"] * ind_params["cold_resistance"]

    metabolic_cost = ind_params["metabolic_rate"] * (1 + ind_params["size"])
    energy_score = np.clip(env_params["food_availability"] - metabolic_cost, 0, 1)

    hazard_score = 1 - env_params["hazard_level"] * (1 - ind_params["resilience"])

    fitness_score = temperature_score * energy_score * hazard_score

    return fitness_score
