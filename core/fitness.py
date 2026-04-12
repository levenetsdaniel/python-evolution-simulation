import numpy as np

MIN_TEMPERATURE = -30.0
MAX_TEMPERATURE = 50.0


def fitness(ind_params: dict, env_params: dict):
    temperature_norm = (env_params["temperature"] - MIN_TEMPERATURE) / (MAX_TEMPERATURE - MIN_TEMPERATURE)
    delta = temperature_norm - 0.625
    temperature_intensity = abs(delta / 0.625)
    if delta > 0.0:
        temperature_score = 1 - temperature_intensity * (1 - ind_params["heat_resistance"]) * ind_params["cold_resistance"]
    elif delta < 0.0:
        temperature_score = 1 - temperature_intensity * (1 - ind_params["cold_resistance"]) * ind_params["heat_resistance"]
    else:
        temperature_score = ind_params["heat_resistance"] * ind_params["cold_resistance"]

    food_need = ind_params["metabolic_rate"] * (1 + ind_params["size"])
    food_gathered = env_params["food_availability"] * (0.5 + 0.5 * ind_params["size"])
    energy_score = np.clip(food_gathered - food_need, 0, 1)

    hazard_score = 1 - env_params["hazard_level"] * (1 - ind_params["resilience"]) * 0.1

    fitness_score = temperature_score * energy_score * hazard_score

    return fitness_score
