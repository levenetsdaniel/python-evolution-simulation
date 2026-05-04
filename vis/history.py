import pandas as pd

from core.model import Model


def _is_extinct(model: Model) -> bool:
    """Return True if the simulation can no longer continue."""
    if len(model.agents) == 0:
        return True
    if model.population.count_females == 0:
        return True
    if model.population.count_males == 0:
        return True
    return False


def collect_history_row(model: Model, env_before: dict) -> dict:
    """Collect one history row after a step using environment values from step start."""
    pop = model.population

    return {
        "Generation": pop.generation,
        "PopulationSize": float(pop.actual_pop_size),
        "AvgFitness": float(pop.avg_fitness),
        "AvgAge": float(pop.avg_age),
        "AvgSatiation": float(pop.avg_satiation),
        "AvgHeatResistance": float(pop.avg_heat_resistance),
        "AvgColdResistance": float(pop.avg_cold_resistance),
        "AvgMetabolicRate": float(pop.avg_metabolic_rate),
        "AvgResilience": float(pop.avg_resilience),
        "AvgSize": float(pop.avg_size),
        "AvgSpeed": float(pop.avg_speed),
        "AvgAggressiveness": float(pop.avg_aggressiveness),
        "TemperatureStart": float(env_before["temperature"]),
        "FoodStart": float(env_before["food_availability"]),
        "HazardStart": float(env_before["hazard_level"]),
    }


def collect_history(model: Model, n_steps: int) -> pd.DataFrame:
    """
    Run the model step by step and collect population means plus environment values.

    Population metrics are taken after each completed step, while environment
    metrics are recorded from the beginning of that step.
    """
    rows: list[dict] = []

    for _ in range(n_steps):
        env_before = model.environment.current_params.copy()

        model.step()
        rows.append(collect_history_row(model, env_before))

        if _is_extinct(model):
            break

    return pd.DataFrame(rows)