import pandas as pd

from core.enums import DeathCause
from core.model import Model


def is_extinct(model: Model) -> bool:
    """Return True if the simulation can no longer continue."""
    if len(model.agents) == 0:
        return True
    if model.population.count_females == 0:
        return True
    if model.population.count_males == 0:
        return True
    return False


def collect_history_row(model: Model) -> dict:
    """Collect one history row from the environment and population after a step."""
    pop = model.population
    metrics = pop.cache_metrics()

    try:
        return {
            "TimeStep": model.step_count,
            "PopulationSize": float(metrics.population_size),
            "FemaleCount": float(metrics.female_count),
            "MaleCount": float(metrics.male_count),
            "AvgFitness": float(metrics.avg_fitness),
            "AvgAge": float(metrics.avg_age),
            "AvgSatiation": float(metrics.avg_satiation),
            "AvgEnergy": float(metrics.avg_energy),
            "AvgHeatResistance": float(metrics.avg_heat_resistance),
            "AvgColdResistance": float(metrics.avg_cold_resistance),
            "AvgMetabolicRate": float(metrics.avg_metabolic_rate),
            "AvgResilience": float(metrics.avg_resilience),
            "AvgSize": float(metrics.avg_size),
            "AvgSpeed": float(metrics.avg_speed),
            "AvgAggressiveness": float(metrics.avg_aggressiveness),
            "Temperature": float(model.environment.current_params["temperature"]),
            "FoodCapacity": float(model.environment.food_capacity),
            "FoodAvailable": float(model.environment.current_food),
            "HazardLevel": float(model.environment.current_params["hazard_level"]),
            "BirthCount": float(model.births_this_step),
            "Deaths_age": float(model.deaths_this_step[DeathCause.AGE]),
            "Deaths_fitness": float(model.deaths_this_step[DeathCause.FITNESS]),
            "Deaths_threshold": float(model.deaths_this_step[DeathCause.THRESHOLD]),
            "Deaths_competition": float(model.deaths_this_step[DeathCause.COMPETITION]),
            "Deaths_total": float(sum(model.deaths_this_step.values())),
            "PopulationDelta": float(model.births_this_step - sum(model.deaths_this_step.values())),
            "AttacksConsidered": float(model.attacks_this_step["considered"],),
            "AttacksDeclined": float(model.attacks_this_step["declined"]),
            "AttacksStarted": float(model.attacks_this_step["started"]),
            "AttacksLost": float(model.attacks_this_step["lost"]),
            "AttacksWon": float(model.attacks_this_step["won"]),
            "FoodStolen": float(model.attacks_this_step["food_stolen"]),
            "AttackEnergySpent": float(model.attacks_this_step["energy_spent"]),
        }
    finally:
        pop.invalidate_metrics()


def collect_history(model: Model, n_steps: int) -> pd.DataFrame:
    """
    Run the model step by step and collect population means plus environment values.

    Population and environment metrics are both taken after each completed step.
    """
    rows = [collect_history_row(model)]

    if is_extinct(model):
        return pd.DataFrame(rows)

    for _ in range(n_steps):
        model.step()
        rows.append(collect_history_row(model))

        if is_extinct(model):
            break

    return pd.DataFrame(rows)
