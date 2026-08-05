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


def collect_history_row(model: Model, env_before: dict) -> dict:
    """Collect one history row after a step using environment values from step start."""
    pop = model.population

    return {
        "Generation": pop.generation,
        "PopulationSize": float(pop.actual_pop_size),
        "FemaleCount": float(pop.count_females),
        "MaleCount": float(pop.count_males),
        "AvgFitness": float(pop.avg_fitness),
        "AvgAge": float(pop.avg_age),
        "AvgSatiation": float(pop.avg_satiation),
        "AvgEnergy": float(pop.avg_energy),
        "AvgHeatResistance": float(pop.avg_heat_resistance),
        "AvgColdResistance": float(pop.avg_cold_resistance),
        "AvgMetabolicRate": float(pop.avg_metabolic_rate),
        "AvgResilience": float(pop.avg_resilience),
        "AvgSize": float(pop.avg_size),
        "AvgSpeed": float(pop.avg_speed),
        "AvgAggressiveness": float(pop.avg_aggressiveness),
        "TemperatureStart": float(env_before["temperature"]),
        "FoodStart": float(env_before["food_availability"]),
        "FoodRemaining": float(model.environment.current_food),
        "HazardStart": float(env_before["hazard_level"]),
        "BirthCount": float(model.births_this_step),
        "Deaths_age": float(model.deaths_this_step[DeathCause.AGE]),
        "Deaths_fitness": float(model.deaths_this_step[DeathCause.FITNESS]),
        "Deaths_threshold": float(model.deaths_this_step[DeathCause.THRESHOLD]),
        "Deaths_competition": float(model.deaths_this_step[DeathCause.COMPETITION]),
        "Deaths_total": float(sum(model.deaths_this_step.values())),
        "PopulationDelta": float(model.births_this_step - sum(model.deaths_this_step.values())),
        "AttacksConsidered": float(model.attacks_this_step["considered"]),
        "AttacksDeclined": float(model.attacks_this_step["declined"]),
        "AttacksStarted": float(model.attacks_this_step["started"]),
        "AttacksLost": float(model.attacks_this_step["lost"]),
        "AttacksWon": float(model.attacks_this_step["won"]),
        "FoodStolen": float(model.attacks_this_step["food_stolen"]),
        "AttackEnergySpent": float(model.attacks_this_step["energy_spent"]),
    }


def collect_history(model: Model, n_steps: int) -> pd.DataFrame:
    """
    Run the model step by step and collect population means plus environment values.

    Population metrics are taken after each completed step, while environment
    metrics are recorded from the beginning of that step.
    """
    rows = [collect_history_row(model, model.environment.current_params.copy())]

    if is_extinct(model):
        return pd.DataFrame(rows)

    for _ in range(n_steps):
        env_before = model.environment.current_params.copy()

        model.step()
        rows.append(collect_history_row(model, env_before))

        if is_extinct(model):
            break

    return pd.DataFrame(rows)
