import argparse
from .sim_config import SimConfig, EnvironmentConfig, PopulationConfig, IndividualConfig

ind = IndividualConfig()
pop = PopulationConfig()
env = EnvironmentConfig()
sim = SimConfig()


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="EvoSim — evolutionary simulation")

    parser.add_argument("--fitness-death-threshold", type=float, default=ind.fitness_death_threshold)
    parser.add_argument("--fitness-death-prob-coef", type=float, default=ind.fitness_death_prob_coef)
    parser.add_argument("--age-scale", type=float, default=ind.age_scale)
    parser.add_argument("--age-death-power", type=float, default=ind.age_death_power)
    parser.add_argument("--food-need-size-coef", type=float, default=ind.food_need_size_coef)
    parser.add_argument("--food-need-resilience-coef", type=float, default=ind.food_need_resilience_coef)
    parser.add_argument("--food-need-speed-coef", type=float, default=ind.food_need_speed_coef)
    parser.add_argument("--food-need-aggr-coef", type=float, default=ind.food_need_aggr_coef)

    parser.add_argument("--population-size", type=int, default=pop.initial_size)
    parser.add_argument("--mutation-std", type=float, default=pop.mutation_std)
    parser.add_argument("--reproduction-rate", type=float, default=pop.reproduction_rate)
    parser.add_argument("--min-reproduction-age", type=int, default=pop.min_reproduction_age)
    parser.add_argument("--wound-base", type=float, default=pop.wound_base)

    parser.add_argument("--food-availability", type=float, default=env.food_availability)
    parser.add_argument("--max-temp", type=float, default=env.max_temperature)
    parser.add_argument("--min-temp", type=float, default=env.min_temperature)

    parser.add_argument("--n-steps", type=int, default=sim.n_steps)
    parser.add_argument("--seed", type=int, default=sim.seed)
    parser.add_argument("--output", type=str, default=sim.output_path)
    parser.add_argument("--steps-info", type=int, default=sim.steps_info)
    parser.add_argument("--model-info", action="store_true", default=sim.model_info)
    parser.add_argument("--population-info", action="store_true", default=sim.population_info)
    parser.add_argument("--individual-info", action="store_true", default=sim.individual_info)
    parser.add_argument("--record", action="store_true", default=sim.record)
    parser.add_argument("--debug", action="store_true", default=sim.debug)

    args = parser.parse_args()

    return SimConfig(
        environment=EnvironmentConfig(
            food_availability=args.food_availability,
            max_temperature=args.max_temp,
            min_temperature=args.min_temp
        ),

        population=PopulationConfig(
            initial_size=args.population_size,
            mutation_std=args.mutation_std,
            reproduction_rate=args.reproduction_rate,
            min_reproduction_age=args.min_reproduction_age,
            wound_base=args.wound_base
        ),

        individual=IndividualConfig(
            fitness_death_threshold=args.fitness_death_threshold,
            fitness_death_prob_coef=args.fitness_death_prob_coef,
            age_scale=args.age_scale,
            age_death_power=args.age_death_power,
            food_need_size_coef=args.food_need_size_coef,
            food_need_resilience_coef=args.food_need_resilience_coef,
            food_need_speed_coef=args.food_need_speed_coef,
            food_need_aggr_coef=args.food_need_aggr_coef
        ),

        n_steps=args.n_steps,
        seed=args.seed,
        output_path=args.output,
        steps_info=args.steps_info,
        model_info=args.model_info,
        population_info=args.population_info,
        individual_info=args.individual_info,
        record=args.record,
        debug=args.debug
    )
