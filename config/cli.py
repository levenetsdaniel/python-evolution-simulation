import argparse
from .sim_config import SimConfig, EnvironmentConfig, PopulationConfig, IndividualConfig, ProfilerConfig
from.scenarios import SCENARIOS

ind = IndividualConfig()
pop = PopulationConfig()
env = EnvironmentConfig()
sim = SimConfig()
prof = ProfilerConfig()


def parse_args() -> tuple[SimConfig, ProfilerConfig]:
    """
    Parse CLI arguments into a simulation configuration.

    Returns:
        Fully constructed simulation configuration.
    """

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
    parser.add_argument("--retrain-steps", type=int, default=sim.retrain_steps)
    parser.add_argument("--seed", type=int, default=sim.seed)
    parser.add_argument("--output", type=str, default=sim.output_path)
    parser.add_argument("--steps-info", type=int, default=sim.steps_info)
    parser.add_argument("--shift-strength", type=float, default=sim.shift_strength)
    parser.add_argument("--model-info", action="store_true", default=sim.model_info)
    parser.add_argument("--population-info", action="store_true", default=sim.population_info)
    parser.add_argument("--individual-info", action="store_true", default=sim.individual_info)
    parser.add_argument("--record", action="store_true", default=sim.record)
    parser.add_argument("--debug", action="store_true", default=sim.debug)

    parser.add_argument("--output-dir", type=str, default=prof.output_dir)
    parser.add_argument("--prof-filename", type=str, default=prof.prof_filename)
    parser.add_argument("--text-summary-filename", type=str, default=prof.text_summary_filename)
    parser.add_argument("--html-report-filename", type=str, default=prof.html_report_filename)
    parser.add_argument("--top-n-text", type=int, default=prof.top_n_text)
    parser.add_argument("--top-n-callers", type=int, default=prof.top_n_callers)
    parser.add_argument("--top-n-flame", type=int, default=prof.top_n_flame)
    parser.add_argument("--top-n-console", type=int, default=prof.top_n_console)

    parser.add_argument("--view", action="store_true", default=prof.view)

    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()), default=None)

    args = parser.parse_args()

    sim_config = SimConfig(
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
        retrain_steps=args.retrain_steps,
        shift_strength=args.shift_strength,
        seed=args.seed,
        output_path=args.output,
        steps_info=args.steps_info,
        model_info=args.model_info,
        population_info=args.population_info,
        individual_info=args.individual_info,
        record=args.record,
        debug=args.debug
    )

    prof_conf = ProfilerConfig(
        simulation_config=sim_config,

        output_dir=args.output_dir,
        prof_filename=args.prof_filename,
        text_summary_filename=args.text_summary_filename,
        html_report_filename=args.html_report_filename,
        top_n_text=args.top_n_text,
        top_n_callers=args.top_n_callers,
        top_n_flame=args.top_n_flame,
        top_n_console=args.top_n_console,

        view=args.view
    )

    if args.scenario:
        sim_config = SCENARIOS[args.scenario](sim_config)

    return sim_config, prof_conf


