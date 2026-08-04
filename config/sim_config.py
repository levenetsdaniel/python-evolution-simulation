from dataclasses import dataclass, field
from math import isfinite

from .output_paths import PROFILING_DIR


@dataclass
class EnvironmentConfig:
    """Parameters controlling the environment."""

    food_availability: float = 10000.0
    food_step: float = 0.0
    temp_start: float = 20.0
    temp_step: float = 0.05
    hazard_level_start: float = 0.1
    hazard_step: float = 0.001
    min_temperature: float = -30.0
    max_temperature: float = 50.0


@dataclass
class PopulationConfig:
    """Parameters controlling population size, reproduction, and genetics."""

    genome_labels: list[str] = field(
        default_factory=lambda: ["heat_resistance", "cold_resistance", "metabolic_rate", "resilience", "size", "speed",
                                 "aggressiveness"])
    initial_size: int = 100
    mutation_std: float = 0.12
    reproduction_rate: float = 0.4
    min_reproduction_age: int = 2
    wound_base: float = 0.4


@dataclass
class IndividualConfig:
    """Parameters controlling individual agent behaviour and survival."""

    fitness_death_threshold: float = 0.1
    fitness_death_prob_coef: float = 0.1
    age_scale: float = 80.0
    age_death_power: float = 1.2
    food_need_size_coef: float = 5.0
    food_need_resilience_coef: float = 2.0
    food_need_speed_coef: float = 1.0
    food_need_aggr_coef: float = 10.0


@dataclass
class FitnessConfig:
    """Parameters controlling the fitness scoring functions."""

    temp_20_norm: float = 0.625
    temp_score_sharpness: float = 0.5
    metabolic_rate_efficiency_penalty: float = 0.5
    resilience_efficiency_penalty: float = 0.2
    aggression_metabolic_penalty: float = 0.1
    speed_metabolic_ratio: float = 2.0
    size_metabolic_ratio: float = 1.5
    proportion_penalty: float = 1.0
    score_floor: float = 0.05


@dataclass
class SimConfig:
    """Simulation configuration. Aggregates all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    individual: IndividualConfig = field(default_factory=IndividualConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    n_steps: int = 600
    seed: int = 42
    steps_info: int = 5
    model_info: bool = False
    population_info: bool = False
    individual_info: bool = False
    view: bool = False
    debug: bool = False


@dataclass
class ComparisonConfig:
    """Configuration for the ML-guided comparison pipeline."""

    simulation: SimConfig = field(default_factory=SimConfig)
    retrain_steps: int = 50
    shift_strength: float = 0.7
    x_trait: str = "heat_resistance"
    y_trait: str = "cold_resistance"
    view: bool = False


@dataclass
class ProfilerConfig:
    """Parameters controlling the profiler."""

    simulation_config: SimConfig = field(default_factory=SimConfig)

    output_dir: str = str(PROFILING_DIR)
    prof_filename: str = "evosim.prof"
    text_summary_filename: str = "cprofile_summary.txt"
    html_report_filename: str = "profiling_report.html"

    top_n_text: int = 50
    top_n_callers: int = 10
    top_n_flame: int = 15
    top_n_console: int = 15

    view: bool = False


def validate_sim_config(config: SimConfig) -> None:
    """Reject invalid simulation parameters before the model starts."""
    environment = config.environment
    population = config.population
    individual = config.individual
    fitness = config.fitness

    _require(environment.min_temperature < environment.max_temperature, "min_temperature must be below max_temperature")
    _require(environment.food_availability >= 0, "food_availability must be non-negative")
    _require(0 <= environment.hazard_level_start <= 1, "hazard_level_start must be in [0, 1]")

    required_labels = {
        "heat_resistance",
        "cold_resistance",
        "metabolic_rate",
        "resilience",
        "size",
        "speed",
        "aggressiveness",
    }
    _require(
        len(population.genome_labels) == len(required_labels) and set(population.genome_labels) == required_labels,
        "genome_labels must contain each required trait exactly once",
    )
    _require(population.initial_size >= 0, "initial_size must be non-negative")
    _require(population.mutation_std >= 0, "mutation_std must be non-negative")
    _require(0 <= population.reproduction_rate <= 1, "reproduction_rate must be in [0, 1]")
    _require(population.min_reproduction_age >= 0, "min_reproduction_age must be non-negative")
    _require(0 <= population.wound_base <= 1, "wound_base must be in [0, 1]")

    _require(0 <= individual.fitness_death_threshold <= 1, "fitness_death_threshold must be in [0, 1]")
    _require(0 <= individual.fitness_death_prob_coef <= 1, "fitness_death_prob_coef must be in [0, 1]")
    _require(individual.age_scale > 0, "age_scale must be positive")
    _require(individual.age_death_power > 0, "age_death_power must be positive")
    _require(
        all(
            coefficient >= 0
            for coefficient in (
                individual.food_need_size_coef,
                individual.food_need_resilience_coef,
                individual.food_need_speed_coef,
                individual.food_need_aggr_coef,
            )
        ),
        "food need coefficients must be non-negative",
    )

    _require(0 <= fitness.temp_20_norm <= 1, "temp_20_norm must be in [0, 1]")
    _require(fitness.temp_score_sharpness > 0, "temp_score_sharpness must be positive")
    _require(
        all(
            coefficient >= 0
            for coefficient in (
                fitness.metabolic_rate_efficiency_penalty,
                fitness.resilience_efficiency_penalty,
                fitness.aggression_metabolic_penalty,
                fitness.speed_metabolic_ratio,
                fitness.size_metabolic_ratio,
                fitness.proportion_penalty,
            )
        ),
        "fitness coefficients must be non-negative",
    )
    _require(0 <= fitness.score_floor <= 1, "score_floor must be in [0, 1]")

    _require(config.n_steps >= 0, "n_steps must be non-negative")
    _require(config.steps_info >= 0, "steps_info must be non-negative")

    for name, value in _numeric_fields(config):
        _require(isfinite(value), f"{name} must be finite")


def _numeric_fields(config: SimConfig):
    for group_name in ("environment", "population", "individual", "fitness"):
        group = getattr(config, group_name)
        for name, value in vars(group).items():
            if isinstance(value, (int, float)):
                yield f"{group_name}.{name}", value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid simulation config: {message}")
