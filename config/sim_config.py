from dataclasses import dataclass, field
from math import isfinite

from .output_paths import PROFILING_DIR


@dataclass
class EnvironmentConfig:
    """Parameters controlling the environment."""

    food_availability: float = 10000.0
    food_regeneration_rate: float = 1.0
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
    attack_base_probability: float = 0.05
    attack_risk_aversion: float = 0.6
    attack_energy_cost: float = 2.0
    reproduction_energy_cost: float = 15.0


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
    energy_capacity: float = 100.0
    energy_initial: float = 70.0
    food_energy_conversion: float = 0.5
    basal_energy_cost: float = 1.0
    speed_energy_cost: float = 2.0
    aggressiveness_energy_cost: float = 3.0


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
class SpatialConfig:
    """Parameters for the independent spatial life cycle (one step is one time unit)."""

    width: float = 100.0
    height: float = 100.0
    initial_population: int = 100
    max_population: int = 500
    initial_energy: float = 100.0
    max_energy: float = 150.0
    cluster_count: int = 5
    sources_per_cluster: int = 12
    cluster_spread: float = 8.0
    food_capacity: float = 100.0
    food_initial: float = 100.0
    food_regeneration: float = 30.0
    food_radius: float = 10.0
    food_bite: float = 10.0
    food_energy: float = 8.0
    food_disappearance_probability: float = 0.6
    food_spawn_probability: float = 0.3
    movement_energy_cost: float = 0.5
    base_speed: float = 0.5
    speed_trait_scale: float = 1.5
    base_perception: float = 25.0
    perception_trait_scale: float = 12.0
    exploration_turn: float = 0.5
    reproduction_energy: float = 90.0
    offspring_energy: float = 30.0
    reproduction_rate: float = 1.0
    min_reproduction_age: int = 2
    mating_radius: float = 25.0
    offspring_offset: float = 2.0
    mutation_std: float = 0.12


@dataclass
class SimConfig:
    """Simulation configuration. Aggregates all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    individual: IndividualConfig = field(default_factory=IndividualConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    mode: str = "spatial"
    spatial_csv: str = ""
    spatial_html: str = "artifacts/plots/spatial/world.html"
    spatial_view_interval: int = 5
    n_steps: int = 600
    seed: int = 42
    steps_info: int = 5
    model_info: bool = False
    population_info: bool = False
    individual_info: bool = False
    collect_agent_data: bool = False
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
    _require(
        0 <= environment.food_regeneration_rate <= 1,
        "food_regeneration_rate must be in [0, 1]",
    )
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
    _require(
        0 <= population.attack_base_probability <= 1,
        "attack_base_probability must be in [0, 1]",
    )
    _require(
        0 <= population.attack_risk_aversion <= 1,
        "attack_risk_aversion must be in [0, 1]",
    )
    _require(population.attack_energy_cost >= 0, "attack_energy_cost must be non-negative")
    _require(population.reproduction_energy_cost >= 0, "reproduction_energy_cost must be non-negative")

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
                individual.food_energy_conversion,
                individual.basal_energy_cost,
                individual.speed_energy_cost,
                individual.aggressiveness_energy_cost,
            )
        ),
        "food need coefficients must be non-negative",
    )
    _require(individual.energy_capacity > 0, "energy_capacity must be positive")
    _require(
        0 <= individual.energy_initial <= individual.energy_capacity,
        "energy_initial must be in [0, energy_capacity]",
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
    _require(config.mode in ("classic", "spatial"), "mode must be classic or spatial")
    _require(config.spatial_view_interval >= 1, "spatial_view_interval must be positive")
    validate_spatial_config(config.spatial)

    for name, value in _numeric_fields(config):
        _require(isfinite(value), f"{name} must be finite")


def _numeric_fields(config: SimConfig):
    for group_name in ("environment", "population", "individual", "fitness", "spatial"):
        group = getattr(config, group_name)
        for name, value in vars(group).items():
            if isinstance(value, (int, float)):
                yield f"{group_name}.{name}", value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid simulation config: {message}")


def validate_spatial_config(config: SpatialConfig) -> None:
    """Validate independently, so SpatialWorld can be used without SimConfig."""
    from dataclasses import fields

    for item in fields(config):
        value = getattr(config, item.name)
        if isinstance(value, (int, float)):
            _require(isfinite(value), f"spatial.{item.name} must be finite")
            _require(value >= 0, f"spatial.{item.name} must be non-negative")
    for name in ("width", "height", "max_population", "max_energy", "food_capacity"):
        _require(getattr(config, name) > 0, f"spatial.{name} must be positive")
    _require(config.initial_population <= config.max_population, "spatial.initial_population exceeds max_population")
    _require(config.initial_energy <= config.max_energy, "spatial.initial_energy exceeds max_energy")
    _require(config.food_initial <= config.food_capacity, "spatial.food_initial exceeds food_capacity")
    _require(config.offspring_energy <= config.max_energy, "spatial.offspring_energy exceeds max_energy")
    _require(config.reproduction_energy >= config.offspring_energy,
             "spatial.reproduction_energy must cover offspring_energy")
    _require(config.reproduction_rate <= 1, "spatial.reproduction_rate must be in [0, 1]")
    _require(config.food_disappearance_probability <= 1,
             "spatial.food_disappearance_probability must be in [0, 1]")
    _require(config.food_spawn_probability <= 1,
             "spatial.food_spawn_probability must be in [0, 1]")
