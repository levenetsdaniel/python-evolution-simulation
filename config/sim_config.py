from dataclasses import dataclass, field
from math import isfinite


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

    fitness_death_threshold: float = 0.05
    fitness_death_prob_coef: float = 0.03
    age_scale: float = 160.0
    age_death_power: float = 1.8
    food_need_size_coef: float = 5.0
    food_need_resilience_coef: float = 2.0
    food_need_speed_coef: float = 1.0
    food_need_aggr_coef: float = 10.0
    energy_capacity: float = 100.0
    energy_initial: float = 70.0
    food_energy_conversion: float = 0.5
    basal_energy_cost: float = 1.0
    speed_energy_cost: float = 0.3
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
    """Spatial life cycle; food values are per-source bases before environment scaling."""

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
    food_regeneration: float = 10.0
    food_radius: float = 6.0
    food_bite: float = 10.0
    food_energy: float = 3.0
    food_disappearance_probability: float = 0.6
    food_spawn_probability: float = 0.3
    movement_energy_cost: float = 0.5
    base_speed: float = 0.5
    speed_trait_scale: float = 1.5
    base_perception: float = 25.0
    perception_trait_scale: float = 12.0
    local_map_size: int = 64
    local_map_span: float = 96.0
    nearby_map_span: float = 12.0
    exploration_turn: float = 0.5
    foraging_threshold: float = 0.6
    satiation_threshold: float = 0.85
    reproduction_energy: float = 90.0
    offspring_energy: float = 45.0
    reproduction_rate: float = 1.0
    min_reproduction_age: int = 2
    mating_radius: float = 2.0  # Local-map cells; converted to world distance.
    offspring_offset: float = 2.0
    mutation_std: float = 0.06


@dataclass
class SpatialGuidanceConfig:
    """Optional CatBoost guidance for spatial offspring mutations."""

    enabled: bool = False
    shift_strength: float = 0.3
    advisor_path: str = "artifacts/models/spatial_advisor.cbm"


@dataclass
class SpatialCNNConfig:
    """Optional visual target selection trained by imitating the heuristic."""

    enabled: bool = False
    device: str = "auto"
    training_steps: int = 80
    training_runs: int = 2
    validation_steps: int = 20
    max_samples: int = 4096
    epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 0.003
    target_loss_weight: float = 1.0
    threads: int = 1
    model_path: str = "artifacts/models/spatial_target_policy.pt"
    save_path: str = "artifacts/models/spatial_target_policy.pt"


@dataclass
class SimConfig:
    """Simulation configuration. Aggregates all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    individual: IndividualConfig = field(default_factory=IndividualConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    spatial_guidance: SpatialGuidanceConfig = field(default_factory=SpatialGuidanceConfig)
    spatial_cnn: SpatialCNNConfig = field(default_factory=SpatialCNNConfig)
    mode: str = "spatial"
    spatial_csv: str = ""
    spatial_html: str = "artifacts/plots/spatial/world.html"
    spatial_view_interval: int = 1
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

    simulation: SimConfig = field(default_factory=lambda: SimConfig(mode="classic"))
    advisor_path: str = "artifacts/models/classic_advisor.cbm"
    shift_strength: float = 0.7
    x_trait: str = "heat_resistance"
    y_trait: str = "cold_resistance"
    view: bool = False


@dataclass
class AdvisorTrainingConfig:
    """Offline CatBoost training on independent baseline simulation runs."""

    simulation: SimConfig = field(default_factory=lambda: SimConfig(n_steps=100))
    training_scenarios: list[str] = field(default_factory=list)
    training_runs: int = 1
    validation_runs: int = 1
    balance_scenarios: bool = True
    iterations: int = 100
    save_path: str = ""  # Defaults to artifacts/models/<simulation.mode>_advisor.cbm.


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
    _require(not config.spatial_cnn.enabled or config.mode == "spatial",
             "spatial_cnn requires mode=spatial")
    for name in ("training_steps", "training_runs", "validation_steps", "max_samples",
                 "epochs", "batch_size", "threads"):
        value = getattr(config.spatial_cnn, name)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                 f"spatial_cnn.{name} must be a positive integer")
    _require(config.spatial_cnn.learning_rate > 0, "spatial_cnn.learning_rate must be positive")
    _require(config.spatial_cnn.target_loss_weight > 0,
             "spatial_cnn.target_loss_weight must be positive")
    device = config.spatial_cnn.device
    _require(isinstance(device, str) and (device in ("auto", "cpu", "cuda", "mps")
             or (device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit())),
             "spatial_cnn.device must be auto, cpu, cuda, cuda:N or mps")
    if config.spatial_cnn.enabled:
        _require(config.spatial.local_map_size >= 8, "spatial_cnn requires local_map_size >= 8")
        _require(config.spatial.local_map_span / 2 >= (config.spatial.base_perception
                 + config.spatial.perception_trait_scale),
                 "spatial_cnn local map must cover the maximum perception radius")
    _require(not config.spatial_guidance.enabled or config.mode == "spatial",
             "spatial_guidance requires mode=spatial")
    _require(not config.spatial_guidance.enabled or bool(config.spatial_guidance.advisor_path.strip()),
             "spatial_guidance.advisor_path is required")
    _require(config.spatial_view_interval >= 1, "spatial_view_interval must be positive")
    _require(0 <= config.spatial_guidance.shift_strength <= 1,
             "spatial_guidance.shift_strength must be in [0, 1]")
    validate_spatial_config(config.spatial)

    for name, value in _numeric_fields(config):
        _require(isfinite(value), f"{name} must be finite")


def _numeric_fields(config: SimConfig):
    for group_name in ("environment", "population", "individual", "fitness", "spatial", "spatial_guidance", "spatial_cnn"):
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
    _require(isinstance(config.local_map_size, int) and not isinstance(config.local_map_size, bool)
             and config.local_map_size > 0, "spatial.local_map_size must be a positive integer")
    _require(config.local_map_span > 0, "spatial.local_map_span must be positive")
    _require(0 < config.nearby_map_span <= config.local_map_span,
             "spatial.nearby_map_span must be positive and no larger than local_map_span")
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
    _require(0 <= config.foraging_threshold < config.satiation_threshold <= 1,
             "spatial energy thresholds must satisfy 0 <= foraging_threshold < satiation_threshold <= 1")
