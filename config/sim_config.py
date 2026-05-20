from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Parameters controlling the environment."""

    food_availability: float = 10000.0
    food_step: float = 0.0
    temp_start: float = 20.0
    temp_step: float = 0.05
    temp_reset: float = -5.0
    hazard_level_start: float = 0.1
    hazard_step: float = 0.001
    min_temperature: float = -30.0
    max_temperature: float = 50.0


@dataclass
class EnvironmentEventConfig:
    """Parameters controlling one temporary environment event."""

    event_type: str
    start_step: int
    duration: int
    temperature_delta: float = 0.0
    hazard_delta: float = 0.0


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
    environment_events: list[EnvironmentEventConfig] = field(default_factory=list)

    n_steps: int = 600
    retrain_steps: int = 50
    seed: int = 42
    output_path: str = "data/training_samples.json"
    steps_info: int = 5
    shift_strength: float = 0.7
    model_info: bool = False
    population_info: bool = False
    individual_info: bool = False
    record: bool = False
    x_trait: str = "heat_resistance"
    y_trait: str = "cold_resistance"
    view: bool = False
    debug: bool = False

@dataclass
class ProfilerConfig:
    """Parameters controlling the profiler."""

    simulation_config: SimConfig = field(default_factory=SimConfig)

    output_dir: str = "reports"
    prof_filename: str = "evosim.prof"
    text_summary_filename: str = "cprofile_summary.txt"
    html_report_filename: str = "profiling_report.html"

    top_n_text: int = 50
    top_n_callers: int = 10
    top_n_flame: int = 15
    top_n_console: int = 15

    view: bool = False
