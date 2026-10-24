from hydra.core.config_store import ConfigStore

from .scenarios import SCENARIOS
from .sim_config import (AdvisorTrainingConfig, ComparisonConfig, FitnessConfig, IndividualConfig,
                         PolicyEvaluationConfig, PopulationConfig, PPOTrainingConfig, SimConfig)


def register_configs() -> None:
    """Register dataclass schemas and scenario variants in Hydra ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="sim_schema", node=SimConfig)
    cs.store(name="compare_schema", node=ComparisonConfig)
    cs.store(name="advisor_schema", node=AdvisorTrainingConfig)
    cs.store(name="ppo_schema", node=PPOTrainingConfig)
    cs.store(name="policy_evaluation_schema", node=PolicyEvaluationConfig)
    cs.store(group="population", name="schema", node=PopulationConfig)
    cs.store(group="individual", name="schema", node=IndividualConfig)
    cs.store(group="fitness", name="schema", node=FitnessConfig)

    for name, env in SCENARIOS.items():
        cs.store(group="environment", name=name, node=env)
