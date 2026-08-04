from enum import Enum


class DeathCause(Enum):
    """Possible causes of an agent's death."""

    THRESHOLD = "threshold"
    FITNESS = "fitness"
    AGE = "age"
    COMPETITION = "competition"


class Gender(Enum):
    """Biological sex of an agent."""

    MALE = "male"
    FEMALE = "female"


class RunStatus(Enum):
    """Terminal state of a simulation run."""

    COMPLETED = "completed"
    EXTINCT = "extinct"
    NO_FEMALES = "no_females"
    NO_MALES = "no_males"
