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
