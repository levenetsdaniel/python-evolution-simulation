from enum import Enum


class DeathCause(Enum):
    THRESHOLD = "threshold"
    FITNESS = "fitness"
    AGE = "age"
    COMPETITION = "competition"


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
