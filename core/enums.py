from enum import Enum


class DeathCause(Enum):
    THRESHOLD = "threshold"
    FITNESS = "fitness"
    AGE = "age"


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
