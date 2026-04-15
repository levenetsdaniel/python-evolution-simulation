import numpy as np
from .enums import DeathCause

WOUND_BASE = 0.4
MAX_FOOD_BASE = 20000
REGEN_RATE = 1000
MAX_CAPACITY = 2000


class Environment:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}

        self.time = 0
        self.current_params = {
            "temperature": 20.0,
            "optimum_temperature": 0.5,
            "food_availability": 1500,
            "hazard_level": 0.1
        }
        self.prev_params = self.current_params.copy()

    def _death_prob(self, loser, pover_dif):
        death_prob = WOUND_BASE * abs(pover_dif) * (1.0 - loser.genes_map["resilience"]) * loser.genes_map["aggressiveness"]
        if self.model.rng.random() < death_prob:
            loser.is_alive = False
            loser.death_cause = DeathCause.COMPETITION

    def _compete(self, hungry: list, fed: list):
        for attacker in hungry:
            if not fed:
                break

            victim = self.model.rng.choice(fed)

            a_power = attacker.genes_map["size"] + attacker.genes_map["aggressiveness"]
            v_power = victim.genes_map["size"] + victim.genes_map["aggressiveness"]
            total = a_power + v_power

            win_prob = a_power / total
            power_diff = (a_power - v_power) / total

            if self.model.rng.random() < win_prob:
                steal_coef = np.clip(0.5 + 0.5 * power_diff, 0.1, 0.9)
                stolen = max(1, int(victim.food_eaten * steal_coef))
                stolen = min(stolen, victim.food_eaten)

                victim.food_eaten -= stolen
                attacker.food_eaten += stolen

                self._death_prob(victim, power_diff)
                if not victim.is_alive:
                    fed.remove(victim)

            self._death_prob(attacker, power_diff)

    def food_distribution(self):
        alive = [a for a in self.model.agents if a.is_alive]

        raw_weight = np.array([a.genes_map["speed"] / a.food_need for a in alive])
        raw_weight = np.clip(raw_weight, 1e-6, None)
        weights = raw_weight / raw_weight.sum()

        ordered = list(self.model.rng.choice(alive, size=len(alive), replace=False, p=weights))

        fed = []
        hungry = []

        for a in ordered:
            if a.food_need < self.current_params["food_availability"]:
                a.food_eaten = a.food_need
                self.current_params["food_availability"] -= a.food_eaten
                fed.append(a)
            else:
                hungry.append(a)

        if hungry:
            self._compete(hungry, fed)

    def step(self):
        self.prev_params = self.current_params.copy()
        self.time += 1

        self.current_params["temperature"] += 0.05
        self.current_params["optimum_temperature"] += 0.00125
        if self.current_params["temperature"] >= 50.0:
            self.current_params["temperature"] = -5.0
            self.current_params["optimum_temperature"] = 0.1

        pop_size = self.model.population.actual_size
        overgrazing = np.clip(pop_size / MAX_CAPACITY, 0.0, 1.0)
        effective_regen = int(REGEN_RATE * (1.0 - 0.8 * overgrazing))
        self.current_params["food_availability"] = min(self.current_params["food_availability"] + effective_regen,
                                                       MAX_FOOD_BASE)
