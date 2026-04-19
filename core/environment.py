import numpy as np

WOUND_BASE = 0.4
MAX_FOOD_BASE = 200000
REGEN_RATE = 5000
MAX_CAPACITY = 2000


class Environment:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}

        self.time = 0
        self.current_params = {
            "temperature": 20.0,
            "optimum_temperature": 0.5,
            "food_availability": 5000,
            "hazard_level": 0.1
        }
        self.prev_params = self.current_params.copy()

    def food_distribution(self):
        alive = [a for a in self.model.agents if a.is_alive]

        raw_weight = np.array([a.genes_map["speed"] * 2.0 / a.food_need for a in alive])
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
            self.model.population.compete(hungry, fed)

    def step(self):
        self.prev_params = self.current_params.copy()
        self.time += 1

        self.current_params["temperature"] += 0.05
        self.current_params["optimum_temperature"] += 0.00125
        if self.current_params["temperature"] >= 50.0:
            self.current_params["temperature"] = -5.0
            self.current_params["optimum_temperature"] = 0.1

        self.current_params["hazard_level"] += 0.001

        self.current_params["food_availability"] = REGEN_RATE
