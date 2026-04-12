import numpy as np


class Environment:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}

        self.time = 0
        self.current_params = {
            "temperature": 20.0,
            "food_availability": 1.0,
            "hazard_level": 0.1
        }
        self.prev_params = self.current_params.copy()

    def step(self):
        self.prev_params = self.current_params.copy()
        self.time += 1

        self.current_params["temperature"] += 0.05
        if self.current_params["temperature"] >= 50.0:
            self.current_params["temperature"] = -5.0

        n_alive = self.model.population.actual_size
        consumption = n_alive * 0.0001
        food = self.current_params["food_availability"]
        food += 0.05 * (1 - food)
        food -= consumption
        food = np.clip(food, 0.0, 1.0)

        self.current_params["food_availability"] = food
