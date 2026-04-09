import numpy as np


class Environment:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}

        self.time = 0
        self.current_params = {
            "temperature": 20.0,
            "resource_level": 1.0,
        }

    def step(self):
        self.time += 1

        self.current_params["temperature"] = 20 + 10 * np.sin(self.time * 0.1)

        self.current_params["resource_level"] *= 0.999
