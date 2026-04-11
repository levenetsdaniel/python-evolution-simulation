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

    def step(self):
        self.time += 1

        self.current_params["temperature"] += 0.05
        if self.current_params["temperature"] >= 50.0:
            self.current_params["temperature"] = 20.0

        self.current_params["food_availability"] *= 0.999
