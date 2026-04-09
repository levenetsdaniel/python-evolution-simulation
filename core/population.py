from .individual import Individual


class Population:
    def __init__(self, model, size=100):
        self.model = model
        self.size = size
        self.generation = 0

    def initialize(self, labels, ranges):
        for _ in range(self.size):
            ind = Individual.random_init(self.model, labels, ranges)
            self.model.agents.add(ind)

    def remove_dead(self):
        dead = self.model.agents.select(lambda a: not a.is_alive)
        for agent in list(dead):
            agent.remove()

    def reproduce(self):
        pass

    def step(self):
        self.model.agents.shuffle_do("step")
        self.remove_dead()
        self.reproduce()
        self.generation += 1
