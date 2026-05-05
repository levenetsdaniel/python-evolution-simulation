"""
Command-line entry point for running the simulation.

Initializes configuration from CLI, runs the model,
and optionally prints collected statistics.
"""

from config.cli import parse_args
from core.model import Model

if __name__ == "__main__":
    config, _ = parse_args()
    model = Model(config)
    model.run(config.n_steps)

    model_df = model.datacollector.get_model_vars_dataframe()

    if config.model_info:
        print("Model stats:")
        print(model_df.tail(config.steps_info))

    if config.population_info:
        print("Population stats:")
        print(model_df.describe())

    agent_df = model.datacollector.get_agent_vars_dataframe()
    if config.individual_info:
        print("\nAgent stats:")
        print(agent_df.describe())
