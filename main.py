"""
Command-line entry point for running the simulation with baseline and neural guided line.

Initializes configuration from CLI, runs two models.
"""

from config.cli import parse_args
from core.engine import Engine


def main():
    config, _ = parse_args()

    simulation = Engine(config)

    simulation.run(config.n_steps)

if __name__ == "__main__":
    main()
