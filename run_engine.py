from config.cli import parse_args
from core.engine import Engine


def main():
    config, _ = parse_args()

    simulation = Engine(config)

    simulation.run(config.n_steps)

    df = simulation.history()
    df.describe()


if __name__ == "__main__":
    main()
