"""Run Engine and collect per-agent snapshots for animated visualization."""

from core.engine import Engine
from vis.history import is_extinct


GENE_KEYS = [
    "heat_resistance",
    "cold_resistance",
    "metabolic_rate",
    "resilience",
    "size",
    "speed",
    "aggressiveness",
]


def _snapshot(model) -> dict:
    """Collect one per-agent snapshot from a model."""
    snapshot = {key: [] for key in GENE_KEYS}
    snapshot["fitness"] = []

    for agent in model.agents:
        for key in GENE_KEYS:
            snapshot[key].append(float(agent.genes_map[key]))

        fitness = float(agent.fitness) if agent.fitness is not None else float("nan")
        snapshot["fitness"].append(fitness)

    return snapshot


def run_with_snapshots(engine: Engine, n_steps: int) -> tuple[list[dict], list[dict]]:
    """Run Engine and collect baseline and neural-line snapshots after each step."""
    baseline_snaps: list[dict] = []
    neural_snaps: list[dict] = []

    for _ in range(n_steps):
        if is_extinct(engine.baseline) and is_extinct(engine.neuralline):
            break

        engine.step()

        baseline_snaps.append(_snapshot(engine.baseline))
        neural_snaps.append(_snapshot(engine.neuralline))

    return baseline_snaps, neural_snaps