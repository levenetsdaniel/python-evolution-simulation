"""Run the comparison runner and collect per-agent snapshots for animated visualization."""

from neural.comparison_runner import ComparisonRunner
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


def run_with_snapshots(runner: ComparisonRunner, n_steps: int) -> tuple[list[dict], list[dict]]:
    """Run the comparison runner and collect baseline/neural snapshots after each step."""
    baseline_snaps: list[dict] = []
    neural_snaps: list[dict] = []

    for _ in range(n_steps):
        if is_extinct(runner.baseline) and is_extinct(runner.neural_branch):
            break

        runner.step()

        baseline_snaps.append(_snapshot(runner.baseline))
        neural_snaps.append(_snapshot(runner.neural_branch))

    return baseline_snaps, neural_snaps
