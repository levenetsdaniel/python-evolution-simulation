from pathlib import Path

from config.sim_config import SimConfig
from core.model import Model
from vis.history import collect_history
from vis.plots import build_all_figures, save_figures


def run_visualization_demo(n_steps: int = 600, seed: int = 0, output_dir: str | Path = "graphics") -> None:
    """
    Run one simulation and save all basic visualization figures.
    """
    config = SimConfig(seed=seed, n_steps=n_steps, debug=False, record=False)
    model = Model(config)

    history_df = collect_history(model, n_steps=n_steps)
    figures = build_all_figures(history_df)
    save_figures(figures, output_dir=output_dir)

    print("visualization OK")
    print("saved to:", Path(output_dir))


if __name__ == "__main__":
    run_visualization_demo()