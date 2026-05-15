"""Build and save the comparison dashboard: baseline vs neural-guided."""

from pathlib import Path

from config.sim_config import SimConfig
from core.engine import Engine
from vis.plots import build_animated_scatter, build_comparison_table
from vis.snapshots import run_with_snapshots


def build_dashboard(config: SimConfig, out_dir: str | Path = "dashboard_output") -> Path:
    engine = Engine(config)

    baseline_snaps, neural_snaps = run_with_snapshots(
        engine,
        n_steps=config.n_steps,
    )

    history = engine.history()

    baseline_hist = history[history["branch"] == "baseline"].reset_index(drop=True)
    neural_hist = history[history["branch"] == "neural"].reset_index(drop=True)

    figures = {
        "Animated scatter (heat × cold)": build_animated_scatter(
            baseline_snaps,
            neural_snaps,
        ),
        "Comparison table": build_comparison_table(
            baseline_hist,
            neural_hist,
        ),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    html_path = out / "dashboard.html"
    _write_combined_html(figures, html_path)

    return html_path


def _write_combined_html(figures: dict, path: Path) -> None:
    parts = [
        '<html><head><meta charset="utf-8"><title>Evolution Dashboard</title></head><body>',
        '<h1>Эволюция: Baseline vs Neural-guided</h1>',
    ]

    for name, fig in figures.items():
        parts.append(f"<h2>{name}</h2>")
        parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    cfg = SimConfig()
    path = build_dashboard(cfg)
    print(f"Dashboard saved to {path.resolve()}")