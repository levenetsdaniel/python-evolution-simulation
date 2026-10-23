"""Build and save the comparison dashboard: baseline vs neural-guided."""

from pathlib import Path

from config.output_paths import COMPARISON_DASHBOARD_DIR
from config.sim_config import ComparisonConfig
from neural.comparison_runner import ComparisonRunner
from .plots import build_animated_scatter, build_comparison_table
from .snapshots import run_with_snapshots


def build_dashboard(
    config: ComparisonConfig,
    out_dir: str | Path = COMPARISON_DASHBOARD_DIR,
) -> Path:
    runner = ComparisonRunner(config)

    baseline_snaps, neural_snaps = run_with_snapshots(
        runner,
        n_steps=config.simulation.n_steps,
    )

    history = runner.history()

    baseline_hist = history[history["branch"] == "baseline"].reset_index(drop=True)
    neural_hist = history[history["branch"] == "neural"].reset_index(drop=True)

    scatter_title = f"Animated scatter ({config.x_trait} x {config.y_trait})"

    figures = {
        scatter_title: build_animated_scatter(
            baseline_snaps,
            neural_snaps,
            x_trait=config.x_trait,
            y_trait=config.y_trait,
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
    cfg = ComparisonConfig()
    path = build_dashboard(cfg)
    print(f"Dashboard saved to {path.resolve()}")
