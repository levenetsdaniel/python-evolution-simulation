from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _build_group_figure(
    history_df: pd.DataFrame,
    title: str,
    left_cols: list[str],
    right_cols: list[str] | None = None,
    x_col: str = "Generation",
    left_axis_title: str = "Population traits",
    right_axis_title: str = "Environment",
) -> go.Figure:
    """
    Build one grouped Plotly figure.

    Left-axis metrics are plotted together. Optional right-axis metrics are
    plotted on a secondary y-axis.
    """
    required = {x_col, *left_cols, *(right_cols or [])}
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    use_secondary = bool(right_cols)

    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

    for col in left_cols:
        fig.add_trace(
            go.Scatter(
                x=history_df[x_col],
                y=history_df[col],
                mode="lines",
                name=col,
            ),
            secondary_y=False,
        )

    if right_cols:
        for col in right_cols:
            fig.add_trace(
                go.Scatter(
                    x=history_df[x_col],
                    y=history_df[col],
                    mode="lines",
                    name=col,
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=550,
    )

    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text=left_axis_title, secondary_y=False)

    if use_secondary:
        fig.update_yaxes(title_text=right_axis_title, secondary_y=True)

    return fig


def build_all_figures(history_df: pd.DataFrame) -> dict[str, go.Figure]:
    """
    Build all basic visualization figures required for the first iteration.

    """
    figures = {
        "temperature_traits": _build_group_figure(
            history_df=history_df,
            title="Thermal Adaptation",
            left_cols=["AvgHeatResistance", "AvgColdResistance"],
            right_cols=["TemperatureStart"],
            left_axis_title="Resistance",
            right_axis_title="Temperature (°C)",
        ),
        "body_traits_food": _build_group_figure(
            history_df=history_df,
            title="Body Traits vs Food",
            left_cols=["AvgSize", "AvgSpeed", "AvgAggressiveness", "AvgResilience", "AvgMetabolicRate"],
            right_cols=["FoodStart"],
            left_axis_title="Trait value",
            right_axis_title="Food",
        ),
        "resilience_hazard": _build_group_figure(
            history_df=history_df,
            title="Resilience vs Hazard",
            left_cols=["AvgResilience"],
            right_cols=["HazardStart"],
            left_axis_title="Resilience",
            right_axis_title="Hazard",
        ),
        "fitness_population": _build_group_figure(
            history_df=history_df,
            title="Fitness vs Population Size",
            left_cols=["AvgFitness"],
            right_cols=["PopulationSize"],
            left_axis_title="Fitness",
            right_axis_title="Population size",
        ),
    }
    return figures

def save_figures(figures: dict[str, go.Figure], output_dir: str | Path = "artifacts") -> None:
    """Save all figures as separate HTML files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig.write_html(str(output_dir / f"{name}.html"))