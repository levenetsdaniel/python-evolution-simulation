from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

def build_animated_scatter(
    baseline_snaps: list[dict],
    neural_snaps: list[dict],
    x_trait: str = "heat_resistance",
    y_trait: str = "cold_resistance",
) -> go.Figure:
    """Build an animated scatter plot comparing baseline and neural populations."""
    n_gens = min(len(baseline_snaps), len(neural_snaps))

    if n_gens == 0:
        return go.Figure()

    _validate_snapshot_trait(baseline_snaps[0], x_trait)
    _validate_snapshot_trait(baseline_snaps[0], y_trait)
    _validate_snapshot_trait(neural_snaps[0], x_trait)
    _validate_snapshot_trait(neural_snaps[0], y_trait)

    frames = []

    for i in range(n_gens):
        generation = str(i + 1)

        frames.append(
            go.Frame(
                data=[
                    _make_scatter(
                        baseline_snaps[i],
                        x_trait,
                        y_trait,
                        "Baseline",
                        "Blues",
                    ),
                    _make_scatter(
                        neural_snaps[i],
                        x_trait,
                        y_trait,
                        "Neural",
                        "Reds",
                    ),
                ],
                name=generation,
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=f"Population evolution: {x_trait} × {y_trait}",
        xaxis=dict(title=x_trait, range=[0.0, 1.0]),
        yaxis=dict(title=y_trait, range=[0.0, 1.0]),
        template="plotly_white",
        height=600,
        hovermode="closest",
        updatemenus=[_play_pause()],
        sliders=[_gen_slider(frames)],
    )

    return fig

def build_comparison_table(
    baseline_history,
    neural_history,
    tail_window: int = 50,
) -> go.Figure:
    """Build a summary table comparing baseline and neural-guided histories."""
    if len(baseline_history) == 0 or len(neural_history) == 0:
        return go.Figure()

    metrics = [
        (
            "Финальная численность",
            _last_value(baseline_history, "PopulationSize"),
            _last_value(neural_history, "PopulationSize"),
            ".3f",
        ),
        (
            "Средний fitness (хвост)",
            _tail_mean(baseline_history, "AvgFitness", tail_window),
            _tail_mean(neural_history, "AvgFitness", tail_window),
            ".3f",
        ),
        (
            "Скорость адаптации",
            _adaptation_speed(baseline_history),
            _adaptation_speed(neural_history),
            ".5f",
        ),
        (
            "Прожито шагов",
            float(len(baseline_history)),
            float(len(neural_history)),
            ".0f",
        ),
    ]

    names = [metric[0] for metric in metrics]
    b_vals = [metric[1] for metric in metrics]
    n_vals = [metric[2] for metric in metrics]
    formats = [metric[3] for metric in metrics]

    b_colors = []
    n_colors = []

    for b_val, n_val in zip(b_vals, n_vals):
        b_color, n_color = _winner_colors(b_val, n_val)
        b_colors.append(b_color)
        n_colors.append(n_color)

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["Метрика", "Baseline", "Neural-guided"],
                fill_color="lightgrey",
                align="left",
                font=dict(size=14),
            ),
            cells=dict(
                values=[
                    names,
                    [format(value, fmt) for value, fmt in zip(b_vals, formats)],
                    [format(value, fmt) for value, fmt in zip(n_vals, formats)],
                ],
                fill_color=[
                    ["white"] * len(names),
                    b_colors,
                    n_colors,
                ],
                align="left",
                font=dict(size=12),
                height=28,
            ),
        )
    )

    fig.update_layout(title="Baseline vs Neural-guided")

    return fig


def _last_value(history, column: str) -> float:
    """Return the last finite value from a history column."""
    values = history[column].dropna()

    if len(values) == 0:
        return 0.0

    return float(values.iloc[-1])


def _tail_mean(history, column: str, tail_window: int) -> float:
    """Return the mean of the last values from a history column."""
    values = history[column].dropna().tail(tail_window)

    if len(values) == 0:
        return 0.0

    return float(values.mean())


def _adaptation_speed(history) -> float:
    """Estimate adaptation speed as a linear trend of AvgFitness over time."""
    fitness = history["AvgFitness"].dropna()

    if len(fitness) < 2:
        return 0.0

    if "Generation" in history:
        x = history.loc[fitness.index, "Generation"].to_numpy(dtype=float)
    else:
        x = np.arange(len(fitness), dtype=float)

    y = fitness.to_numpy(dtype=float)

    slope, _ = np.polyfit(x, y, deg=1)

    return float(slope)


def _winner_colors(baseline_value: float, neural_value: float) -> tuple[str, str]:
    """Return table cell colors highlighting the better value."""
    if np.isclose(baseline_value, neural_value):
        return "white", "white"

    if baseline_value > neural_value:
        return "lightgreen", "white"

    return "white", "lightgreen"

def _make_scatter(
    snap: dict,
    x_trait: str,
    y_trait: str,
    name: str,
    scale: str,
) -> go.Scatter:
    """Build one scatter trace for one population snapshot."""
    return go.Scatter(
        x=snap[x_trait],
        y=snap[y_trait],
        mode="markers",
        marker=dict(
            color=snap["fitness"],
            colorscale=scale,
            cmin=0.0,
            cmax=1.0,
            size=8,
            line=dict(width=0.5, color="black"),
            showscale=False,
        ),
        name=name,
    )


def _validate_snapshot_trait(snap: dict, trait: str) -> None:
    """Raise a clear error if a requested trait is missing from a snapshot."""
    if trait not in snap:
        raise ValueError(f"Unknown trait for animated scatter: {trait}")


def _play_pause() -> dict:
    """Build Plotly play and pause animation controls."""
    return {
        "type": "buttons",
        "direction": "left",
        "x": 0.1,
        "y": -0.05,
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 200, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    },
                ],
            },
        ],
    }


def _gen_slider(frames: list) -> dict:
    """Build a Plotly generation slider for animation frames."""
    return {
        "active": 0,
        "x": 0.1,
        "y": 0,
        "len": 0.85,
        "currentvalue": {
            "prefix": "Поколение: ",
            "visible": True,
            "xanchor": "right",
        },
        "steps": [
            {
                "method": "animate",
                "label": frame.name,
                "args": [
                    [frame.name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    },
                ],
            }
            for frame in frames
        ],
    }