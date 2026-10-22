"""Interactive HTML visualization for the headless spatial simulation."""

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.spatial import SpatialWorld


@dataclass(frozen=True)
class SpatialSnapshot:
    step: int
    organisms: tuple[tuple[int, float, float, float, int, float | None, float], ...]
    food: tuple[tuple[float, float, float, float], ...]


def capture_spatial_snapshot(world: SpatialWorld) -> SpatialSnapshot:
    """Copy only display data, so later world steps cannot alter earlier frames."""
    return SpatialSnapshot(
        step=world.step_count,
        organisms=tuple((organism.unique_id, organism.state.x, organism.state.y,
                         organism.energy, organism.age, organism.fitness, world.perception(organism))
                        for _, organism in sorted(world.organisms.items())),
        food=tuple((source.x, source.y, source.stock, source.capacity)
                   for source in world.food_sources),
    )


def collect_spatial_snapshots(world: SpatialWorld, steps: int, interval: int = 5) -> list[SpatialSnapshot]:
    """Run the world and capture step zero, sampled steps, and the final step."""
    if steps < 0 or interval < 1:
        raise ValueError("steps must be non-negative and interval must be positive")
    snapshots = [capture_spatial_snapshot(world)]
    final_step = world.step_count + steps
    for _ in range(steps):
        world.step()
        if world.step_count % interval == 0 or world.step_count == final_step:
            snapshots.append(capture_spatial_snapshot(world))
    return snapshots


def _food_trace(snapshot: SpatialSnapshot) -> go.Scatter:
    return go.Scatter(
        xaxis="x", yaxis="y",
        x=[item[0] for item in snapshot.food],
        y=[item[1] for item in snapshot.food],
        mode="markers",
        name="Пища",
        marker={"size": [6 + 14 * item[2] / item[3] for item in snapshot.food],
                "color": [item[2] / item[3] for item in snapshot.food],
                "colorscale": "YlGn", "cmin": 0, "cmax": 1,
                "line": {"color": "#477549", "width": 0.5}},
        customdata=[[item[2], item[3]] for item in snapshot.food],
        hovertemplate="Пища: %{customdata[0]:.1f}/%{customdata[1]:.1f}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
    )


def _organism_trace(snapshot: SpatialSnapshot, max_energy: float) -> go.Scatter:
    return go.Scatter(
        xaxis="x", yaxis="y",
        x=[item[1] for item in snapshot.organisms],
        y=[item[2] for item in snapshot.organisms],
        mode="markers",
        name="Организмы",
        marker={"size": 7, "color": [item[3] for item in snapshot.organisms],
                "colorscale": "Turbo", "cmin": 0, "cmax": max_energy,
                "colorbar": {"title": "Энергия"},
                "line": {"color": "#182b46", "width": 0.4}},
        customdata=[[item[0], item[3], item[4], item[5], item[6]] for item in snapshot.organisms],
        hovertemplate="ID %{customdata[0]}<br>Энергия %{customdata[1]:.1f}<br>Возраст %{customdata[2]}<br>Fitness %{customdata[3]:.3f}<br>Радиус зрения %{customdata[4]:.1f}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
    )


def _history_traces(world: SpatialWorld, step: int) -> tuple[go.Scatter, go.Scatter]:
    rows = [row for row in world.history if row["TimeStep"] <= step]
    return (
        go.Scatter(xaxis="x2", yaxis="y2",
                   x=[row["TimeStep"] for row in rows], y=[row["PopulationSize"] for row in rows],
                   mode="lines", name="Численность", line={"color": "#2b65ad"}),
        go.Scatter(xaxis="x2", yaxis="y3",
                   x=[row["TimeStep"] for row in rows], y=[row["FoodAvailable"] for row in rows],
                   mode="lines", name="Запас пищи", line={"color": "#4a9b55"}),
    )


def build_spatial_figure(world: SpatialWorld, snapshots: list[SpatialSnapshot]) -> go.Figure:
    """Build world frames; the HTML exporter adds playback and hover controls."""
    if not snapshots:
        raise ValueError("at least one spatial snapshot is required")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.1,
                        specs=[[{}], [{"secondary_y": True}]])
    first = snapshots[0]
    population, food_history = _history_traces(world, first.step)
    fig.add_trace(_food_trace(first), row=1, col=1)
    fig.add_trace(_organism_trace(first, world.config.max_energy), row=1, col=1)
    fig.add_trace(population, row=2, col=1, secondary_y=False)
    fig.add_trace(food_history, row=2, col=1, secondary_y=True)
    frames = []
    for snapshot in snapshots:
        population, food_history = _history_traces(world, snapshot.step)
        frames.append(go.Frame(
            name=str(snapshot.step),
            data=[_food_trace(snapshot), _organism_trace(snapshot, world.config.max_energy),
                  population, food_history],
            traces=[0, 1, 2, 3],
            layout=go.Layout(title_text=f"Пространственная симуляция · шаг {snapshot.step} · "
                                        f"организмов {len(snapshot.organisms)} · "
                                        f"температура {world.history[snapshot.step]['Temperature']:.1f} · "
                                        f"опасность {world.history[snapshot.step]['HazardLevel']:.2f}"),
        ))
    fig.frames = frames
    fig.update_xaxes(title_text="X", range=[0, world.width], row=1, col=1)
    fig.update_yaxes(title_text="Y", range=[0, world.height], scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="Шаг", range=[0, max(1, world.step_count)], row=2, col=1)
    fig.update_yaxes(title_text="Организмы", range=[0, max(1, max(row["PopulationSize"] for row in world.history) * 1.1)],
                     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Пища", range=[0, max(1, max(row["FoodAvailable"] for row in world.history) * 1.1)],
                     row=2, col=1, secondary_y=True)
    fig.update_layout(
        title=frames[0].layout.title.text,
        template="plotly_white",
        height=850,
        legend={"orientation": "h", "y": 1.02},
        margin={"l": 65, "r": 85, "t": 90, "b": 55},
    )
    return fig


def save_spatial_figure(figure: go.Figure, path: str | Path) -> Path:
    """Write a self-contained HTML file that opens without a Python server."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("spatial.js").read_text(encoding="utf-8")
    figure.write_html(str(output), include_plotlyjs=True, full_html=True,
                      post_script=script, auto_play=False)
    return output
