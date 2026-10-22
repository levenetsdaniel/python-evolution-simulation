"""The visualization reflects sampled world state without affecting simulation."""

from dataclasses import replace

import pytest

from config.sim_config import SpatialConfig
from core.spatial import SpatialWorld
from vis.spatial import (build_spatial_figure, capture_spatial_snapshot,
                         collect_spatial_snapshots, save_spatial_figure)


def test_snapshots_include_initial_sampled_and_final_steps():
    config = replace(SpatialConfig(), initial_population=4, max_population=8,
                     cluster_count=1, sources_per_cluster=2)
    world = SpatialWorld(config, seed=3)
    before = capture_spatial_snapshot(world)
    snapshots = collect_spatial_snapshots(world, steps=12, interval=5)
    assert [snapshot.step for snapshot in snapshots] == [0, 5, 10, 12]
    assert snapshots[0] == before
    assert snapshots[-1].step == world.step_count
    assert len(world.history) == 13
    assert snapshots[0].organisms != snapshots[-1].organisms
    later = collect_spatial_snapshots(world, steps=2, interval=5)
    assert [snapshot.step for snapshot in later] == [12, 14]


def test_plot_frames_and_html_reflect_world(tmp_path):
    world = SpatialWorld(replace(SpatialConfig(), initial_population=2,
                                 max_population=4, cluster_count=1, sources_per_cluster=1), seed=9)
    snapshots = collect_spatial_snapshots(world, steps=2, interval=1)
    figure = build_spatial_figure(world, snapshots)
    assert [frame.name for frame in figure.frames] == ["0", "1", "2"]
    assert list(figure.frames[-1].data[1].x) == [item[1] for item in snapshots[-1].organisms]
    assert list(figure.frames[-1].data[0].marker.color) == [
        item[2] / item[3] for item in snapshots[-1].food]
    assert list(figure.frames[-1].data[2].y)[-1] == world.history[-1]["PopulationSize"]
    assert figure.frames[-1].data[2].xaxis == "x2"
    output = save_spatial_figure(figure, tmp_path / "nested" / "world.html")
    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "Пространственная симуляция" in figure.layout.title.text
    assert 'data-control="stride"' in html
    assert 'data-control="step"' in html
    assert 'plotly_hover' in html
    assert '{plot_id}' not in html
    assert '"auto_play"' not in html
    assert not figure.layout.updatemenus
    for frame, snapshot in zip(figure.frames, snapshots):
        assert [row[4] for row in frame.data[1].customdata] == [
            item[6] for item in snapshot.organisms]


def test_snapshots_capture_actual_perception_without_changing_world():
    world = SpatialWorld(replace(SpatialConfig(), initial_population=3), seed=42)
    snapshot = capture_spatial_snapshot(world)
    for item in snapshot.organisms:
        organism = world.organisms[item[0]]
        assert item[6] == world.perception(organism)
    before = world.rng.bit_generator.state
    figure = build_spatial_figure(world, [snapshot])
    assert world.rng.bit_generator.state == before
    assert capture_spatial_snapshot(world) == snapshot
    assert "Радиус зрения" in figure.data[1].hovertemplate


def test_visualization_works_for_empty_world_and_zero_steps():
    world = SpatialWorld(replace(SpatialConfig(), initial_population=0,
                                 cluster_count=0), seed=1)
    snapshots = collect_spatial_snapshots(world, steps=0)
    figure = build_spatial_figure(world, snapshots)
    assert len(figure.frames) == 1
    assert len(figure.frames[0].data[1].x) == 0
    with pytest.raises(ValueError, match="positive"):
        collect_spatial_snapshots(world, steps=1, interval=0)
