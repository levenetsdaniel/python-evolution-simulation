"""Smoke tests for entrypoints and architecture boundaries."""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

import comparison_run
import main as main_entry
from config.sim_config import ComparisonConfig, SimConfig
from core.model import Model


def test_core_model_runs_without_recorder():
    model = Model(SimConfig(n_steps=2, seed=0))

    model.run(2)

    model_df = model.datacollector.get_model_vars_dataframe()
    assert len(model_df) > 0


def test_main_run_mode_executes_model_and_prints_optional_summaries(monkeypatch, capsys):
    calls: dict[str, object] = {"run_steps": None}

    class StubCollector:
        def get_model_vars_dataframe(self):
            return pd.DataFrame([{"Step": 1, "AvgFitness": 0.5}])

        def get_agent_vars_dataframe(self):
            return pd.DataFrame([{"Fitness": 0.5, "Age": 2}])

    class StubModel:
        def __init__(self, cfg):
            calls["config"] = cfg
            self.datacollector = StubCollector()

        def run(self, n_steps):
            calls["run_steps"] = n_steps

    monkeypatch.setattr(main_entry, "Model", StubModel)

    cfg = OmegaConf.structured(
        SimConfig(
            n_steps=12,
            model_info=True,
            population_info=True,
            individual_info=True,
        )
    )

    main_entry.main.__wrapped__(cfg)

    out = capsys.readouterr().out
    assert calls["run_steps"] == 12
    assert "Model stats:" in out
    assert "Population stats:" in out
    assert "Agent stats:" in out


def test_main_view_mode_builds_and_saves_figures(monkeypatch):
    calls: dict[str, object] = {}

    class StubModel:
        def __init__(self, cfg):
            calls["config"] = cfg

        def run(self, n_steps):
            raise AssertionError("run() should not be called in view mode")

    def fake_collect_history(model, n_steps):
        calls["collect_history"] = (model, n_steps)
        return pd.DataFrame([{"Generation": 1}])

    def fake_build_all_figures(history):
        calls["build_figures"] = history
        return {"figure": object()}

    def fake_save_figures(figures, output_dir="artifacts"):
        calls["save_figures"] = (figures, output_dir)

    monkeypatch.setattr(main_entry, "Model", StubModel)
    monkeypatch.setattr(main_entry, "collect_history", fake_collect_history)
    monkeypatch.setattr(main_entry, "build_all_figures", fake_build_all_figures)
    monkeypatch.setattr(main_entry, "save_figures", fake_save_figures)

    cfg = OmegaConf.structured(SimConfig(n_steps=8, view=True))
    main_entry.main.__wrapped__(cfg)

    assert calls["collect_history"][1] == 8
    assert calls["save_figures"][1] == "artifacts"


def test_comparison_run_executes_runner(monkeypatch):
    calls: dict[str, object] = {"run_steps": None}

    class StubRunner:
        def __init__(self, cfg):
            calls["config"] = cfg

        def run(self, n_steps):
            calls["run_steps"] = n_steps

    monkeypatch.setattr(comparison_run, "ComparisonRunner", StubRunner)

    cfg = OmegaConf.structured(ComparisonConfig())
    cfg.simulation.n_steps = 9
    comparison_run.main.__wrapped__(cfg)

    assert calls["run_steps"] == 9


def test_comparison_run_view_mode_builds_dashboard(monkeypatch):
    calls: dict[str, object] = {}

    class StubRunner:
        def __init__(self, cfg):
            raise AssertionError("runner should not be constructed in view mode")

    def fake_build_dashboard(cfg):
        calls["dashboard_cfg"] = cfg

    monkeypatch.setattr(comparison_run, "ComparisonRunner", StubRunner)
    monkeypatch.setattr(comparison_run, "build_dashboard", fake_build_dashboard)

    cfg = OmegaConf.structured(ComparisonConfig(view=True))
    comparison_run.main.__wrapped__(cfg)

    assert calls["dashboard_cfg"].view is True
    assert calls["dashboard_cfg"].simulation.n_steps == 600


def test_core_modules_do_not_import_neural_package():
    core_dir = Path(__file__).resolve().parent.parent / "core"

    for path in core_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from neural" not in text
        assert "import neural" not in text
