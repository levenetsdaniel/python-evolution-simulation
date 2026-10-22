"""Offline fitting and fixed pretrained regressors in both simulation modes."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from config.scenarios import SCENARIOS
from config.sim_config import AdvisorTrainingConfig, ComparisonConfig, SimConfig, SpatialConfig
from neural.advisor import CatBoostAdvisor
from neural.advisor_training import load_pretrained_advisor, train_advisor_model
from neural.comparison_runner import ComparisonRunner


@pytest.fixture(scope="module", params=["spatial", "classic"])
def trained_advisor(request, tmp_path_factory):
    mode = request.param
    path = tmp_path_factory.mktemp(mode) / "advisor.cbm"
    config = AdvisorTrainingConfig(simulation=SimConfig(mode=mode, seed=42, n_steps=15),
                                   iterations=5, save_path=str(path))
    advisor, report = train_advisor_model(config)
    return config, advisor, report


def test_offline_training_saves_model_report_and_schema(trained_advisor):
    config, advisor, report = trained_advisor
    loaded = load_pretrained_advisor(config.save_path, config.simulation)
    features = np.zeros(22)
    np.testing.assert_allclose(advisor.predict(features), loaded.predict(features))
    assert loaded.predict(features).shape == (7,)
    assert report["samples"] > 0
    assert report["mode"] == config.simulation.mode
    assert report == json.loads(Path(config.save_path + ".json").read_text())
    assert json.loads(loaded.model.get_metadata()["evosim_schema"])["mode"] == config.simulation.mode


def test_pretrained_advisor_rejects_incompatible_schema(trained_advisor):
    config, _, _ = trained_advisor
    simulation = config.simulation
    other_mode = "classic" if simulation.mode == "spatial" else "spatial"
    changed = [replace(simulation, mode=other_mode),
               replace(simulation, population=replace(simulation.population,
                                                      genome_labels=simulation.population.genome_labels[::-1])),
               replace(simulation, environment=replace(simulation.environment, min_temperature=-40))]
    for incompatible in changed:
        with pytest.raises(ValueError, match="incompatible"):
            load_pretrained_advisor(config.save_path, incompatible)


def test_classic_comparison_never_fits_or_changes_advisor(tmp_path, monkeypatch):
    config = AdvisorTrainingConfig(simulation=SimConfig(mode="classic", n_steps=15),
                                   iterations=5, save_path=str(tmp_path / "classic.cbm"))
    train_advisor_model(config)

    def no_fit(*args, **kwargs):
        raise AssertionError("Comparison must never fit the advisor")

    monkeypatch.setattr(CatBoostAdvisor, "fit", no_fit)
    runner = ComparisonRunner(ComparisonConfig(simulation=config.simulation, advisor_path=config.save_path))
    advisor = runner.neural_branch.mutation_strategy._advisor
    before = advisor.model.get_leaf_values().copy()
    count = advisor.model.tree_count_
    runner.run(60)
    assert runner.baseline.recorder is None and runner.neural_branch.recorder is None
    np.testing.assert_array_equal(advisor.model.get_leaf_values(), before)
    assert advisor.model.tree_count_ == count


def test_offline_training_requires_evaluated_offspring(tmp_path):
    config = AdvisorTrainingConfig(simulation=SimConfig(n_steps=2,
                                   spatial=SpatialConfig(initial_population=0, cluster_count=0)),
                                   iterations=5, save_path=str(tmp_path / "empty.cbm"))
    with pytest.raises(ValueError, match="no evaluated offspring"):
        train_advisor_model(config)


def test_multiple_baseline_runs_have_separate_seeds(tmp_path):
    config = AdvisorTrainingConfig(simulation=SimConfig(mode="classic", seed=5, n_steps=15),
                                   training_runs=2, iterations=5, save_path=str(tmp_path / "two.cbm"))
    _, report = train_advisor_model(config)
    assert report["training_seeds"] == [5, 6]
    assert report["samples"] > 0


@pytest.mark.parametrize("mode", ["spatial", "classic"])
def test_multiple_scenarios_are_balanced_and_validated_on_separate_seeds(tmp_path, monkeypatch, mode):
    original_fit = CatBoostAdvisor.fit
    fitted = {}

    def capture_fit(self, features, targets, weights=None):
        fitted["features"] = features.copy()
        fitted["weights"] = weights.copy()
        original_fit(self, features, targets, weights)

    monkeypatch.setattr(CatBoostAdvisor, "fit", capture_fit)
    simulation = SimConfig(mode=mode, seed=8, n_steps=15)
    config = AdvisorTrainingConfig(simulation=simulation, training_scenarios=["stable", "famine"],
                                   training_runs=2, validation_runs=1, iterations=5,
                                   save_path=str(tmp_path / "mixed.cbm"))
    advisor, report = train_advisor_model(config)
    assert report["training_seeds"] == [8, 9]
    assert report["validation_seeds"] == [10]
    assert simulation.environment.temp_step == 0.05
    assert report["samples"] == len(fitted["features"])
    offset = 0
    for row in report["scenarios"]:
        count = row["samples"]
        assert count == sum(run["samples"] for run in row["training_runs"])
        assert row["environment"]["temp_step"] == SCENARIOS[row["name"]].temp_step
        assert row["training_weight"] == 0.5
        assert fitted["weights"][offset:offset + count].sum() == pytest.approx(0.5)
        assert row["validation_samples"] > 0
        assert all(np.isfinite(value) and value >= 0 for value in row["validation"].values())
        assert row["food_normalization"] == (6000.0 if mode == "spatial" else
                                             SCENARIOS[row["name"]].food_availability)
        offset += count
    loaded = load_pretrained_advisor(config.save_path, replace(simulation, environment=SCENARIOS["famine"]))
    np.testing.assert_allclose(advisor.predict(np.zeros(22)), loaded.predict(np.zeros(22)))


def test_validation_can_be_disabled(tmp_path):
    config = AdvisorTrainingConfig(simulation=SimConfig(mode="classic", n_steps=15),
                                   validation_runs=0, iterations=5, save_path=str(tmp_path / "no_validation.cbm"))
    _, report = train_advisor_model(config)
    assert report["validation_seeds"] == []
    assert report["scenarios"][0]["validation_samples"] == 0
    assert report["scenarios"][0]["validation"] is None


def test_training_entrypoint_saves_regressor(tmp_path, capsys):
    from train_advisor import main

    path = tmp_path / "entrypoint.cbm"
    config = AdvisorTrainingConfig(simulation=SimConfig(mode="classic", n_steps=15),
                                   iterations=5, save_path=str(path))
    main.__wrapped__(OmegaConf.structured(config))
    assert path.exists()
    assert "Advisor saved:" in capsys.readouterr().out


@pytest.mark.parametrize("changes", [{"training_runs": 0}, {"iterations": 0}, {"save_path": " "},
                                     {"validation_runs": -1}, {"validation_runs": True},
                                     {"balance_scenarios": "yes"},
                                     {"training_scenarios": ["missing"]},
                                     {"training_scenarios": ["stable", "stable"]},
                                     {"training_scenarios": "stable"}])
def test_offline_training_rejects_invalid_settings(changes):
    with pytest.raises(ValueError):
        train_advisor_model(AdvisorTrainingConfig(**changes))
