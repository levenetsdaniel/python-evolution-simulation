"""Offline CatBoost training and loading for spatial and classic simulations."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from config.sim_config import AdvisorTrainingConfig, SimConfig, validate_sim_config
from config.scenarios import SCENARIOS
from core.model import Model
from core.spatial import SpatialWorld

from .advisor import CatBoostAdvisor
from .trainer import train_advisor
from .training_buffer import TrainingBuffer


def _schema(config: SimConfig) -> dict:
    return {"version": 1, "mode": config.mode,
            "genome_labels": list(config.population.genome_labels),
            "min_temperature": config.environment.min_temperature,
            "max_temperature": config.environment.max_temperature}


def load_pretrained_advisor(path: str | Path, config: SimConfig) -> CatBoostAdvisor:
    """Load without fitting. Embedded schemas are checked when present."""
    if not str(path).strip():
        raise ValueError("advisor_path is required; train a model with python -m neural.advisor_training first")
    advisor = CatBoostAdvisor(verbose=False)
    advisor.load(path)
    metadata = advisor.model.get_metadata()
    if "evosim_schema" in metadata and json.loads(metadata["evosim_schema"]) != _schema(config):
        raise ValueError("Advisor is incompatible with simulation mode, genome order or temperature normalization")
    if len(advisor.model.feature_names_) != 22:
        raise ValueError("Advisor must accept the 22 simulation features")
    prediction = advisor.predict(np.zeros(22, dtype=np.float32))
    if prediction.shape != (7,) or not np.isfinite(prediction).all():
        raise ValueError("Advisor must predict seven finite genome traits")
    return advisor


def _collect_samples(simulation: SimConfig, seeds: list[int]) -> tuple[TrainingBuffer, list[dict]]:
    """Keep pending birth IDs local to each independent baseline run."""
    combined = None
    runs = []
    for seed in seeds:
        run_config = replace(simulation, seed=seed)
        world = (SpatialWorld.from_sim_config(run_config) if simulation.mode == "spatial"
                 else Model(run_config))
        buffer = TrainingBuffer(simulation.environment, max_food=max(1.0, world.environment.food_capacity))
        world.recorder = buffer
        if combined is None:
            combined = TrainingBuffer(simulation.environment, max_food=buffer.max_food)
        world.run(simulation.n_steps)
        combined.samples.extend(buffer.samples)
        runs.append({"seed": seed, "samples": len(buffer.samples),
                     "final_population": sum(agent.is_alive for agent in world.agents)})
    return (combined if combined is not None else TrainingBuffer(simulation.environment, max_food=1.0)), runs


def _evaluate_advisor(advisor: CatBoostAdvisor, buffer: TrainingBuffer) -> dict | None:
    """Measure held-out offspring reconstruction, not long-term survival."""
    if not buffer.samples:
        return None
    features, deltas, weights = buffer.to_numpy()
    before = features[:, :deltas.shape[1]]
    targets = np.clip(before + deltas, 0.0, 1.0)
    predictions = np.clip(advisor.model.predict(features), 0.0, 1.0)
    error = np.mean((predictions - targets) ** 2, axis=1)
    unchanged_error = np.mean((before - targets) ** 2, axis=1)
    return {"rmse": float(np.sqrt(error.mean())),
            "weighted_rmse": float(np.sqrt(np.dot(error, weights))),
            "unchanged_genome_rmse": float(np.sqrt(unchanged_error.mean())),
            "unchanged_genome_weighted_rmse": float(np.sqrt(np.dot(unchanged_error, weights)))}


def train_advisor_model(config: AdvisorTrainingConfig) -> tuple[CatBoostAdvisor, dict]:
    """Collect baseline birth outcomes and save a model for later inference."""
    simulation = config.simulation
    validate_sim_config(simulation)
    for name in ("training_runs", "iterations"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (not isinstance(config.validation_runs, int) or isinstance(config.validation_runs, bool)
            or config.validation_runs < 0):
        raise ValueError("validation_runs must be a non-negative integer")
    if not isinstance(config.balance_scenarios, bool):
        raise ValueError("balance_scenarios must be boolean")
    names = config.training_scenarios
    if not isinstance(names, list) or any(not isinstance(name, str) or name not in SCENARIOS for name in names):
        raise ValueError(f"training_scenarios must list known scenarios: {', '.join(SCENARIOS)}")
    if len(names) != len(set(names)):
        raise ValueError("training_scenarios must not contain duplicates")
    if config.save_path and not config.save_path.strip():
        raise ValueError("save_path must not be whitespace")
    save_path = config.save_path or f"artifacts/models/{simulation.mode}_advisor.cbm"
    if simulation.n_steps <= 0:
        raise ValueError("simulation.n_steps must be positive for advisor training")

    scenarios = [("configured", simulation)] if not names else [
        (name, replace(simulation, environment=replace(
            SCENARIOS[name], min_temperature=simulation.environment.min_temperature,
            max_temperature=simulation.environment.max_temperature))) for name in names]
    for _, run_config in scenarios:
        validate_sim_config(run_config)
    seeds = [simulation.seed + index for index in range(config.training_runs)]
    validation_seeds = [simulation.seed + config.training_runs + index
                        for index in range(config.validation_runs)]
    combined = TrainingBuffer(simulation.environment, max_food=max(1.0, simulation.environment.food_availability))
    scenario_reports = []
    scenario_weights = []
    validation_buffers = []
    for name, run_config in scenarios:
        print(f"Advisor data: scenario={name}, training seeds={seeds}, "
              f"validation seeds={validation_seeds}", flush=True)
        buffer, runs = _collect_samples(run_config, seeds)
        validation, validation_details = _collect_samples(run_config, validation_seeds)
        combined.samples.extend(buffer.samples)
        if buffer.samples:
            scenario_weights.append(buffer.to_numpy()[2])
        else:
            print(f"Warning: scenario={name} produced no evaluated training offspring", flush=True)
        validation_buffers.append(validation)
        scenario_reports.append({"name": name, "environment": asdict(run_config.environment),
                                 "samples": len(buffer.samples), "training_runs": runs,
                                 "validation_samples": len(validation.samples),
                                 "validation_runs": validation_details,
                                 "food_normalization": buffer.max_food})
        print(f"Advisor data: scenario={name}, training samples={len(buffer.samples)}, "
              f"validation samples={len(validation.samples)}", flush=True)
    if not combined.samples:
        raise ValueError("Advisor training produced no evaluated offspring; increase simulation.n_steps "
                         "or check reproduction settings")
    advisor = CatBoostAdvisor(iterations=config.iterations, random_seed=simulation.seed, verbose=False)
    weights = (np.concatenate(scenario_weights) / len(scenario_weights)
               if config.balance_scenarios else None)
    train_advisor(combined, advisor, sample_weights=weights)
    for row, validation in zip(scenario_reports, validation_buffers):
        row["validation"] = _evaluate_advisor(advisor, validation)
        row["training_weight"] = (1 / len(scenario_weights) if row["samples"] else 0.0) if weights is not None else None
    advisor.model.get_metadata()["evosim_schema"] = json.dumps(_schema(simulation))
    advisor.save(save_path)
    report = {"mode": simulation.mode, "training_seeds": seeds, "steps_per_run": simulation.n_steps,
              "training_scenarios": names, "validation_seeds": validation_seeds,
              "balance_scenarios": config.balance_scenarios, "scenarios": scenario_reports,
              "iterations": config.iterations, "samples": len(combined.samples),
              "food_normalization": (scenario_reports[0]["food_normalization"] if len({
                  row["food_normalization"] for row in scenario_reports}) == 1 else None),
              "schema": _schema(simulation),
              "model_path": save_path}
    Path(save_path + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return advisor, report


@hydra.main(version_base=None, config_path="../config", config_name="advisor_config")
def main(cfg: AdvisorTrainingConfig) -> None:
    """Train a standalone advisor with ``python -m neural.advisor_training``."""
    config = OmegaConf.to_object(cfg)
    _, report = train_advisor_model(config)
    print(f"Advisor saved: {report['model_path']}")
    print(report)


if __name__ == "__main__":
    from config.registry import register_configs

    register_configs()
    main()
