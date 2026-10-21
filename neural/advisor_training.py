"""Offline CatBoost training and loading for spatial and classic simulations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from config.sim_config import AdvisorTrainingConfig, SimConfig, validate_sim_config
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
        raise ValueError("advisor_path is required; train a model with train_advisor.py first")
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


def train_advisor_model(config: AdvisorTrainingConfig) -> tuple[CatBoostAdvisor, dict]:
    """Collect baseline birth outcomes and save a model for later inference."""
    simulation = config.simulation
    validate_sim_config(simulation)
    for name in ("training_runs", "iterations"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.save_path and not config.save_path.strip():
        raise ValueError("save_path must not be whitespace")
    save_path = config.save_path or f"artifacts/models/{simulation.mode}_advisor.cbm"
    if simulation.n_steps <= 0:
        raise ValueError("simulation.n_steps must be positive for advisor training")

    food_capacity = (simulation.spatial.cluster_count * simulation.spatial.sources_per_cluster
                     * simulation.spatial.food_capacity if simulation.mode == "spatial"
                     else simulation.environment.food_availability)
    combined = TrainingBuffer(simulation.environment, max_food=max(1.0, food_capacity))
    seeds = [simulation.seed + index for index in range(config.training_runs)]
    for seed in seeds:
        buffer = TrainingBuffer(simulation.environment, max_food=max(1.0, food_capacity))
        run_config = replace(simulation, seed=seed)
        world = (SpatialWorld.from_sim_config(run_config, recorder=buffer) if simulation.mode == "spatial"
                 else Model(run_config, recorder=buffer))
        world.run(simulation.n_steps)
        # IDs restart in each world. Only completed samples cross run boundaries.
        combined.samples.extend(buffer.samples)
    if not combined.samples:
        raise ValueError("Advisor training produced no evaluated offspring; increase simulation.n_steps "
                         "or check reproduction settings")
    advisor = CatBoostAdvisor(iterations=config.iterations, random_seed=simulation.seed, verbose=False)
    train_advisor(combined, advisor)
    advisor.model.get_metadata()["evosim_schema"] = json.dumps(_schema(simulation))
    advisor.save(save_path)
    report = {"mode": simulation.mode, "training_seeds": seeds, "steps_per_run": simulation.n_steps,
              "iterations": config.iterations, "samples": len(combined.samples),
              "food_normalization": max(1.0, food_capacity), "schema": _schema(simulation),
              "model_path": save_path}
    Path(save_path + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return advisor, report
