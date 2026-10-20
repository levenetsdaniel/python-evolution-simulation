"""Guided mutation integration with the spatial life cycle."""

from dataclasses import replace

import numpy as np
import pytest

from config.sim_config import IndividualConfig, SimConfig, SpatialConfig, SpatialGuidanceConfig
from core.enums import Gender
from core.model import Model
from core.spatial import MoveAction, SpatialWorld
from neural.guided_mutation import GuidedMutation
from neural.spatial_guidance import build_guided_spatial_world
from neural.training_buffer import TrainingBuffer
from vis.spatial import collect_spatial_snapshots


def parent_world(*, strategy=None, recorder=None):
    config = SpatialConfig(initial_population=0, max_population=3, cluster_count=0,
                           reproduction_rate=1, min_reproduction_age=0,
                           reproduction_energy=50, offspring_energy=20)
    costs = IndividualConfig(basal_energy_cost=0, speed_energy_cost=0,
                             aggressiveness_energy_cost=0,
                             fitness_death_threshold=0, fitness_death_prob_coef=0,
                             age_scale=1e9)
    world = SpatialWorld(config, seed=17, individual_config=costs,
                         mutation_strategy=strategy, recorder=recorder)
    genome = np.full(7, 0.5)
    mother = world.add_organism(genome, 10, 10, energy=150)
    father = world.add_organism(genome, 11, 10, energy=100)
    mother.gender, father.gender = Gender.FEMALE, Gender.MALE
    return world, mother, father


def test_spatial_birth_and_fitness_complete_training_sample():
    buffer = TrainingBuffer(max_food=1)
    world, mother, father = parent_world(recorder=buffer)
    world.step({mother.unique_id: MoveAction(0, 0), father.unique_id: MoveAction(0, 0)})
    child = next(agent for agent in world.organisms.values() if agent.parent_ids)
    assert child.unique_id in buffer._pending
    world.step({key: MoveAction(0, 0) for key in world.organisms})
    assert child.unique_id not in buffer._pending
    assert len(buffer.samples) == 1
    features, deltas, weights = buffer.to_numpy()
    assert features.shape == (1, 22)
    assert deltas.shape == (1, 7)
    assert weights == pytest.approx([1])


def test_guided_mutation_receives_spatial_environment_features():
    class AdvisorStub:
        is_trained = True

        def predict(self, features):
            self.features = features.copy()
            return np.ones(7, dtype=np.float32)

    advisor = AdvisorStub()
    world, mother, father = parent_world(strategy=GuidedMutation(advisor, shift_strength=1))
    world.step({mother.unique_id: MoveAction(0, 0), father.unique_id: MoveAction(0, 0)})
    child = next(agent for agent in world.organisms.values() if agent.parent_ids)
    assert np.all(child.genome == 1)
    assert advisor.features.shape == (22,)
    assert advisor.features[7] == pytest.approx((world.history[-1]["Temperature"] + 30) / 80)
    assert advisor.features[8] == 0  # No food sources in this controlled world.


def test_guided_mutation_still_accepts_classic_model():
    class AdvisorStub:
        is_trained = True

        def predict(self, features):
            self.features = features.copy()
            return np.ones(7, dtype=np.float32)

    model = Model(SimConfig(mode="classic", seed=17))
    mother, father = list(model.agents)[:2]
    mother.fitness = father.fitness = 0.5
    advisor = AdvisorStub()
    genome, _ = GuidedMutation(advisor, shift_strength=1).mutate(
        np.full(7, 0.5), mother, father, model)
    assert np.all(genome == 1)
    assert advisor.features.shape == (22,)
    assert advisor.features[8] == pytest.approx(1)


def test_guided_spatial_world_trains_saves_and_loads(tmp_path):
    path = tmp_path / "spatial_advisor.cbm"
    config = SimConfig(n_steps=5, seed=42,
                       spatial_guidance=SpatialGuidanceConfig(enabled=True, training_steps=15,
                                                              iterations=5, save_path=str(path)))
    trained = build_guided_spatial_world(config)
    assert path.exists()
    assert trained.step_count == 0
    assert isinstance(trained.mutation_strategy, GuidedMutation)
    assert trained.mutation_strategy._advisor.is_trained
    trained.run(config.n_steps)

    loaded_config = replace(config, spatial_guidance=replace(config.spatial_guidance,
                                                              advisor_path=str(path), save_path="",
                                                              training_steps=1))
    loaded = build_guided_spatial_world(loaded_config)
    loaded.run(config.n_steps)
    assert trained.history == loaded.history


def test_guided_spatial_world_retrains_on_new_offspring_and_saves(tmp_path):
    path = tmp_path / "updated.cbm"
    config = SimConfig(seed=42, spatial_guidance=SpatialGuidanceConfig(
        enabled=True, training_steps=15, retrain_steps=2,
        iterations=5, save_path=str(path)))
    world = build_guided_spatial_world(config)
    initial_trees = world.advisor.model.tree_count_
    initial_file = path.read_bytes()

    snapshots = collect_spatial_snapshots(world, 6, 2)

    assert [snapshot.step for snapshot in snapshots] == [0, 2, 4, 6]
    assert world.retraining_steps == [4, 6]
    assert world.advisor.model.tree_count_ == initial_trees + 2 * config.spatial_guidance.iterations
    assert path.read_bytes() != initial_file
    assert world.recorder.samples == []


def test_guided_spatial_world_skips_empty_retraining_intervals(tmp_path):
    path = tmp_path / "loaded.cbm"
    trained = build_guided_spatial_world(SimConfig(spatial_guidance=SpatialGuidanceConfig(
        enabled=True, training_steps=15, iterations=5, save_path=str(path))))
    config = SimConfig(spatial=SpatialConfig(initial_population=0, cluster_count=0),
                       spatial_guidance=SpatialGuidanceConfig(
                           enabled=True, advisor_path=str(path), retrain_steps=1, iterations=5))
    world = build_guided_spatial_world(config)
    initial_trees = world.advisor.model.tree_count_

    world.run(3)

    assert world.retraining_steps == []
    assert world.advisor.model.tree_count_ == initial_trees


def test_loaded_spatial_advisor_continues_training(tmp_path):
    path = tmp_path / "advisor.cbm"
    build_guided_spatial_world(SimConfig(spatial_guidance=SpatialGuidanceConfig(
        enabled=True, training_steps=15, iterations=5, save_path=str(path))))
    config = SimConfig(seed=42, spatial_guidance=SpatialGuidanceConfig(
        enabled=True, advisor_path=str(path), retrain_steps=2, iterations=5))
    world = build_guided_spatial_world(config)
    initial_trees = world.advisor.model.tree_count_

    world.run(4)

    assert world.retraining_steps == [4]
    assert world.advisor.model.tree_count_ == initial_trees + config.spatial_guidance.iterations


def test_guided_spatial_retrain_interval_must_be_positive():
    config = SimConfig(spatial_guidance=SpatialGuidanceConfig(enabled=True, retrain_steps=0))
    with pytest.raises(ValueError, match="retrain_steps"):
        build_guided_spatial_world(config)


def test_guided_spatial_training_requires_evaluated_offspring():
    config = SimConfig(spatial=SpatialConfig(initial_population=0, cluster_count=0),
                       spatial_guidance=SpatialGuidanceConfig(enabled=True,
                                                              training_steps=2, iterations=5))
    with pytest.raises(ValueError, match="no evaluated offspring"):
        build_guided_spatial_world(config)


def test_guided_spatial_loading_reports_missing_file(tmp_path):
    config = SimConfig(spatial_guidance=SpatialGuidanceConfig(
        enabled=True, advisor_path=str(tmp_path / "missing.cbm")))
    with pytest.raises(FileNotFoundError, match="Model file does not exist"):
        build_guided_spatial_world(config)
