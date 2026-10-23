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


@pytest.fixture(scope="module")
def pretrained_spatial(tmp_path_factory):
    from config.sim_config import AdvisorTrainingConfig
    from neural.advisor_training import train_advisor_model

    path = tmp_path_factory.mktemp("spatial_advisor") / "advisor.cbm"
    training = SimConfig(seed=42, n_steps=15)
    advisor, _ = train_advisor_model(AdvisorTrainingConfig(
        simulation=training, iterations=5, save_path=str(path)))
    config = replace(training, spatial_guidance=SpatialGuidanceConfig(enabled=True, advisor_path=str(path)))
    return config, path, advisor


def test_guided_spatial_world_only_loads_and_never_fits(pretrained_spatial, monkeypatch):
    from neural.advisor import CatBoostAdvisor

    config, path, _ = pretrained_spatial

    def no_fit(*args, **kwargs):
        raise AssertionError("Simulation must never fit the advisor")

    monkeypatch.setattr(CatBoostAdvisor, "fit", no_fit)
    before_file = path.read_bytes()
    world = build_guided_spatial_world(config)
    advisor = world.mutation_strategy._advisor
    before_leaves = advisor.model.get_leaf_values().copy()
    before_trees = advisor.model.tree_count_
    assert world.step_count == 0 and world.recorder is None
    world.run(60)
    assert advisor.model.tree_count_ == before_trees
    np.testing.assert_array_equal(advisor.model.get_leaf_values(), before_leaves)
    assert path.read_bytes() == before_file


def test_guided_spatial_loading_is_reproducible(pretrained_spatial):
    config, _, _ = pretrained_spatial
    first, second = build_guided_spatial_world(config), build_guided_spatial_world(config)
    first.run(5)
    second.run(5)
    assert first.history == second.history


def test_guided_spatial_loading_reports_missing_file(tmp_path):
    config = SimConfig(spatial_guidance=SpatialGuidanceConfig(
        enabled=True, advisor_path=str(tmp_path / "missing.cbm")))
    with pytest.raises(FileNotFoundError, match="Model file does not exist"):
        build_guided_spatial_world(config)


def test_guided_spatial_requires_an_advisor_path():
    config = SimConfig(spatial_guidance=SpatialGuidanceConfig(enabled=True, advisor_path=""))
    with pytest.raises(ValueError, match="advisor_path"):
        build_guided_spatial_world(config)
