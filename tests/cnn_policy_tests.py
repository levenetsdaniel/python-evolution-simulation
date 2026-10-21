"""Movement imitation, checkpoint compatibility and spatial physics boundaries."""

from copy import deepcopy
from dataclasses import replace
from math import hypot

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config.sim_config import IndividualConfig, SimConfig, SpatialCNNConfig, SpatialConfig, validate_sim_config
from core.spatial import SpatialWorld
from core.spatial_policy import FoodObservation, MoveAction
from neural.cnn_policy import CNNPolicy, MovementCNN, encode_observation, normalize_action
from neural.cnn_training import build_cnn_policy, collect_movement_samples


@pytest.fixture(scope="module")
def trained_policy():
    config = SimConfig(seed=42,
                       spatial=SpatialConfig(initial_population=12, max_population=30,
                                             local_map_size=16, cluster_count=2, sources_per_cluster=3),
                       spatial_cnn=SpatialCNNConfig(enabled=True, training_steps=6, training_runs=1,
                                                    validation_steps=4, max_samples=64, epochs=4,
                                                    batch_size=16, save_path=""))
    return config, build_cnn_policy(config)


def test_teacher_actions_normalize_to_the_movement_budget():
    assert normalize_action(MoveAction(10, 0), np.float64(2)).dtype == np.float32
    np.testing.assert_allclose(normalize_action(MoveAction(10, 0), 2), [1, 0])
    np.testing.assert_allclose(normalize_action(MoveAction(1, 0), 2), [0.5, 0])
    np.testing.assert_allclose(normalize_action(MoveAction(0, 0), 2), [0, 0])
    np.testing.assert_allclose(normalize_action(MoveAction(1, 0), 0), [0, 0])


def test_training_uses_a_separate_validation_world_and_bounded_reservoir(trained_policy):
    config, policy = trained_policy
    report = policy.training_report
    assert report["training_seeds"] == [config.seed]
    assert report["validation_seed"] == config.seed + 1
    assert report["training_samples"] <= config.spatial_cnn.max_samples
    assert report["validation_mse"] <= report["initial_validation_mse"]
    assert report["epochs"][-1]["training_mse"] < report["epochs"][0]["training_mse"]


def test_cnn_training_is_reproducible_with_the_same_seed(trained_policy):
    config, first = trained_policy
    second = build_cnn_policy(config)
    assert first.training_report == second.training_report
    for name, weights in first.network.state_dict().items():
        assert torch.equal(weights, second.network.state_dict()[name])


def test_recording_samples_does_not_change_the_heuristic_trajectory(trained_policy):
    from neural.cnn_training import MovementBuffer, RecordingHeuristicPolicy

    config, _ = trained_policy
    reference, recorded = SpatialWorld.from_sim_config(config), SpatialWorld.from_sim_config(config)
    recorded.policy = RecordingHeuristicPolicy(MovementBuffer(3, seed=100))
    reference.run(6)
    recorded.run(6)
    assert reference.history == recorded.history
    assert len(recorded.policy.buffer.samples) == 3


def test_cnn_inputs_exclude_absolute_position_and_teacher_target(trained_policy):
    config, policy = trained_policy
    world = SpatialWorld.from_sim_config(config)
    agent = next(iter(world.organisms.values()))
    observation = world.observation_provider.observe(world, agent)
    changed = replace(observation, x=-1e9, y=1e9,
                      nearest_food=FoodObservation(-1e9, 1e9, 1e20))
    before = deepcopy(world.rng.bit_generator.state)
    first = policy.choose(observation, world.rng, 0.5)
    second = policy.choose(changed, world.rng, 0.5)
    assert first == second
    assert hypot(first.dx, first.dy) <= observation.speed + 1e-6
    assert world.rng.bit_generator.state == before
    pixels, state = encode_observation(observation)
    assert pixels.dtype == np.uint8 and state.shape == (10,)


def test_cnn_can_learn_to_distinguish_food_on_opposite_sides():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        network = MovementCNN()
        pixels = torch.zeros(2, 4, 16, 16)
        pixels[:, 3] = 1
        pixels[0, 0, 8, 2] = 1
        pixels[1, 0, 8, 13] = 1
        state = torch.zeros(2, 10)
        targets = torch.tensor([[-0.8, 0], [0.8, 0]])
        optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(network(pixels, state), targets)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            predicted = network(pixels, state)
        assert float(torch.nn.functional.mse_loss(predicted, targets)) < 0.03
        assert predicted[0, 0] < -0.5 and predicted[1, 0] > 0.5


def test_checkpoint_roundtrip_preserves_actions_and_world_history(trained_policy, tmp_path):
    config, policy = trained_policy
    path = tmp_path / "nested" / "policy.pt"
    policy.save(path)
    loaded = CNNPolicy.load(path, config)
    worlds = [SpatialWorld.from_sim_config(config), SpatialWorld.from_sim_config(config)]
    worlds[0].policy, worlds[1].policy = policy, loaded
    for world in worlds:
        world.run(4)
    assert worlds[0].history == worlds[1].history
    assert loaded.training_report == policy.training_report
    with pytest.raises(ValueError, match="incompatible"):
        CNNPolicy.load(path, replace(config, spatial=replace(config.spatial, local_map_span=100)))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CNNPolicy.load(tmp_path / "missing.pt", config)


def test_cnn_batch_respects_explicit_actions_walls_and_energy(trained_policy):
    config, policy = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=0,
                                             cluster_count=0, reproduction_rate=0),
                     individual=IndividualConfig(fitness_death_threshold=0, fitness_death_prob_coef=0,
                                                 basal_energy_cost=0, speed_energy_cost=0,
                                                 aggressiveness_energy_cost=0, age_scale=1e9))
    network = MovementCNN()
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        network.movement[-2].bias.fill_(3)
    world = SpatialWorld.from_sim_config(config)
    world.policy = CNNPolicy(network, config)
    first = world.add_organism(np.full(7, 0.5), 99.9, 99.9, energy=100)
    second = world.add_organism(np.full(7, 0.5), 10, 10, energy=100)
    world.step({second.unique_id: MoveAction(0, 0)})
    assert first.pos == pytest.approx((100, 100))
    assert first.energy == pytest.approx(100 - hypot(0.1, 0.1) * config.spatial.movement_energy_cost)
    assert second.pos == (10, 10)


def test_empty_training_population_reports_missing_samples(trained_policy):
    config, _ = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=0))
    buffer = collect_movement_samples(config, [config.seed], 2, 3)
    with pytest.raises(ValueError, match="no observations"):
        buffer.tensors()


def test_cnn_main_loads_checkpoint_and_exports_visualization(trained_policy, tmp_path, capsys):
    from omegaconf import OmegaConf
    from main import main

    config, policy = trained_policy
    path = tmp_path / "policy.pt"
    policy.save(path)
    config = replace(config, n_steps=2, view=True, spatial_html=str(tmp_path / "world.html"),
                     spatial_csv=str(tmp_path / "history.csv"),
                     spatial_cnn=replace(config.spatial_cnn, model_path=str(path)))
    main.__wrapped__(OmegaConf.structured(config))
    assert (tmp_path / "world.html").exists()
    assert len((tmp_path / "history.csv").read_text().splitlines()) == 4
    assert "CNN validation MSE" in capsys.readouterr().out


def test_cnn_and_guided_mutation_can_run_together(trained_policy, tmp_path):
    from config.sim_config import AdvisorTrainingConfig, SpatialGuidanceConfig
    from neural.advisor_training import train_advisor_model
    from neural.spatial_guidance import build_guided_spatial_world

    config, policy = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=100,
                                             max_population=500, cluster_count=5,
                                             sources_per_cluster=12),
                     n_steps=15)
    path = tmp_path / "advisor.cbm"
    train_advisor_model(AdvisorTrainingConfig(simulation=config, iterations=5, save_path=str(path)))
    config = replace(config, spatial_guidance=SpatialGuidanceConfig(enabled=True, advisor_path=str(path)))
    world = build_guided_spatial_world(config)
    world.policy = policy
    world.run(6)
    assert world.step_count == 6
    assert world.recorder is None
    assert all(np.isfinite(row["AvgEnergy"]) for row in world.history)


@pytest.mark.parametrize("changes", [{"epochs": 0}, {"batch_size": 0}, {"threads": 0},
                                     {"learning_rate": float("nan")}, {"max_samples": 1.5}])
def test_invalid_cnn_config_is_rejected(changes):
    with pytest.raises(ValueError, match="spatial_cnn"):
        validate_sim_config(SimConfig(spatial_cnn=SpatialCNNConfig(**changes)))
