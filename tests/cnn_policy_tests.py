"""Visual target selection, label geometry, controllers and checkpoint compatibility."""

from copy import deepcopy
from dataclasses import replace
from math import hypot, pi

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config.sim_config import IndividualConfig, SimConfig, SpatialCNNConfig, SpatialConfig, validate_sim_config
from core.environment import FoodSource
from core.spatial import SpatialWorld
from core.spatial_policy import FoodObservation, MateObservation, MoveAction, SpatialTarget, TargetController
from neural.cnn_policy import CNNPolicy, GOAL_NAMES, IGNORE_TARGET, STATE_SIZE, TargetCNN, encode_observation
from neural.cnn_training import (TargetBuffer, RecordingHeuristicPolicy, build_cnn_policy,
                                 collect_target_samples, encode_target, rotate_examples, target_loss, target_scores)


@pytest.fixture(scope="module")
def trained_policy():
    config = SimConfig(seed=42,
                       spatial=SpatialConfig(initial_population=12, max_population=30,
                                             local_map_size=16, cluster_count=2, sources_per_cluster=3),
                       spatial_cnn=SpatialCNNConfig(enabled=True, training_steps=6, training_runs=1,
                                                    validation_steps=4, max_samples=64, epochs=8,
                                                    batch_size=16, save_path=""))
    return config, build_cnn_policy(config)


def food_observation(distance=2):
    world = SpatialWorld(SpatialConfig(initial_population=0, cluster_count=0))
    agent = world.add_organism(np.full(7, 0.5), 50, 50, energy=30)
    world.food_sources.append(FoodSource(50 + distance, 50, 100, 100, 0, 2))
    return world, agent, world.observation_provider.observe(world, agent)


def test_teacher_labels_select_visible_cells_and_prefer_the_detailed_map():
    for distance, scale in [(2, 1), (20, 0)]:
        _, _, observation = food_observation(distance)
        goal, cell = encode_target(observation, 96, 12)
        assert GOAL_NAMES[goal] == "forage"
        assert cell // (64 * 64) == scale
        _, cell = divmod(cell, 64 * 64)
        row, column = divmod(cell, 64)
        pixels = observation.nearby_map if scale else observation.local_map
        assert pixels[0, row, column] == 1
    _, _, observation = food_observation()
    hidden = replace(observation, local_map=np.zeros_like(observation.local_map),
                     nearby_map=np.zeros_like(observation.nearby_map))
    assert encode_target(hidden, 96, 12) == (GOAL_NAMES.index("explore"), IGNORE_TARGET)


def test_training_rotations_preserve_target_scale_heading_and_visibility():
    pixels = torch.zeros(4, 10, 8, 8, dtype=torch.uint8)
    pixels[:, 0, 4, 6] = pixels[:, 5, 4, 6] = 255
    state = torch.zeros(4, STATE_SIZE)
    state[:, 1] = 1
    targets = torch.tensor([4 * 8 + 6, 64 + 4 * 8 + 6, 4 * 8 + 6, IGNORE_TARGET])
    rotated, headings, labels = rotate_examples(pixels, state, targets, torch.arange(4))
    torch.testing.assert_close(headings[:, 1:3], torch.tensor([[1., 0], [0, 1], [-1, 0], [0, -1]]))
    assert labels.tolist() == [38, 64 + 6 * 8 + 3, 3 * 8 + 1, IGNORE_TARGET]
    for turn in range(4):
        torch.testing.assert_close(rotated[turn], torch.rot90(pixels[turn], turn, (-2, -1)))
    assert torch.equal(state[:, 1], torch.ones(4))
    assert targets[1] == 102


def test_training_uses_held_out_worlds_and_bounded_samples(trained_policy):
    config, policy = trained_policy
    report = policy.training_report
    assert report["training_seeds"] == [config.seed]
    assert report["validation_seed"] == config.seed + 1
    assert report["training_samples"] <= config.spatial_cnn.max_samples
    assert report["validation_loss"] <= report["initial_validation_loss"]
    assert report["validation_loss"] <= min(epoch["validation_loss"] for epoch in report["epochs"])
    assert report["validation_by_goal"]
    assert 0 <= report["goal_accuracy"] <= 1 and 0 <= report["target_accuracy"] <= 1


def test_training_is_reproducible(trained_policy):
    config, first = trained_policy
    second = build_cnn_policy(config)
    assert first.training_report == second.training_report
    for name, weights in first.network.state_dict().items():
        assert torch.equal(weights, second.network.state_dict()[name])


def test_recording_targets_preserves_the_heuristic_trajectory(trained_policy):
    config, _ = trained_policy
    reference, recorded = SpatialWorld.from_sim_config(config), SpatialWorld.from_sim_config(config)
    recorded.policy = RecordingHeuristicPolicy(TargetBuffer(3, seed=100))
    reference.run(6)
    recorded.run(6)
    assert reference.history == recorded.history
    assert len(recorded.policy.buffer.samples) == 3


def test_target_selection_and_execution_never_read_teacher_or_absolute_coordinates(trained_policy):
    config, policy = trained_policy
    world = SpatialWorld.from_sim_config(config)
    agent = next(iter(world.organisms.values()))
    observation = world.observation_provider.observe(world, agent)
    changed = replace(observation, x=-1e9, y=1e9,
                      nearest_food=FoodObservation(-1e9, 1e9, 1e20),
                      nearest_mate=MateObservation(-1e9, 1e9, 1e20))
    assert policy.select_targets([observation]) == policy.select_targets([changed])
    rng = deepcopy(world.rng)
    first = policy.choose(observation, world.rng, 0.5)
    second = policy.choose(changed, rng, 0.5)
    assert first == second
    assert hypot(first.dx, first.dy) <= observation.speed + 1e-6
    pixels, state = encode_observation(observation)
    assert pixels.dtype == np.uint8 and pixels.shape[0] == 10 and state.shape == (STATE_SIZE,)
    np.testing.assert_array_equal(state[-3:], [observation.is_female, observation.ready_to_reproduce,
                                              observation.foraging])


def test_cnn_learns_to_choose_a_food_cell_instead_of_averaging_directions():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        network = TargetCNN()
        pixels = torch.zeros(4, 10, 16, 16)
        pixels[:, 3] = 1
        positions = [((8, 2), (8, 6)), ((8, 13), (8, 9)),
                     ((2, 8), (6, 8)), ((13, 8), (9, 8))]
        for index, pairs in enumerate(positions):
            for row, column in pairs:
                pixels[index, 0, row, column] = 1
        state = torch.zeros(4, STATE_SIZE)
        state[:, -1] = 1
        goals = torch.full((4,), GOAL_NAMES.index("forage"))
        targets = torch.tensor([pairs[-1][0] * 16 + pairs[-1][1] for pairs in positions])
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss, _, _ = target_loss(network, (pixels * 255, state, goals, targets), torch.ones(5), 1)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            goal_logits, target_logits = network(pixels, state)
        assert torch.equal(goal_logits.argmax(1), goals)
        assert torch.equal(target_scores(target_logits, goals).argmax(1), targets)


def test_empty_and_invisible_cells_are_never_selected_as_targets():
    world, _, observation = food_observation(2)
    config = SimConfig()
    network = TargetCNN()
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        network.goal[-1].bias[GOAL_NAMES.index("forage")] = 10
    policy = CNNPolicy(network, config)
    selected = policy.select_targets([observation])[0]
    assert selected.kind == "forage"
    assert selected.dx > 0
    assert selected.cell_size in (96 / 64, 12 / 64)
    empty = replace(observation, local_map=np.zeros_like(observation.local_map),
                    nearby_map=np.zeros_like(observation.nearby_map))
    assert policy.select_targets([empty])[0].kind in ("rest", "explore")


def test_controller_approaches_a_partner_and_stops_safely_inside_radius():
    _, _, observation = food_observation()
    observation = replace(observation, energy=150, speed=1.25, mating_distance=3)
    controller = TargetController()
    rng = np.random.default_rng(42)
    target = SpatialTarget("mate", 3.375, 0, 0.1875)
    action = controller.choose(target, observation, rng, 0.5)
    assert 0 < action.dx <= observation.speed and action.dy == 0
    assert target.dx - action.dx < observation.mating_distance
    assert controller.choose(SpatialTarget("mate", 2, 0, 0.1875), observation, rng, 0.5) == MoveAction(0, 0)
    assert controller.choose(SpatialTarget("forage", 0.09375, 0.09375, 0.1875), observation, rng, 0.5) == MoveAction(0, 0)
    assert controller.choose(SpatialTarget("rest"), observation, rng, 0.5) == MoveAction(0, 0)
    assert controller.choose(target, replace(observation, energy=0), rng, 0.5) == MoveAction(0, 0)
    explored = controller.choose(SpatialTarget("explore"), observation, rng, 0.5)
    assert hypot(explored.dx, explored.dy) == pytest.approx(observation.speed)


def test_checkpoint_roundtrip_preserves_targets_and_world_history(trained_policy, tmp_path):
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
    for changes in ({"local_map_span": 100}, {"nearby_map_span": 10}, {"mating_radius": 3}):
        with pytest.raises(ValueError, match="incompatible"):
            CNNPolicy.load(path, replace(config, spatial=replace(config.spatial, **changes)))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CNNPolicy.load(tmp_path / "missing.pt", config)
    checkpoint = torch.load(path, weights_only=True)
    for version in (1, 2, 3):
        checkpoint["metadata"]["version"] = version
        torch.save(checkpoint, tmp_path / "legacy.pt")
        with pytest.raises(ValueError, match="retrain"):
            CNNPolicy.load(tmp_path / "legacy.pt", config)


def test_cnn_actions_obey_walls_energy_and_explicit_commands(trained_policy):
    config, _ = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=0, cluster_count=0,
                                             reproduction_rate=0, exploration_turn=0),
                     individual=IndividualConfig(fitness_death_threshold=0, fitness_death_prob_coef=0,
                                                 basal_energy_cost=0, speed_energy_cost=0,
                                                 aggressiveness_energy_cost=0, age_scale=1e9))
    network = TargetCNN()
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        network.goal[-1].bias[GOAL_NAMES.index("explore")] = 10
    world = SpatialWorld.from_sim_config(config)
    world.policy = CNNPolicy(network, config)
    first = world.add_organism(np.full(7, 0.5), 99.9, 99.9, energy=100, heading=pi / 4)
    second = world.add_organism(np.full(7, 0.5), 10, 10, energy=100)
    world.step({second.unique_id: MoveAction(0, 0)})
    assert first.pos == pytest.approx((100, 100))
    assert first.energy == pytest.approx(100 - hypot(0.1, 0.1) * config.spatial.movement_energy_cost)
    assert second.pos == (10, 10)


def test_empty_training_population_reports_missing_samples(trained_policy):
    config, _ = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=0))
    buffer = collect_target_samples(config, [config.seed], 2, 3)
    with pytest.raises(ValueError, match="no observations"):
        buffer.tensors()


def test_cnn_main_loads_model_and_exports_visualization(trained_policy, tmp_path, capsys):
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
    assert "CNN validation loss" in capsys.readouterr().out


def test_cnn_and_guided_mutation_can_run_together(trained_policy, tmp_path):
    from config.sim_config import AdvisorTrainingConfig, SpatialGuidanceConfig
    from neural.advisor_training import train_advisor_model
    from neural.spatial_guidance import build_guided_spatial_world
    config, policy = trained_policy
    config = replace(config, spatial=replace(config.spatial, initial_population=100, max_population=500,
                                             cluster_count=5, sources_per_cluster=12), n_steps=15)
    path = tmp_path / "advisor.cbm"
    train_advisor_model(AdvisorTrainingConfig(simulation=config, iterations=5, save_path=str(path)))
    config = replace(config, spatial_guidance=SpatialGuidanceConfig(enabled=True, advisor_path=str(path)))
    world = build_guided_spatial_world(config)
    world.policy = policy
    world.run(6)
    assert world.step_count == 6 and world.recorder is None
    assert all(np.isfinite(row["AvgEnergy"]) for row in world.history)


@pytest.mark.parametrize("changes", [{"epochs": 0}, {"batch_size": 0}, {"threads": 0},
                                     {"learning_rate": float("nan")}, {"max_samples": 1.5},
                                     {"target_loss_weight": 0}])
def test_invalid_cnn_config_is_rejected(changes):
    with pytest.raises(ValueError, match="spatial_cnn"):
        validate_sim_config(SimConfig(spatial_cnn=SpatialCNNConfig(**changes)))
