"""Behavioral tasks, local observations, PPO mathematics and portable inference."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from config.sim_config import (
    IndividualConfig,
    PPOTrainingConfig,
    PolicyEvaluationConfig,
    SimConfig,
    SpatialConfig,
    SpatialCurriculumConfig,
)
from core.enums import Gender
from core.spatial import SpatialWorld
from core.spatial_policy import MoveAction
from neural.cnn_policy import CNNPolicy, GOAL_NAMES, TargetCNN
from neural.ppo_policy import ActorCritic, PPOPolicy, action_statistics, encode_batch
from neural.ppo_training import (
    collect_rollout,
    generalized_advantages,
    ppo_loss,
    train_ppo_policy,
    transition_rewards,
    validate_ppo_config,
)
from neural.training_scenarios import TRAINING_SCENARIOS, build_training_world
from neural.policy_evaluation import check_held_out_seeds, evaluate_policies, paired_difference


def small_config():
    return SimConfig(
        seed=7,
        spatial=replace(
            SpatialConfig(),
            initial_population=8,
            max_population=20,
            local_map_size=8,
            cluster_count=1,
            sources_per_cluster=2,
        ),
        individual=replace(
            IndividualConfig(), fitness_death_threshold=0, fitness_death_prob_coef=0, age_scale=1e9
        ),
    )


def test_local_counts_only_include_visible_agents_and_ready_opposite_sex():
    world = SpatialWorld(SpatialConfig(initial_population=0, cluster_count=0))
    agents = [world.add_organism(np.full(7, 0.5), x, 50) for x in (50, 55, 60, 99)]
    for agent in agents:
        agent.age = 2
    agents[0].gender = agents[2].gender = Gender.FEMALE
    agents[1].gender = agents[3].gender = Gender.MALE
    state = world.rng.bit_generator.state
    observation = world.observation_provider.observe(world, agents[0])
    assert observation.visible_population == 2
    assert observation.visible_mates == 1
    assert world.rng.bit_generator.state == state


@pytest.mark.parametrize("scenario", TRAINING_SCENARIOS)
def test_curriculum_worlds_are_reproducible_and_record_the_actual_initial_population(scenario):
    config = small_config()
    settings = SpatialCurriculumConfig(group_count=1, group_size=4)
    first = build_training_world(config, scenario, 42, settings)
    second = build_training_world(config, scenario, 42, settings)
    assert first.history[0]["PopulationSize"] == (8 if scenario == "baseline" else 4)
    if scenario != "baseline":
        assert sum(a.gender == Gender.FEMALE for a in first.agents) == 2
        assert all(a.age >= config.spatial.min_reproduction_age for a in first.agents)
    if scenario == "depleted_food":
        assert first.food_sources[0].stock < first.food_sources[1].stock
        assert first.config.food_disappearance_probability == 1
    first.run(3)
    second.run(3)
    assert first.history == second.history
    assert len(first.datacollector.get_model_vars_dataframe()) == 4
    assert all(0 <= a.state.x <= first.width and 0 <= a.state.y <= first.height for a in first.agents)
    assert config.spatial.initial_population == 8


def test_sparse_population_is_local_to_each_visible_group():
    config = replace(small_config(), spatial=replace(small_config().spatial, max_population=30))
    world = build_training_world(config, "sparse_population", 1, SpatialCurriculumConfig())
    assert len(world.agents) == 8
    assert all(world.observation_provider.observe(world, a).visible_population == 3 for a in world.agents)


def test_gae_masks_death_and_bootstraps_time_limits():
    advantage, returns = generalized_advantages([1, 2], [0, 0], [False, True], 999, 1, 1)
    np.testing.assert_array_equal(advantage, [3, 2])
    np.testing.assert_array_equal(returns, [3, 2])
    advantage, _ = generalized_advantages([1], [0], [False], 5, 0.9, 0.95)
    np.testing.assert_allclose(advantage, [5.5])


def test_joint_likelihood_ignores_target_cells_for_rest_and_explore():
    goals = torch.zeros(2, 5)
    targets = torch.full((2, 2, 2, 8, 8), -1e9)
    actions = torch.tensor([GOAL_NAMES.index("rest"), GOAL_NAMES.index("explore")])
    probability, entropy = action_statistics(goals, targets, actions, torch.tensor([0, 127]))
    torch.testing.assert_close(probability, torch.full((2,), -np.log(5), dtype=torch.float32))
    assert torch.isfinite(entropy).all()


def test_local_context_starts_neutral_and_cannot_enable_an_unavailable_goal():
    config = small_config()
    world = SpatialWorld(replace(config.spatial, initial_population=0, cluster_count=0))
    agent = world.add_organism(np.full(7, 0.5), 50, 50)
    observation = world.observation_provider.observe(world, agent)
    pixels, state, local = encode_batch([observation])
    network = ActorCritic(TargetCNN())
    with torch.no_grad():
        original_goals, original_targets = network.actor(pixels.float() / 255, state)
        goals, targets, _ = network(pixels.float() / 255, state, local)
        torch.testing.assert_close(goals, original_goals)
        torch.testing.assert_close(targets, original_targets)
        network.context[-1].bias[GOAL_NAMES.index("mate")] = 1000
        goals, _, _ = network(pixels.float() / 255, state, local)
    assert goals.argmax(-1).item() != GOAL_NAMES.index("mate")


def test_rollout_stops_bootstrapping_when_an_organism_dies():
    simulation = small_config()
    simulation = replace(simulation, individual=replace(simulation.individual, fitness_death_threshold=1))
    config = PPOTrainingConfig(
        simulation=simulation, curriculum=SpatialCurriculumConfig(group_count=1, group_size=4)
    )
    world = build_training_world(simulation, "sparse_population", 1, config.curriculum)
    decoder = CNNPolicy(TargetCNN(), simulation)
    network = ActorCritic(decoder.network)
    samples = collect_rollout(world, network, decoder, config, torch.Generator().manual_seed(2), set())
    assert len(samples) == 4 and not world.agents
    for sample in samples:
        assert sample["done"]
        assert sample["return"] == pytest.approx(sample["reward"])


def test_ppo_clipping_applies_to_positive_and_negative_advantages():
    config = PPOTrainingConfig(value_coef=0, entropy_coef=0)
    _, loss, _ = ppo_loss(
        torch.tensor([np.log(2), np.log(0.5)]),
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        config,
    )
    assert float(loss) == pytest.approx(-0.2)


def test_birth_bonus_uses_local_density_and_does_not_penalize_energy_transferred_to_offspring():
    spatial = replace(
        SpatialConfig(), initial_population=0, cluster_count=0, reproduction_rate=1, min_reproduction_age=0
    )
    individual = IndividualConfig(
        fitness_death_threshold=0,
        fitness_death_prob_coef=0,
        age_scale=1e9,
        basal_energy_cost=0,
        speed_energy_cost=0,
        aggressiveness_energy_cost=0,
    )
    world = SpatialWorld(spatial, individual_config=individual)
    mother = world.add_organism(np.full(7, 0.5), 50, 50, energy=150)
    father = world.add_organism(np.full(7, 0.5), 51, 50, energy=150)
    mother.gender, father.gender = Gender.FEMALE, Gender.MALE
    before = world.organisms
    observations = [world.observation_provider.observe(world, a) for a in before.values()]
    world.step({key: MoveAction(0, 0) for key in before})
    assert world.history[-1]["BirthCount"] == 1
    config = PPOTrainingConfig()
    sparse = transition_rewards(before, observations, world, config, set())
    crowded = transition_rewards(
        before, [replace(o, visible_population=20) for o in observations], world, config, set()
    )
    assert sparse[mother.unique_id] > crowded[mother.unique_id]
    assert sparse[mother.unique_id] == pytest.approx(sparse[father.unique_id])
    child = next(a for a in world.agents if a.parent_ids)
    child.age = config.offspring_survival_steps
    matured = set()
    obs = [world.observation_provider.observe(world, child)]
    first = transition_rewards({child.unique_id: child}, obs, world, config, matured)
    second = transition_rewards({child.unique_id: child}, obs, world, config, matured)
    assert first[child.unique_id] - second[child.unique_id] == pytest.approx(config.offspring_survival_reward)


@pytest.fixture(scope="module")
def trained_ppo(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("ppo")
    simulation = small_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        initial = CNNPolicy(TargetCNN(), simulation)
    path = root / "initial.pt"
    initial.save(path)
    config = PPOTrainingConfig(
        simulation=simulation,
        curriculum=SpatialCurriculumConfig(group_count=1, group_size=4),
        device="cpu",
        iterations=2,
        rollout_steps=3,
        episode_steps=6,
        validation_steps=4,
        validation_runs=1,
        validation_interval=1,
        epochs=1,
        batch_size=8,
        validation_scenarios=["sparse_population", "separated_mates", "depleted_food"],
        initial_model_path=str(path),
        save_path=str(root / "ppo.pt"),
    )
    before = path.read_bytes()
    policy = train_ppo_policy(config)
    assert path.read_bytes() == before
    return config, policy


def test_training_updates_the_actor_and_keeps_validation_seeds_separate(trained_ppo):
    config, policy = trained_ppo
    report = policy.training_report
    assert report["actor_updated_during_training"]
    assert report["decisions"] > 0 and report["updates"] == 2
    assert set(report["training_seeds"]).isdisjoint(report["validation_seeds"])
    assert report["best_validation"]["score"] >= report["initial_validation"]["score"]
    assert all(np.isfinite(row["loss"]) for row in report["history"])
    assert next(policy.network.parameters()).device.type == "cpu"
    assert Path(config.save_path + ".json").exists()


def test_ppo_training_is_reproducible(trained_ppo):
    config, first = trained_ppo
    second = train_ppo_policy(config)
    assert first.training_report == second.training_report
    for name, value in first.network.state_dict().items():
        assert torch.equal(value, second.network.state_dict()[name])


def test_ppo_checkpoint_loads_through_the_normal_cnn_api_and_preserves_physics(trained_ppo):
    config, policy = trained_ppo
    loaded = CNNPolicy.load(config.save_path, config.simulation)
    assert isinstance(loaded, PPOPolicy)
    worlds = [
        build_training_world(config.simulation, "sparse_population", 8, config.curriculum) for _ in range(2)
    ]
    worlds[0].policy, worlds[1].policy = policy, loaded
    for world in worlds:
        world.run(4)
    assert worlds[0].history == worlds[1].history
    with pytest.raises(ValueError, match="incompatible"):
        CNNPolicy.load(
            config.save_path,
            replace(config.simulation, spatial=replace(config.simulation.spatial, mating_radius=4)),
        )


def test_ppo_main_can_export_the_usual_visualization(trained_ppo, tmp_path, capsys):
    from omegaconf import OmegaConf
    from main import main

    config, _ = trained_ppo
    simulation = replace(
        config.simulation,
        n_steps=2,
        view=True,
        spatial_html=str(tmp_path / "ppo.html"),
        spatial_cnn=replace(config.simulation.spatial_cnn, enabled=True, model_path=config.save_path),
    )
    main.__wrapped__(OmegaConf.structured(simulation))
    assert (tmp_path / "ppo.html").exists()
    assert "PPO policy:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "changes",
    [
        {"iterations": 0},
        {"clip_ratio": 0},
        {"gamma": float("nan")},
        {"training_scenarios": ["unknown"]},
        {"validation_scenarios": []},
        {"save_path": ""},
        {"low_density_bonus": -1},
    ],
)
def test_invalid_ppo_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        validate_ppo_config(PPOTrainingConfig(**changes))


@pytest.mark.parametrize(
    "report",
    [
        {"training_seeds": [10]},
        {"validation_seeds": [10]},
        {"validation_seed": 10},
    ],
)
def test_evaluation_rejects_training_and_selection_seed_leakage(report):
    with pytest.raises(ValueError, match="overlap"):
        check_held_out_seeds(report, [10, 11])


def test_paired_comparison_joins_by_seed_instead_of_row_order():
    rows = [
        {
            "model": model,
            "scenario": "stable",
            "seed": seed,
            "final_population": value,
            "mean_population": value,
            "births": value,
            "survived": True,
        }
        for model, seed, value in [("cnn", 2, 20), ("cnn", 1, 10), ("ppo", 1, 15), ("ppo", 2, 25)]
    ]
    result = paired_difference(rows, "cnn", "ppo", "stable")
    assert result["paired_runs"] == 2
    assert result["final_population"] == {"mean_delta": 5.0, "ci95": [5.0, 5.0]}
    assert result["survived"]["mean_delta"] == 0


def test_evaluation_uses_fixed_horizon_exports_and_resumes_verified_runs(tmp_path, monkeypatch):
    simulation = small_config()
    simulation = replace(simulation, individual=replace(simulation.individual, fitness_death_threshold=1))
    path = tmp_path / "cnn.pt"
    CNNPolicy(TargetCNN(), simulation).save(path)
    config = PolicyEvaluationConfig(
        simulation=simulation,
        curriculum=SpatialCurriculumConfig(group_count=1, group_size=4),
        models={"cnn": str(path), "copy": str(path)},
        environments=["stable"],
        behavior_scenarios=["sparse_population"],
        seeds=[100, 101],
        steps=3,
        output_prefix=str(tmp_path / "comparison"),
    )
    report = evaluate_policies(config)
    assert len(report["runs"]) == 8
    assert all(not row["survived"] and row["extinction_step"] == 1 for row in report["runs"])
    assert all(row["mean_population"] == row["initial_population"] / 4 for row in report["runs"])
    assert all(
        report["paired"]["copy"][case]["final_population"]["mean_delta"] == 0
        for case in ("stable", "sparse_population")
    )
    assert all(
        Path(config.output_prefix + extension).exists()
        for extension in (".jsonl", ".json", ".csv", ".md", ".html")
    )

    def no_world(*args, **kwargs):
        raise AssertionError("Verified cached run must not be repeated")

    monkeypatch.setattr("neural.policy_evaluation.build_training_world", no_world)
    assert evaluate_policies(config) == report
    with pytest.raises(AssertionError, match="must not"):
        evaluate_policies(replace(config, steps=4))


def test_parallel_evaluation_matches_sequential_results(tmp_path):
    simulation = small_config()
    simulation = replace(
        simulation, spatial=replace(simulation.spatial, initial_population=0, cluster_count=0)
    )
    path = tmp_path / "cnn.pt"
    CNNPolicy(TargetCNN(), simulation).save(path)
    config = PolicyEvaluationConfig(
        simulation=simulation,
        models={"cnn": str(path)},
        environments=["stable"],
        behavior_scenarios=[],
        seeds=[100, 101],
        steps=3,
        output_prefix=str(tmp_path / "sequential"),
    )
    sequential = evaluate_policies(config)
    parallel = evaluate_policies(replace(config, workers=2, output_prefix=str(tmp_path / "parallel")))
    assert sequential["summary"] == parallel["summary"]
    for first, second in zip(sequential["runs"], parallel["runs"]):
        assert {key: value for key, value in first.items() if key != "seconds"} == {
            key: value for key, value in second.items() if key != "seconds"
        }
