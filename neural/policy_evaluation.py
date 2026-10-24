"""Reproducible paired evaluation of saved CNN/PPO policies, without learning."""

from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from config.scenarios import SCENARIOS
from config.sim_config import PolicyEvaluationConfig, validate_sim_config
from .cnn_policy import CNNPolicy
from .training_scenarios import TRAINING_SCENARIOS, build_training_world


def summarize_world(world, steps):
    """Extinction contributes zeros for the unobserved remainder of the horizon."""
    history = world.history
    return {
        "initial_population": history[0]["PopulationSize"],
        "final_population": history[-1]["PopulationSize"],
        "survived": bool(history[-1]["PopulationSize"]),
        "observed_steps": world.step_count,
        "extinction_step": world.step_count if not world.agents else None,
        "mean_population": sum(row["PopulationSize"] for row in history) / (steps + 1),
        "births": sum(row["BirthCount"] for row in history),
        "deaths": sum(row["Deaths_total"] for row in history),
        "energy_deaths": sum(row["Deaths_energy"] for row in history),
        "final_energy": history[-1]["AvgEnergy"],
        "mean_lifespan": history[-1]["AvgLifespan"],
    }


def paired_difference(rows, baseline, candidate, scenario):
    reference = {r["seed"]: r for r in rows if r["model"] == baseline and r["scenario"] == scenario}
    selected = {r["seed"]: r for r in rows if r["model"] == candidate and r["scenario"] == scenario}
    seeds = sorted(reference.keys() & selected.keys())
    if not seeds:
        return {}
    result = {"paired_runs": len(seeds)}
    # Resample seed pairs, rather than treating organism decisions as independent.
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(seeds), (10000, len(seeds)))
    for metric in ("final_population", "mean_population", "births", "survived"):
        differences = np.asarray([float(selected[s][metric]) - float(reference[s][metric]) for s in seeds])
        low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
        result[metric] = {"mean_delta": float(differences.mean()), "ci95": [float(low), float(high)]}
    return result


def validate_evaluation_config(config):
    validate_sim_config(config.simulation)
    if config.simulation.mode != "spatial" or config.simulation.spatial_guidance.enabled:
        raise ValueError("Policy comparison requires spatial mode with CatBoost disabled")
    if not config.models or config.baseline not in config.models:
        raise ValueError("models must include the baseline")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in config.seeds):
        raise ValueError("Evaluation seeds must be non-negative integers")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (config.steps, config.threads, config.workers)
    ):
        raise ValueError("steps, threads and workers must be positive integers")
    if len(set(config.environments)) != len(config.environments) or any(
        name not in SCENARIOS for name in config.environments
    ):
        raise ValueError("Unknown or duplicate environment")
    if len(set(config.behavior_scenarios)) != len(config.behavior_scenarios) or any(
        name not in TRAINING_SCENARIOS or name == "baseline" for name in config.behavior_scenarios
    ):
        raise ValueError("Unknown or duplicate behavioral scenario")
    if not config.environments and not config.behavior_scenarios:
        raise ValueError("At least one evaluation scenario is required")
    if not config.output_prefix.strip():
        raise ValueError("output_prefix is required")


def check_held_out_seeds(report, seeds):
    used = set(report.get("training_seeds", [])) | set(report.get("validation_seeds", []))
    if report.get("validation_seed") is not None:
        used.add(report["validation_seed"])
    if used.intersection(seeds):
        raise ValueError("Evaluation seeds overlap with model training or validation seeds")


def write_comparison_chart(summary, path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Доля прогонов с живой популяцией к концу",
            "Конечная численность",
            "Средняя численность за весь горизонт",
        ),
    )
    colors = ["#4477aa", "#ee7733", "#228833", "#cc6677"]
    for index, name in enumerate(dict.fromkeys(row["model"] for row in summary)):
        rows = [row for row in summary if row["model"] == name]
        for position, metric in enumerate(("survived_runs", "final_population", "mean_population"), 1):
            values = [row[metric] / row["runs"] if metric == "survived_runs" else row[metric] for row in rows]
            figure.add_trace(
                go.Bar(
                    name=name,
                    legendgroup=name,
                    showlegend=position == 1,
                    x=[row["scenario"] for row in rows],
                    y=values,
                    marker_color=colors[index % len(colors)],
                ),
                row=position,
                col=1,
            )
    figure.update_layout(barmode="group", height=1050, title="CNN и PPO: отдельные проверочные seed")
    figure.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=1)
    figure.write_html(path, include_plotlyjs=True, auto_open=False)


_POLICIES = {}


def evaluate_case(job):
    path, simulation, curriculum, task, steps, threads, details = job
    torch.set_num_threads(threads)
    if path not in _POLICIES:
        _POLICIES[path] = CNNPolicy.load(path, simulation)
    started = time.monotonic()
    world = build_training_world(simulation, task, simulation.seed, curriculum)
    world.policy = _POLICIES[path]
    for _ in range(steps):
        if not world.agents:
            break
        world.step()
    return {**details, **summarize_world(world, steps), "seconds": time.monotonic() - started}


def evaluate_policies(config: PolicyEvaluationConfig):
    validate_evaluation_config(config)
    torch.set_num_threads(config.threads)
    prefix = Path(config.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(str(prefix) + ".jsonl")
    cache = {}
    if cache_path.exists():
        # Ignore an interrupted trailing write; all complete records are reusable.
        for line in cache_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[row["fingerprint"]] = row
    cases = [(name, "baseline", name) for name in config.environments]
    cases += [(name, name, "stable") for name in config.behavior_scenarios]
    rows, models, jobs, fingerprints = [], {}, [], []
    for name, path in config.models.items():
        checkpoint = Path(path)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        policy = CNNPolicy.load(path, config.simulation)
        check_held_out_seeds(policy.training_report, config.seeds)
        models[name] = {"path": str(checkpoint), "sha256": digest}
        for scenario, task, climate in cases:
            for seed in config.seeds:
                simulation = replace(config.simulation, seed=seed, environment=SCENARIOS[climate])
                specification = {
                    "version": 1,
                    "model": name,
                    "sha256": digest,
                    "simulation": asdict(simulation),
                    "curriculum": asdict(config.curriculum),
                    "task": task,
                    "climate": climate,
                    "steps": config.steps,
                }
                fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
                fingerprints.append(fingerprint)
                if fingerprint in cache:
                    rows.append(cache[fingerprint])
                else:
                    details = {"fingerprint": fingerprint, "model": name, "scenario": scenario, "seed": seed}
                    jobs.append(
                        (
                            str(checkpoint),
                            simulation,
                            config.curriculum,
                            task,
                            config.steps,
                            config.threads,
                            details,
                        )
                    )
    print(f"Evaluation: {len(rows)} cached runs, {len(jobs)} pending, workers={config.workers}", flush=True)

    def record(row):
        rows.append(row)
        with cache_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
        print(
            f"{row['model']}/{row['scenario']}/seed={row['seed']}: population={row['final_population']}, "
            f"births={row['births']}, steps={row['observed_steps']}",
            flush=True,
        )

    if config.workers == 1:
        _POLICIES.clear()
        for job in jobs:
            record(evaluate_case(job))
    elif jobs:
        with ProcessPoolExecutor(
            max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [executor.submit(evaluate_case, job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    order = {fingerprint: index for index, fingerprint in enumerate(fingerprints)}
    rows.sort(key=lambda row: order[row["fingerprint"]])
    for model in models.values():
        if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"]:
            path = model["path"]
            raise RuntimeError(f"Model changed during evaluation: {path}")
    summary = []
    for name in config.models:
        for scenario, _, _ in cases:
            group = [r for r in rows if r["model"] == name and r["scenario"] == scenario]
            summary.append(
                {
                    "model": name,
                    "scenario": scenario,
                    "runs": len(group),
                    "survived_runs": sum(r["survived"] for r in group),
                    **{
                        metric: float(np.mean([r[metric] for r in group]))
                        for metric in (
                            "final_population",
                            "mean_population",
                            "births",
                            "energy_deaths",
                            "mean_lifespan",
                        )
                    },
                }
            )
    paired = {
        name: {scenario: paired_difference(rows, config.baseline, name, scenario) for scenario, _, _ in cases}
        for name in config.models
        if name != config.baseline
    }
    pairwise = {
        reference: {
            candidate: {
                scenario: paired_difference(rows, reference, candidate, scenario) for scenario, _, _ in cases
            }
            for candidate in config.models
            if candidate != reference
        }
        for reference in config.models
    }
    report = {
        "config": asdict(config),
        "models": models,
        "summary": summary,
        "paired": paired,
        "pairwise": pairwise,
        "runs": rows,
    }
    Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with Path(str(prefix) + ".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Сравнение пространственных политик",
        "",
        f"Горизонт: {config.steps} шагов; seed: {config.seeds}. "
        "Модели заморожены, CatBoost отключён. Вымирание обнуляет оставшуюся часть горизонта.",
        "",
        "| Модель | Сценарий | Выжившие прогоны | Конечная популяция | Средняя популяция | Рождения |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['scenario']} | {row['survived_runs']}/{row['runs']} | "
            f"{row['final_population']:.1f} | {row['mean_population']:.1f} | {row['births']:.1f} |"
        )
    lines += [
        "",
        "Парные разности и 95% bootstrap-интервалы по seed сохранены в JSON. "
        "Небольшое число seed ограничивает точность оценки; эти результаты не используются для выбора весов.",
    ]
    Path(str(prefix) + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_comparison_chart(summary, str(prefix) + ".html")
    return report


@hydra.main(version_base=None, config_path="../config", config_name="policy_evaluation_config")
def main(cfg: PolicyEvaluationConfig):
    evaluate_policies(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    from config.registry import register_configs

    register_configs()
    main()
