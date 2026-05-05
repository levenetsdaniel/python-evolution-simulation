import cProfile
from pathlib import Path

from config.sim_config import ProfilerConfig
from core.model import Model

from .stats import print_console_summary, save_text_summary


def run_profile(config: ProfilerConfig, project_root: Path) -> None:
    """
    Run the simulation under cProfile.

    Args:
        config: Profiler configuration.
        project_root: Absolute path to the EvoSim project root.
    """

    sim_config = config.simulation_config
    n_steps = sim_config.n_steps
    seed = sim_config.seed

    output_dir = (project_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prof_path = output_dir / config.prof_filename.format(n_steps=n_steps)

    print(f"\n[EvoSim profiler]  n_steps={n_steps}  seed={seed}")
    print("  imports already warmed up — profiler will capture simulation only")
    print("-" * 60)

    pr = cProfile.Profile()
    pr.enable()
    Model(sim_config).run(n_steps)
    pr.disable()

    pr.dump_stats(str(prof_path))
    print(f"  .prof file    -> {prof_path}")
    print(f"  text summary  -> {save_text_summary(pr, config, output_dir)}")

    print_console_summary(pr, config.top_n_console)

    print("\nDone.")