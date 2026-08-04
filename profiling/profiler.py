import cProfile
import pstats
import webbrowser
from pathlib import Path

from config.sim_config import ProfilerConfig
from core.model import Model

from .report import render_html
from .stats import print_console_summary, save_text_summary


def run_profile(config: ProfilerConfig, project_root: Path) -> None:
    """
    Run the core simulation under cProfile.
    """

    sim_config = config.simulation_config
    n_steps = sim_config.n_steps
    seed = sim_config.seed

    output_dir = (project_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prof_path = output_dir / config.prof_filename.format(n_steps=n_steps)

    print(f"\n[EvoSim profiler]  n_steps={n_steps}  seed={seed}")
    print("  profiler will capture core simulation only")
    print("-" * 60)

    model = Model(sim_config)
    pr = cProfile.Profile()
    pr.enable()
    model.run(n_steps)
    pr.disable()

    pr.dump_stats(str(prof_path))
    print(f"  .prof file    -> {prof_path}")
    stats = pstats.Stats(pr)

    print(f"  text summary  -> {save_text_summary(stats, config, output_dir)}")
    print(f"  HTML report   -> {render_html(stats, config, output_dir, project_root)}")

    print_console_summary(stats, config.top_n_console)

    html_path = output_dir / config.html_report_filename
    if config.view:
        webbrowser.open(html_path.as_uri())

    print("\nDone.")
