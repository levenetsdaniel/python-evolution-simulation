import json
import platform
import pstats
import sys
from datetime import datetime
from pathlib import Path

from config.sim_config import ProfilerConfig

from .stats import collect_rows

_TEMPLATE_PATH = Path(__file__).parent / "report_template.html"


def _format_calls(n: int) -> str:
    """Format a call count for compact display."""

    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1000:.2f}K"
    return str(n)


def render_html(stats: pstats.Stats, profiler_config: ProfilerConfig,
                output_dir: Path, project_root: Path) -> Path:
    """
    Render the HTML profiling report and write it to disk.

    Args:
        stats: A loaded pstats.Stats instance.
        profiler_config: Profiler configuration.
        output_dir: Directory where the HTML file will be created.
        project_root: Absolute path to the EvoSim project root.

    Returns:
        Path to the written HTML file.
    """

    sim_config = profiler_config.simulation_config
    n_steps = sim_config.n_steps
    seed = sim_config.seed

    total_tt = stats.total_tt or 1e-9

    rows = collect_rows(stats, project_root)
    flame_rows = rows[:profiler_config.top_n_flame]

    hot_name = f'{rows[0]["func"]}()' if rows else "—"
    hot_pct = rows[0]["pct_tot"] if rows else 0.0

    py_ver = (f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
              f" · {platform.python_implementation()}")

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    html = (template
            .replace("__RUN__", f"{n_steps} steps, seed {seed}")
            .replace("__POP__", f"{sim_config.population.initial_size} agents (initial)")
            .replace("__PYVER__", py_ver)
            .replace("__RUNTIME__", f"{total_tt:.2f} s")
            .replace("__STEPS_SUB__", f"{n_steps} steps · {datetime.now():%Y-%m-%d %H:%M}")
            .replace("__HOT_FN__", hot_name)
            .replace("__HOT_PCT__", f"{hot_pct:.1f}% self-time")
            .replace("__CALLS__", _format_calls(stats.total_calls))
            .replace("__PRIM__", f"{_format_calls(stats.prim_calls)} primitive")
            .replace("__N_FN__", str(len(rows)))
            .replace("__EVOSIM_JSON__", json.dumps(flame_rows))
            )

    out = output_dir / profiler_config.html_report_filename
    out.write_text(html, encoding="utf-8")
    return out
