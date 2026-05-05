import cProfile
import io
import pstats
from pathlib import Path

from config.sim_config import ProfilerConfig


def is_project_file(filename: str, project_root: Path) -> bool:
    """
    Determine whether a stats entry belongs to EvoSim source code.

    Excludes stdlib, third-party packages (site-packages / venv),
    profiling tooling and the entry-point script itself.

    Args:
        filename: Source file reported by cProfile.
        project_root: Absolute path to the EvoSim project root.

    Returns:
        True if the entry should be reported as project code.
    """

    if not filename or filename in ("~", "<string>"):
        return False
    try:
        p = Path(filename).resolve()
        rel = p.relative_to(project_root)
    except (OSError, ValueError):
        return False
    if {"site-packages", ".venv", "venv"}.intersection(p.parts):
        return False
    if "profiling" in rel.parts:
        return False
    return p.name != "profile_run.py"


def collect_rows(pr: cProfile.Profile, total_tt: float, project_root: Path) -> list[dict]:
    """
    Extract one row per project-code function from the cProfile stats.

    Args:
        pr: A finalized cProfile.Profile instance.
        total_tt: Total runtime, used to compute percentages.
        project_root: Absolute path to the EvoSim project root.

    Returns:
        List of dicts sorted by self time descending.
    """

    rows = []
    for (filename, lineno, funcname), (_cc, nc, tt, ct, _) in pstats.Stats(pr).stats.items():
        if not is_project_file(filename, project_root):
            continue
        rel = Path(filename).resolve().relative_to(project_root).as_posix()
        rows.append({
            "func": funcname,
            "file": rel,
            "line": lineno,
            "ncalls": nc,
            "tottime": tt,
            "cumtime": ct,
            "pct_tot": tt / total_tt * 100 if total_tt > 0 else 0.0,
            "pct_cum": ct / total_tt * 100 if total_tt > 0 else 0.0,
        })
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows


def save_text_summary(pr: cProfile.Profile, config: ProfilerConfig, output_dir: Path) -> Path:
    """
    Write a sorted text summary (cumulative + self time + callers) to disk.

    Args:
        pr: A finalized cProfile.Profile instance.
        config: Profiler configuration controlling top-N sizes and filename.
        output_dir: Directory where the summary file will be created.

    Returns:
        Path to the written summary file.
    """

    n_steps = config.simulation_config.n_steps
    out = output_dir / config.text_summary_filename
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"EvoSim cProfile report — {n_steps} steps\n")
        f.write("=" * 70 + "\n\n")
        for sort_key, label in [
            ("cumulative", f"CUMULATIVE TIME — top {config.top_n_text}"),
            ("tottime", f"SELF TIME — top {config.top_n_text}"),
        ]:
            f.write(f"--- {label} ---\n")
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(config.top_n_text)
            f.write(buf.getvalue() + "\n")
        f.write(f"--- CALLERS of top {config.top_n_callers} hot functions ---\n")
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_callers(config.top_n_callers)
        f.write(buf.getvalue())
    return out


def print_console_summary(pr: cProfile.Profile, n: int) -> None:
    """
    Print the top-N self-time entries to stdout.

    Args:
        pr: A finalized cProfile.Profile instance.
        n: Number of entries to print.
    """

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(n)
    print(f"\nTop {n} by self time:")
    for line in buf.getvalue().splitlines():
        print(line)
