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


def collect_rows(stats: pstats.Stats, project_root: Path) -> list[dict]:
    """
    Extract one row per project-code function from the cProfile stats.

    Args:
        stats: A loaded pstats.Stats instance (from a Profile or a .prof file).
        project_root: Absolute path to the EvoSim project root.

    Returns:
        List of dicts sorted by self time descending.
    """

    total_tt = stats.total_tt or 1e-9

    rows: list[dict] = []
    for (filename, lineno, funcname), (_cc, nc, tt, ct, _) in stats.stats.items():
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


def _capture(stats: pstats.Stats, sort_key: str, n: int, callers: bool = False) -> str:
    """
    Capture the output of ``print_stats`` / ``print_callers`` into a string.

    Args:
        stats: A loaded pstats.Stats instance.
        sort_key: pstats sort key (e.g. ``"tottime"``, ``"cumulative"``).
        n: Top-N rows to include.
        callers: If True, emit ``print_callers`` instead of ``print_stats``.

    Returns:
        Captured text output as a string.
    """

    buf = io.StringIO()
    old_stream = stats.stream
    stats.stream = buf
    try:
        sorted_stats = stats.sort_stats(sort_key)
        if callers:
            sorted_stats.print_callers(n)
        else:
            sorted_stats.print_stats(n)
    finally:
        stats.stream = old_stream
    return buf.getvalue()


def save_text_summary(stats: pstats.Stats, config: ProfilerConfig, output_dir: Path) -> Path:
    """
    Write a sorted text summary (cumulative + self time + callers) to disk.

    Args:
        stats: A loaded pstats.Stats instance.
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
            f.write(_capture(stats, sort_key, config.top_n_text) + "\n")

        f.write(f"--- CALLERS of top {config.top_n_callers} hot functions ---\n")
        f.write(_capture(stats, "tottime", config.top_n_callers, callers=True))
    return out


def print_console_summary(stats: pstats.Stats, n: int) -> None:
    """
    Print the top-N self-time entries to stdout.

    Args:
        stats: A loaded pstats.Stats instance.
        n: Number of entries to print.
    """

    print(f"\nTop {n} by self time:")
    for line in _capture(stats, "tottime", n).splitlines():
        print(line)
