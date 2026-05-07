"""
profile_run.py — entry point for profiling EvoSim with cProfile.

Usage:
    python profile_run.py
    python profile_run.py --n-steps 500
    python profile_run.py --view
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.cli import parse_args
from profiling.profiler import run_profile


def main():
    _, profiler_config = parse_args()
    run_profile(profiler_config, PROJECT_ROOT)


if __name__ == "__main__":
    main()
