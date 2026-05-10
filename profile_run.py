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

import hydra
from omegaconf import OmegaConf

from config.registry import register_configs
from config.sim_config import ProfilerConfig
from profiling.profiler import run_profile

register_configs()


@hydra.main(version_base=None, config_path="config", config_name="profile_config")
def main(cfg: ProfilerConfig):
    cfg = OmegaConf.to_object(cfg)
    run_profile(cfg, PROJECT_ROOT)


if __name__ == "__main__":
    main()

