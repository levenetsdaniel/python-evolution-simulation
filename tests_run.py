"""
Run all EvoSim test files.

Usage:
    python run_tests.py                # run all three files
    python run_tests.py -k satiation   # filter by name (forwarded to pytest)
    python run_tests.py -x              # stop on first failure
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.resolve()

TEST_FILES = [
    "tests/fitness_tests.py",
    "tests/individual_tests.py",
    "tests/population_tests.py",
    "tests/environment_tests.py",
]


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))

    extra = sys.argv[1:]
    args = ["-v", *[str(PROJECT_ROOT / f) for f in TEST_FILES], *extra]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
