import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "bench"))

from synth import building_plan, scan  # noqa: E402


@pytest.fixture(scope="session")
def plan():
    return building_plan(7)


@pytest.fixture(scope="session")
def plan_scan(plan):
    return scan(plan, 7, dpi=200)
