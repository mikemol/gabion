from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest

from gabion.analysis.timeout_context import Deadline, deadline_scope
from gabion.analysis.timeout_context import forest_scope
from gabion.analysis.aspf import Forest


@pytest.fixture(autouse=True)
def _deadline_scope_fixture():
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ms(120_000)):
            yield
