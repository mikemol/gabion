from __future__ import annotations

import pytest

from gabion import invariants
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_never_raises_never_thrown() -> None:
    with pytest.raises(NeverThrown):
        invariants.never("boom", flag=True)
