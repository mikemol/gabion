from __future__ import annotations

import os
import pytest

from gabion import invariants
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::invariants.py::gabion.invariants.never
def test_never_raises_never_thrown() -> None:
    with pytest.raises(NeverThrown):
        invariants.never("boom", flag=True)


def test_require_not_none_non_strict() -> None:
    assert invariants.require_not_none(None, strict=False) is None
    assert invariants.require_not_none("ok", strict=False) == "ok"


def test_require_not_none_strict_raises() -> None:
    with pytest.raises(NeverThrown):
        invariants.require_not_none(None, strict=True)


def test_require_not_none_env_strict() -> None:
    key = "GABION_PROOF_MODE"
    previous = os.environ.get(key)
    os.environ[key] = "strict"
    try:
        with pytest.raises(NeverThrown):
            invariants.require_not_none(None)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
