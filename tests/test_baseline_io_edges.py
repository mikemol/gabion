from __future__ import annotations

import pytest

from gabion.analysis.baseline_io import parse_version


def test_parse_version_rejects_empty_allowed_versions() -> None:
    with pytest.raises(ValueError):
        parse_version({}, expected=())
