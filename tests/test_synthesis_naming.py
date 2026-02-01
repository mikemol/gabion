from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.synthesis.model import NamingContext
    from gabion.synthesis.naming import suggest_name

    return NamingContext, suggest_name


def test_suggest_name_uses_frequency() -> None:
    NamingContext, suggest_name = _load()
    context = NamingContext(frequency={"ctx": 3, "config": 1})
    name = suggest_name(["config", "ctx"], context)
    assert name == "CtxBundle"


def test_suggest_name_avoids_collisions() -> None:
    NamingContext, suggest_name = _load()
    context = NamingContext(existing_names={"CtxBundle"}, frequency={"ctx": 2})
    name = suggest_name(["ctx"], context)
    assert name == "CtxBundle2"
