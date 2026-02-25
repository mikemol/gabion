from __future__ import annotations

from pathlib import Path

def _load():
    repo_root = Path(__file__).resolve().parents[1]
    from gabion.synthesis.model import NamingContext
    from gabion.synthesis.naming import _normalize_identifier, suggest_name

    return NamingContext, _normalize_identifier, suggest_name

# gabion:evidence E:function_site::naming.py::gabion.synthesis.naming._normalize_identifier E:function_site::naming.py::gabion.synthesis.naming.suggest_name E:decision_surface/direct::naming.py::gabion.synthesis.naming._normalize_identifier::stale_35b0eb6fe806
def test_suggest_name_uses_frequency() -> None:
    NamingContext, _normalize_identifier, suggest_name = _load()
    context = NamingContext(frequency={"ctx": 3, "config": 1})
    name = suggest_name(["config", "ctx"], context)
    assert name == "CtxBundle"
    assert _normalize_identifier("$$$", "Fallback") == "Fallback"

# gabion:evidence E:function_site::naming.py::gabion.synthesis.naming._normalize_identifier E:function_site::naming.py::gabion.synthesis.naming.suggest_name E:decision_surface/direct::naming.py::gabion.synthesis.naming._normalize_identifier::stale_56301240fc66_2035c37c
def test_suggest_name_avoids_collisions() -> None:
    NamingContext, _normalize_identifier, suggest_name = _load()
    context = NamingContext(existing_names={"CtxBundle"}, frequency={"ctx": 2})
    name = suggest_name(["ctx"], context)
    assert name == "CtxBundle2"
    assert _normalize_identifier("1value", "X") == "X1value"

# gabion:evidence E:function_site::naming.py::gabion.synthesis.naming.suggest_name E:decision_surface/direct::naming.py::gabion.synthesis.naming.suggest_name::stale_0df85d153806
def test_suggest_name_with_empty_fields() -> None:
    NamingContext, _normalize_identifier, suggest_name = _load()
    context = NamingContext(fallback_prefix="Fallback")
    name = suggest_name([], context)
    assert name == "FallbackBundle"
