from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import schema_audit as sa

    return sa


def test_find_anonymous_schema_surfaces_finds_common_sites(tmp_path: Path) -> None:
    sa = _load()
    path = tmp_path / "mod.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from typing import Any, Mapping\n"
        "\n"
        "class Service:\n"
        "    def method(self, payload: dict[str, object]) -> dict[str, Any]:\n"
        "        local: list[dict[str, object]] = []\n"
        "        self.payload: dict[str, object] = {}\n"
        "        return {}\n"
        "\n"
        "async def fetch() -> dict[str, object]:\n"
        "    return {}\n"
        "\n"
        "def top(ctx: Mapping[str, Any]) -> None:\n"
        "    return None\n"
    )

    surfaces = sa.find_anonymous_schema_surfaces([path], project_root=tmp_path)
    assert surfaces
    assert {surface.path for surface in surfaces} == {"mod.py"}

    contexts = {surface.context for surface in surfaces}
    assert "Service.method(payload)" in contexts
    assert "Service.method.returns" in contexts
    assert "Service.method.local" in contexts
    assert "Service.method.self.payload" in contexts
    assert "fetch.returns" in contexts
    assert "top(ctx)" in contexts

    payload = next(surface for surface in surfaces if surface.context == "Service.method(payload)")
    assert payload.annotation == "dict[str, object]"
    assert payload.suggestion == "Payload"

    returns = next(surface for surface in surfaces if surface.context == "Service.method.returns")
    assert returns.annotation == "dict[str, Any]"
    assert returns.suggestion == "Method"

    attr = next(surface for surface in surfaces if surface.context == "Service.method.self.payload")
    assert attr.suggestion is None


def test_find_anonymous_schema_surfaces_ignores_test_roles(tmp_path: Path) -> None:
    sa = _load()
    test_prefixed = tmp_path / "test_mod.py"
    test_prefixed.write_text("def f(x: dict[str, object]) -> None:\n    return None\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    nested = tests_dir / "mod.py"
    nested.write_text("def f(x: dict[str, object]) -> None:\n    return None\n")
    dotfile = tmp_path / ".hidden.py"
    dotfile.write_text("def f(x: dict[str, object]) -> None:\n    return None\n")
    missing = tmp_path / "missing.py"
    syntax_error = tmp_path / "broken.py"
    syntax_error.write_text("def f(x: dict[str, object]) -> None\n    return None\n")

    assert sa.find_anonymous_schema_surfaces(
        [test_prefixed, nested, dotfile, missing, syntax_error], project_root=tmp_path
    ) == []


def test_suggest_type_name_singularizes_and_handles_provenance() -> None:
    sa = _load()
    assert sa._suggest_type_name("deadness_witnesses") == "DeadnessWitness"
    assert sa._suggest_type_name("entries") == "Entry"
    assert sa._suggest_type_name("cats") == "Cat"
    assert sa._suggest_type_name("class") == "Class"
    assert sa._suggest_type_name("fingerprint_provenance") == "FingerprintProvenanceEntry"
    assert sa._suggest_type_name("___") is None


def test_normalize_path_outside_root_returns_absolute(tmp_path: Path) -> None:
    sa = _load()
    root = tmp_path / "root"
    root.mkdir()
    path = tmp_path / "mod.py"
    assert sa._normalize_path(path, root) == str(path)


def test_subscript_helpers_cover_non_tuple_slices() -> None:
    sa = _load()
    tree = sa.ast.parse("x: list[int]\n")
    ann = tree.body[0].annotation
    assert isinstance(ann, sa.ast.Subscript)
    args = sa._subscript_args(ann)
    assert len(args) == 1

    dict_tree = sa.ast.parse("y: dict[str]\n")
    dict_ann = dict_tree.body[0].annotation
    assert isinstance(dict_ann, sa.ast.Subscript)
    assert sa._is_anonymous_dict_subscript(dict_ann) is False


def test_name_handles_attribute_and_unknown_nodes() -> None:
    sa = _load()
    attr = sa.ast.Attribute(value=sa.ast.Name(id="typing", ctx=sa.ast.Load()), attr="Dict")
    assert sa._name(attr) == "Dict"
    assert sa._name(sa.ast.Constant(value=1)) is None


def test_unparse_fallback_for_invalid_ast() -> None:
    sa = _load()
    # ast.unparse expects well-formed nodes; this intentionally violates that.
    bad = sa.ast.Name(id=None, ctx=sa.ast.Load())
    assert sa._unparse(bad) == "<annotation>"
