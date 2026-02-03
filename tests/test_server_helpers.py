from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion import server

    return server


def test_normalize_transparent_decorators() -> None:
    server = _load()
    assert server._normalize_transparent_decorators(None) is None
    assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert server._normalize_transparent_decorators(["a", "b, c"]) == {"a", "b", "c"}
    assert server._normalize_transparent_decorators([1, "a"]) == {"a"}
    assert server._normalize_transparent_decorators([]) is None


def test_uri_to_path() -> None:
    server = _load()
    path = Path("/tmp/demo.txt")
    assert server._uri_to_path(path.as_uri()) == path
    assert server._uri_to_path("relative/path.py") == Path("relative/path.py")


def test_diagnostics_for_path_reports_bundle(tmp_path: Path) -> None:
    server = _load()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def callee(x):\n"
        "    return x\n"
        "\n"
        "def caller(a, b):\n"
        "    callee(a)\n"
        "    callee(b)\n"
    )
    diagnostics = server._diagnostics_for_path(str(sample), tmp_path)
    assert diagnostics
    assert any("Implicit bundle" in diag.message for diag in diagnostics)


def test_start_uses_injected_callable() -> None:
    server = _load()
    called = {"value": False}

    def _start() -> None:
        called["value"] = True

    server.start(_start)
    assert called["value"] is True
