from __future__ import annotations

from gabion import server


def test_normalize_transparent_decorators() -> None:
    assert server._normalize_transparent_decorators(None) is None
    assert server._normalize_transparent_decorators("a, b") == {"a", "b"}
    assert server._normalize_transparent_decorators(["a", "b,c"]) == {"a", "b", "c"}
    assert server._normalize_transparent_decorators([]) is None


def test_uri_to_path_file_scheme() -> None:
    path = server._uri_to_path("file:///tmp/example.py")
    assert str(path).endswith("/tmp/example.py")


def test_uri_to_path_plain_path() -> None:
    path = server._uri_to_path("/tmp/example.py")
    assert str(path).endswith("/tmp/example.py")
