from __future__ import annotations

import io
import json

from gabion.lsp_client import LspClientError, _read_response, _read_rpc


def _rpc_message(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


def test_read_rpc_invalid_length() -> None:
    stream = io.BytesIO(b"Content-Length: 0\r\n\r\n{}")
    try:
        _read_rpc(stream)
    except LspClientError as exc:
        assert "Content-Length" in str(exc)
    else:
        raise AssertionError("Expected LspClientError for invalid Content-Length")


def test_read_rpc_missing_content_length_header() -> None:
    stream = io.BytesIO(b"Foo: bar\r\n\r\n{}")
    try:
        _read_rpc(stream)
    except LspClientError as exc:
        assert "Content-Length" in str(exc)
    else:
        raise AssertionError("Expected LspClientError for missing Content-Length")


def test_read_rpc_stream_closed() -> None:
    stream = io.BytesIO(b"")
    try:
        _read_rpc(stream)
    except LspClientError as exc:
        assert "stream closed" in str(exc).lower()
    else:
        raise AssertionError("Expected LspClientError for closed stream")


def test_read_response_skips_unmatched_ids() -> None:
    msg1 = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    msg2 = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {"answer": 42}})
    stream = io.BytesIO(msg1 + msg2)
    response = _read_response(stream, 2)
    assert response["id"] == 2
    assert response["result"]["answer"] == 42


def test_read_rpc_skips_non_content_length_headers() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": {}}
    body = json.dumps(payload).encode("utf-8")
    header = b"Foo: bar\r\nContent-Length: " + str(len(body)).encode("utf-8") + b"\r\n\r\n"
    stream = io.BytesIO(header + body)
    message = _read_rpc(stream)
    assert message["id"] == 1


def test_read_rpc_accepts_prefetched_body() -> None:
    payload = {"jsonrpc": "2.0", "id": 9, "result": {"ok": True}}
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")

    class _Chunky:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._sent = False

        def read(self, _n: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return self._data

    stream = _Chunky(header + body)
    message = _read_rpc(stream)
    assert message["id"] == 9
