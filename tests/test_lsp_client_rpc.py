from __future__ import annotations

import io
import json
import time

from gabion.lsp_client import LspClientError, _read_exact, _read_response, _read_rpc


def _rpc_message(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
def test_read_rpc_invalid_length() -> None:
    stream = io.BytesIO(b"Content-Length: 0\r\n\r\n{}")
    deadline_ns = time.monotonic_ns() + 1_000_000_000
    try:
        _read_rpc(stream, deadline_ns)
    except LspClientError as exc:
        assert "Content-Length" in str(exc)
    else:
        raise AssertionError("Expected LspClientError for invalid Content-Length")


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
def test_read_rpc_missing_content_length_header() -> None:
    stream = io.BytesIO(b"Foo: bar\r\n\r\n{}")
    deadline_ns = time.monotonic_ns() + 1_000_000_000
    try:
        _read_rpc(stream, deadline_ns)
    except LspClientError as exc:
        assert "Content-Length" in str(exc)
    else:
        raise AssertionError("Expected LspClientError for missing Content-Length")


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
def test_read_rpc_stream_closed() -> None:
    stream = io.BytesIO(b"")
    deadline_ns = time.monotonic_ns() + 1_000_000_000
    try:
        _read_rpc(stream, deadline_ns)
    except LspClientError as exc:
        assert "stream closed" in str(exc).lower()
    else:
        raise AssertionError("Expected LspClientError for closed stream")


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::request_id
def test_read_response_skips_unmatched_ids() -> None:
    msg1 = _rpc_message({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    msg2 = _rpc_message({"jsonrpc": "2.0", "id": 2, "result": {"answer": 42}})
    stream = io.BytesIO(msg1 + msg2)
    response = _read_response(stream, 2, time.monotonic_ns() + 1_000_000_000)
    assert response["id"] == 2
    assert response["result"]["answer"] == 42


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
def test_read_rpc_skips_non_content_length_headers() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": {}}
    body = json.dumps(payload).encode("utf-8")
    header = b"Foo: bar\r\nContent-Length: " + str(len(body)).encode("utf-8") + b"\r\n\r\n"
    stream = io.BytesIO(header + body)
    message = _read_rpc(stream, time.monotonic_ns() + 1_000_000_000)
    assert message["id"] == 1


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
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
    message = _read_rpc(stream, time.monotonic_ns() + 1_000_000_000)
    assert message["id"] == 9


# gabion:evidence E:function_site::lsp_client.py::gabion.lsp_client._read_rpc
def test_read_rpc_rejects_non_object_payload() -> None:
    body = json.dumps([]).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream = io.BytesIO(header + body)
    try:
        _read_rpc(stream, time.monotonic_ns() + 1_000_000_000)
    except LspClientError as exc:
        assert "payload" in str(exc).lower()
    else:
        raise AssertionError("Expected LspClientError for non-object payload")


def test_read_exact_rejects_closed_stream() -> None:
    stream = io.BytesIO(b"")
    try:
        _read_exact(stream, 1, time.monotonic_ns() + 1_000_000_000)
    except LspClientError as exc:
        assert "stream closed" in str(exc).lower()
    else:
        raise AssertionError("Expected LspClientError for closed stream")


def test_read_rpc_truncates_prefetched_excess_body() -> None:
    payload = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
    body = json.dumps(payload).encode("utf-8")

    class _Chunky:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._sent = False

        def read(self, _n: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return self._data

    stream = _Chunky(
        f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8") + body + b"extra-bytes"
    )
    message = _read_rpc(stream, time.monotonic_ns() + 1_000_000_000)
    assert message["id"] == 7


# gabion:evidence E:decision_surface/direct::lsp_client.py::gabion.lsp_client._read_response::notification_callback
def test_read_response_dispatches_notifications() -> None:
    notification = _rpc_message(
        {"jsonrpc": "2.0", "method": "$/progress", "params": {"value": 1}}
    )
    response_msg = _rpc_message({"jsonrpc": "2.0", "id": 8, "result": {"answer": 7}})
    stream = io.BytesIO(notification + response_msg)
    seen: list[dict] = []

    response = _read_response(
        stream,
        8,
        time.monotonic_ns() + 1_000_000_000,
        notification_callback=seen.append,
    )

    assert response["id"] == 8
    assert len(seen) == 1
    assert seen[0]["method"] == "$/progress"
