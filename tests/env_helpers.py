from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def set_env(values: dict[str, str | None]) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@contextmanager
def env_scope(values: dict[str, str | None]) -> Iterator[None]:
    previous = set_env(values)
    try:
        yield
    finally:
        restore_env(previous)
