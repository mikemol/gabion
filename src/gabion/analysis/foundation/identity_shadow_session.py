# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.identity_registry_mirror import (
    IdentityRegistryMirror,
    build_identity_registry_mirror,
)
from gabion.analysis.foundation.identity_shadow_runtime import (
    DEFAULT_IDENTITY_SHADOW_RUN_ID,
    IdentityShadowRuntime,
    build_identity_shadow_runtime,
)
from gabion.analysis.foundation.timeout_context import check_deadline


def derive_identity_shadow_run_id(*, root: Path) -> str:
    check_deadline()
    return f"{DEFAULT_IDENTITY_SHADOW_RUN_ID}:{sha1(str(root).encode('utf-8')).hexdigest()}"


@dataclass
class IdentityShadowSession:
    runtime: IdentityShadowRuntime
    registry_mirror: IdentityRegistryMirror
    _started: bool = False

    def start(self) -> None:
        if self._started:
            return
        self.registry_mirror.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.registry_mirror.stop()
        except Exception:
            pass
        self._started = False

    def identity_seed_payload(self) -> dict[str, object]:
        check_deadline()
        return self.runtime.identity_seed_payload()

    def __enter__(self) -> "IdentityShadowSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def build_identity_shadow_session(
    *,
    root: Path,
    registry: PrimeRegistry,
) -> IdentityShadowSession:
    check_deadline()
    runtime = build_identity_shadow_runtime(
        run_id=derive_identity_shadow_run_id(root=root),
        registry=registry,
    )
    return IdentityShadowSession(
        runtime=runtime,
        registry_mirror=build_identity_registry_mirror(
            registry=registry,
            identity_space=runtime.run_context.identity_space,
        ),
    )


__all__ = [
    "IdentityShadowSession",
    "build_identity_shadow_session",
    "derive_identity_shadow_run_id",
]
