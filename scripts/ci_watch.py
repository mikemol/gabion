from __future__ import annotations

from gabion.tooling.ci_watch import (
    CiWatchDeps,
    CollectionStatus,
    FailureCollectionResult,
    StatusWatchOptions,
    StatusWatchResult,
    _COLLECTION_FAILURE_EXIT,
    _DEFAULT_FAILURE_ARTIFACT_ROOT,
    _deadline_scope,
    main,
)

__all__ = [
    "CiWatchDeps",
    "CollectionStatus",
    "FailureCollectionResult",
    "StatusWatchOptions",
    "StatusWatchResult",
    "_COLLECTION_FAILURE_EXIT",
    "_DEFAULT_FAILURE_ARTIFACT_ROOT",
    "_deadline_scope",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
