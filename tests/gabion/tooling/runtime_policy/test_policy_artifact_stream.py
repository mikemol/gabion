from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_substrate import policy_artifact_stream as stream
from gabion.tooling.policy_substrate.policy_queue_identity import PolicyQueueIdentitySpace


# gabion:behavior primary=desired
def test_write_json_matches_render_json_value_for_recursive_artifact_units(
    tmp_path: Path,
) -> None:
    artifact = stream.document(
        identity="root",
        title="root",
        children=lambda: iter(
            (
                stream.section(
                    identity="meta",
                    key="meta",
                    title="meta",
                    children=lambda: iter(
                        (
                            stream.scalar(
                                identity="queue_id",
                                key="queue_id",
                                title="queue_id",
                                value="PSF-007",
                            ),
                            stream.lazy(
                                identity="lazy-single",
                                children=lambda: iter(
                                    (
                                        stream.scalar(
                                            identity="status",
                                            key="status",
                                            title="status",
                                            value="in_progress",
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                stream.bullet_list(
                    identity="items",
                    key="items",
                    title="items",
                    children=lambda: iter(
                        (
                            stream.list_item(
                                identity="item-0",
                                children=lambda: iter(
                                    (
                                        stream.scalar(
                                            identity="object_id",
                                            key="object_id",
                                            title="object_id",
                                            value="PSF-007-TP-005",
                                        ),
                                        stream.scalar(
                                            identity="count",
                                            key="count",
                                            title="count",
                                            value=1,
                                        ),
                                    )
                                ),
                            ),
                            stream.list_item(
                                identity="item-1",
                                value="done",
                            ),
                        )
                    ),
                ),
                stream.table(
                    identity="rows",
                    key="rows",
                    title="rows",
                    columns=(
                        stream.ArtifactColumn(key="object_id", title="object_id"),
                        stream.ArtifactColumn(key="count", title="count"),
                    ),
                    children=lambda: iter(
                        (
                            stream.row(
                                identity="row-0",
                                children=lambda: iter(
                                    (
                                        stream.cell(
                                            identity="row-0-object",
                                            key="object_id",
                                            title="object_id",
                                            value="PSF-007-SQ-001",
                                        ),
                                        stream.cell(
                                            identity="row-0-count",
                                            key="count",
                                            title="count",
                                            value=5,
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        ),
    )

    output_path = tmp_path / "artifact.json"
    stream.write_json(output_path, artifact)

    assert json.loads(output_path.read_text(encoding="utf-8")) == stream.render_json_value(
        artifact
    )


# gabion:behavior primary=desired
def test_write_json_does_not_delegate_to_render_json_value(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact = stream.document(
        identity="root",
        children=lambda: iter(
            (
                stream.scalar(
                    identity="value",
                    key="value",
                    title="value",
                    value="ok",
                ),
            )
        ),
    )

    def _boom(_unit: object) -> object:
        raise AssertionError("render_json_value should not be called")

    monkeypatch.setattr(stream, "render_json_value", _boom)

    output_path = tmp_path / "artifact.json"
    stream.write_json(output_path, artifact)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"value": "ok"}


# gabion:behavior primary=desired
def test_artifact_stream_uses_default_renderer_for_typed_identities() -> None:
    identity_space = PolicyQueueIdentitySpace()
    touchpoint = identity_space.touchpoint_id("PSF-007-TP-005")
    touchsite = identity_space.touchsite_id("PSF-007-TS:semantic_fragment")

    artifact = stream.document(
        identity=touchpoint,
        title="workstream",
        children=lambda: iter(
            (
                stream.scalar(
                    identity=touchpoint,
                    key="recommended_cut",
                    title="recommended_cut",
                    value=touchpoint,
                ),
                stream.bullet_list(
                    identity=touchsite,
                    key="touchsites",
                    title="touchsites",
                    children=lambda: iter(
                        (
                            stream.list_item(
                                identity=touchsite,
                                children=lambda: iter(
                                    (
                                        stream.scalar(
                                            identity=touchsite,
                                            key="touchsite",
                                            title="touchsite",
                                            value=touchsite,
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        ),
    )

    rendered = str(artifact)
    assert "recommended_cut" in rendered
    assert "PSF-007-TP-005" in rendered
    assert "PSF-007-TS:semantic_fragment" in rendered
    assert stream.render_json_value(artifact) == {
        "recommended_cut": "PSF-007-TP-005",
        "touchsites": [{"touchsite": "PSF-007-TS:semantic_fragment"}],
    }
