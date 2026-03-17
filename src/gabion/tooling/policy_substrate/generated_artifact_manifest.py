from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

_MANIFEST_BEGIN = "<!-- BEGIN:generated_artifact_manifest -->"
_MANIFEST_END = "<!-- END:generated_artifact_manifest -->"


@dataclass(frozen=True)
class ArtifactFamily:
    family_id: str
    title: str
    description: str


@dataclass(frozen=True)
class GeneratedArtifactEntry:
    artifact_id: str
    family_id: str
    paths: tuple[str, ...]
    format: str
    process_domain: str
    emitter_kind: str
    source: str
    source_refs: tuple[str, ...]
    conditional: bool
    trigger_condition: str
    regeneration: tuple[str, ...]
    primary_consumers: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class GeneratedArtifactCatalog:
    version: int
    families: tuple[ArtifactFamily, ...]
    artifacts: tuple[GeneratedArtifactEntry, ...]


def _require_mapping(raw: object, *, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _require_list(raw: object, *, context: str) -> list[object]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list")
    return list(raw)


def _require_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw.strip()


def _require_bool(raw: object, *, context: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{context} must be a boolean")
    return raw


def _load_string_list(raw: object, *, context: str) -> tuple[str, ...]:
    values = tuple(
        _require_string(item, context=f"{context}[{index}]")
        for index, item in enumerate(_require_list(raw, context=context), start=1)
    )
    if not values:
        raise ValueError(f"{context} must be a non-empty list")
    return values


def load_catalog(path: Path) -> GeneratedArtifactCatalog:
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = _require_mapping(raw_payload, context="generated_artifact_manifest")
    version = payload.get("version")
    if not isinstance(version, int):
        raise ValueError("generated_artifact_manifest.version must be an integer")

    families = tuple(
        ArtifactFamily(
            family_id=_require_string(
                _require_mapping(raw_family, context=f"families[{index}]").get("family_id"),
                context=f"families[{index}].family_id",
            ),
            title=_require_string(
                _require_mapping(raw_family, context=f"families[{index}]").get("title"),
                context=f"families[{index}].title",
            ),
            description=_require_string(
                _require_mapping(raw_family, context=f"families[{index}]").get(
                    "description"
                ),
                context=f"families[{index}].description",
            ),
        )
        for index, raw_family in enumerate(
            _require_list(payload.get("families"), context="families"),
            start=1,
        )
    )

    artifacts = tuple(
        GeneratedArtifactEntry(
            artifact_id=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("artifact_id"),
                context=f"artifacts[{index}].artifact_id",
            ),
            family_id=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("family_id"),
                context=f"artifacts[{index}].family_id",
            ),
            paths=_load_string_list(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("paths"),
                context=f"artifacts[{index}].paths",
            ),
            format=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("format"),
                context=f"artifacts[{index}].format",
            ),
            process_domain=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "process_domain"
                ),
                context=f"artifacts[{index}].process_domain",
            ),
            emitter_kind=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "emitter_kind"
                ),
                context=f"artifacts[{index}].emitter_kind",
            ),
            source=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("source"),
                context=f"artifacts[{index}].source",
            ),
            source_refs=_load_string_list(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "source_refs"
                ),
                context=f"artifacts[{index}].source_refs",
            ),
            conditional=_require_bool(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "conditional"
                ),
                context=f"artifacts[{index}].conditional",
            ),
            trigger_condition=_require_string(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "trigger_condition"
                ),
                context=f"artifacts[{index}].trigger_condition",
            ),
            regeneration=_load_string_list(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "regeneration"
                ),
                context=f"artifacts[{index}].regeneration",
            ),
            primary_consumers=_load_string_list(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get(
                    "primary_consumers"
                ),
                context=f"artifacts[{index}].primary_consumers",
            ),
            notes=str(
                _require_mapping(raw_row, context=f"artifacts[{index}]").get("notes", "")
            ).strip(),
        )
        for index, raw_row in enumerate(
            _require_list(payload.get("artifacts"), context="artifacts"),
            start=1,
        )
    )

    catalog = GeneratedArtifactCatalog(
        version=version,
        families=families,
        artifacts=artifacts,
    )
    validate_catalog(catalog)
    return catalog


def _is_relative_repo_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and value != "." and not value.startswith("../")


def validate_catalog(catalog: GeneratedArtifactCatalog) -> None:
    family_ids: set[str] = set()
    for family in catalog.families:
        if family.family_id in family_ids:
            raise ValueError(f"duplicate family_id: {family.family_id}")
        family_ids.add(family.family_id)

    artifact_ids: set[str] = set()
    for artifact in catalog.artifacts:
        if artifact.artifact_id in artifact_ids:
            raise ValueError(f"duplicate artifact_id: {artifact.artifact_id}")
        artifact_ids.add(artifact.artifact_id)
        if artifact.family_id not in family_ids:
            raise ValueError(
                f"artifact {artifact.artifact_id} references unknown family_id {artifact.family_id}"
            )
        for rel_path in artifact.paths:
            if not _is_relative_repo_path(rel_path):
                raise ValueError(
                    f"artifact {artifact.artifact_id} path must be repo-relative: {rel_path}"
                )
        for rel_path in artifact.source_refs:
            if not _is_relative_repo_path(rel_path):
                raise ValueError(
                    f"artifact {artifact.artifact_id} source_ref must be repo-relative: {rel_path}"
                )


def validate_catalog_against_repo(
    catalog: GeneratedArtifactCatalog,
    *,
    repo_root: Path,
) -> None:
    validate_catalog(catalog)
    for artifact in catalog.artifacts:
        for rel_path in artifact.source_refs:
            if not (repo_root / rel_path).exists():
                raise ValueError(
                    f"artifact {artifact.artifact_id} source_ref does not exist: {rel_path}"
                )


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_list(values: tuple[str, ...]) -> str:
    return "<br>".join(f"`{_escape_cell(value)}`" for value in values)


def render_family_section(
    *,
    family: ArtifactFamily,
    artifacts: tuple[GeneratedArtifactEntry, ...],
) -> str:
    lines = [
        f"## {family.title}",
        "",
        family.description,
        "",
        "| Artifact ID | Path(s) | Format | Emitted by | Trigger | Regeneration | Primary consumers | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(
            "| {artifact_id} | {paths} | {format} | {emitted_by} | {trigger} | {regeneration} | {consumers} | {notes} |".format(
                artifact_id=f"`{_escape_cell(artifact.artifact_id)}`",
                paths=_format_list(artifact.paths),
                format=f"`{_escape_cell(artifact.format)}`",
                emitted_by=_escape_cell(
                    f"{artifact.emitter_kind}: {artifact.source}"
                ),
                trigger=_escape_cell(
                    ("conditional" if artifact.conditional else "always")
                    + f" - {artifact.trigger_condition}"
                ),
                regeneration=_format_list(artifact.regeneration),
                consumers=_format_list(artifact.primary_consumers),
                notes=_escape_cell(artifact.notes),
            )
        )
    return "\n".join(lines)


def render_manifest(catalog: GeneratedArtifactCatalog) -> str:
    family_by_id = {family.family_id: family for family in catalog.families}
    grouped: dict[str, list[GeneratedArtifactEntry]] = {
        family.family_id: [] for family in catalog.families
    }
    for artifact in catalog.artifacts:
        grouped[artifact.family_id].append(artifact)
    sections: list[str] = []
    for family in catalog.families:
        sections.append(
            render_family_section(
                family=family,
                artifacts=tuple(grouped[family.family_id]),
            )
        )
    return "\n\n".join(sections)


def render_manifest_block(catalog: GeneratedArtifactCatalog) -> str:
    return "\n".join(
        (
            _MANIFEST_BEGIN,
            "_This manifest section is generated from `docs/generated_artifact_manifest.yaml` "
            "via `mise exec -- python -m scripts.policy.render_generated_artifact_manifest`._",
            "",
            render_manifest(catalog),
            _MANIFEST_END,
        )
    )


def _replace_block(text: str, replacement: str) -> str:
    try:
        begin = text.index(_MANIFEST_BEGIN)
        end = text.index(_MANIFEST_END)
    except ValueError as exc:
        raise ValueError("generated artifact manifest markers are missing") from exc
    if end < begin:
        raise ValueError("generated artifact manifest markers are out of order")
    end += len(_MANIFEST_END)
    return text[:begin] + replacement + text[end:]


def run(
    *,
    catalog_path: Path,
    doc_path: Path,
    check: bool = False,
) -> int:
    catalog = load_catalog(catalog_path)
    rendered = render_manifest_block(catalog)
    existing = doc_path.read_text(encoding="utf-8")
    rewritten = _replace_block(existing, rendered)
    if check:
        return 0 if rewritten == existing else 1
    doc_path.write_text(rewritten, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/generated_artifact_manifest.yaml"),
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=Path("docs/generated_artifact_manifest.md"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    return run(
        catalog_path=args.catalog.resolve(),
        doc_path=args.doc.resolve(),
        check=bool(args.check),
    )


__all__ = [
    "ArtifactFamily",
    "GeneratedArtifactCatalog",
    "GeneratedArtifactEntry",
    "load_catalog",
    "main",
    "render_manifest",
    "render_manifest_block",
    "run",
    "validate_catalog",
    "validate_catalog_against_repo",
]
