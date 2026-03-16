from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re

import yaml

_CLAUSE_HEADING_RE = re.compile(
    r'<a id="(?P<anchor>[^"]+)"></a>\s*\n### `(?P<clause_id>[^`]+)`',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ClauseIndexEntry:
    clause_id: str
    anchor: str


@dataclass(frozen=True)
class ClauseDeckEntry:
    clause_id: str
    template: str


@dataclass(frozen=True)
class ClauseDeck:
    deck_id: str
    doc_path: Path
    begin_marker: str
    end_marker: str
    entries: tuple[ClauseDeckEntry, ...]


@dataclass(frozen=True)
class ClauseObligationDeckCatalog:
    version: int
    decks: tuple[ClauseDeck, ...]


def _require_mapping(raw: object, *, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _require_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw.strip()


def _require_list(raw: object, *, context: str) -> list[object]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list")
    return list(raw)


def load_catalog(path: Path) -> ClauseObligationDeckCatalog:
    payload = _require_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        context="clause_obligation_decks",
    )
    version = payload.get("version")
    if not isinstance(version, int):
        raise ValueError("clause_obligation_decks.version must be an integer")
    decks: list[ClauseDeck] = []
    for index, raw_deck in enumerate(
        _require_list(payload.get("decks"), context="clause_obligation_decks.decks"),
        start=1,
    ):
        deck = _require_mapping(raw_deck, context=f"clause_obligation_decks.decks[{index}]")
        entries: list[ClauseDeckEntry] = []
        for entry_index, raw_entry in enumerate(
            _require_list(
                deck.get("entries"),
                context=f"clause_obligation_decks.decks[{index}].entries",
            ),
            start=1,
        ):
            entry = _require_mapping(
                raw_entry,
                context=f"clause_obligation_decks.decks[{index}].entries[{entry_index}]",
            )
            entries.append(
                ClauseDeckEntry(
                    clause_id=_require_string(
                        entry.get("clause_id"),
                        context=(
                            "clause_obligation_decks.decks"
                            f"[{index}].entries[{entry_index}].clause_id"
                        ),
                    ),
                    template=_require_string(
                        entry.get("template"),
                        context=(
                            "clause_obligation_decks.decks"
                            f"[{index}].entries[{entry_index}].template"
                        ),
                    ),
                )
            )
        decks.append(
            ClauseDeck(
                deck_id=_require_string(
                    deck.get("deck_id"),
                    context=f"clause_obligation_decks.decks[{index}].deck_id",
                ),
                doc_path=Path(
                    _require_string(
                        deck.get("doc_path"),
                        context=f"clause_obligation_decks.decks[{index}].doc_path",
                    )
                ),
                begin_marker=_require_string(
                    deck.get("begin_marker"),
                    context=f"clause_obligation_decks.decks[{index}].begin_marker",
                ),
                end_marker=_require_string(
                    deck.get("end_marker"),
                    context=f"clause_obligation_decks.decks[{index}].end_marker",
                ),
                entries=tuple(entries),
            )
        )
    return ClauseObligationDeckCatalog(version=version, decks=tuple(decks))


def load_clause_index(path: Path) -> dict[str, ClauseIndexEntry]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, ClauseIndexEntry] = {}
    for match in _CLAUSE_HEADING_RE.finditer(text):
        clause_id = match.group("clause_id")
        if clause_id in entries:
            raise ValueError(f"duplicate clause id in clause index: {clause_id}")
        entries[clause_id] = ClauseIndexEntry(
            clause_id=clause_id,
            anchor=match.group("anchor"),
        )
    if not entries:
        raise ValueError("no clause headings found in normative clause index")
    return entries


def _clause_link(*, clause_id: str, clause_index: dict[str, ClauseIndexEntry]) -> str:
    entry = clause_index.get(clause_id)
    if entry is None:
        raise ValueError(f"unknown clause id in clause obligation deck: {clause_id}")
    return f"[`{entry.clause_id}`](docs/normative_clause_index.md#{entry.anchor})"


def render_deck_block(*, deck: ClauseDeck, clause_index: dict[str, ClauseIndexEntry]) -> str:
    bullets = [
        "- " + entry.template.format(
            clause_link=_clause_link(clause_id=entry.clause_id, clause_index=clause_index)
        )
        for entry in deck.entries
    ]
    return "\n".join(
        (
            deck.begin_marker,
            (
                "_The clause-backed bullets below are generated from "
                "`docs/clause_obligation_decks.yaml` and "
                "`docs/normative_clause_index.md` via "
                "`mise exec -- python -m scripts.policy.render_clause_obligation_decks`._"
            ),
            "",
            *bullets,
            deck.end_marker,
        )
    )


def _replace_block(
    *,
    document_text: str,
    begin_marker: str,
    end_marker: str,
    replacement: str,
) -> str:
    start = document_text.find(begin_marker)
    if start < 0:
        raise ValueError(f"missing marker {begin_marker}")
    end = document_text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"missing marker {end_marker}")
    end += len(end_marker)
    return document_text[:start] + replacement + document_text[end:]


def run(
    *,
    repo_root: Path,
    catalog_path: Path,
    clause_index_path: Path,
    check: bool = False,
) -> int:
    catalog = load_catalog(catalog_path)
    clause_index = load_clause_index(clause_index_path)
    drifted = False
    for deck in catalog.decks:
        doc_path = repo_root / deck.doc_path
        current = doc_path.read_text(encoding="utf-8")
        rendered = _replace_block(
            document_text=current,
            begin_marker=deck.begin_marker,
            end_marker=deck.end_marker,
            replacement=render_deck_block(deck=deck, clause_index=clause_index),
        )
        if check:
            drifted = drifted or rendered != current
            continue
        if rendered != current:
            doc_path.write_text(rendered, encoding="utf-8")
    return 1 if check and drifted else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render clause-backed obligation decks for governance docs.",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/clause_obligation_decks.yaml"),
    )
    parser.add_argument(
        "--clause-index",
        type=Path,
        default=Path("docs/normative_clause_index.md"),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        repo_root=args.root,
        catalog_path=args.root / args.catalog,
        clause_index_path=args.root / args.clause_index,
        check=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
