from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

from gabion.frontmatter import parse_strict_yaml_frontmatter

_PLAYBOOK_BEGIN = "<!-- BEGIN:generated_policy_rule_playbooks -->"
_PLAYBOOK_END = "<!-- END:generated_policy_rule_playbooks -->"


@dataclass(frozen=True)
class PlaybookReference:
    label: str
    href: str


@dataclass(frozen=True)
class PlaybookSection:
    rule_id: str
    anchor: str
    meaning: str
    preferred: tuple[str, ...]
    avoid: tuple[str, ...]
    references: tuple[PlaybookReference, ...]


def _require_mapping(raw: object, *, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(raw)


def _require_string(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw.strip()


def _string_items(raw: object, *, context: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return () if not text else (text,)
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a string or list of strings")
    values: list[str] = []
    for index, item in enumerate(raw, start=1):
        values.append(_require_string(item, context=f"{context}[{index}]"))
    return tuple(values)


def _references_by_rule(frontmatter: dict[str, object]) -> dict[str, tuple[PlaybookReference, ...]]:
    raw_rendering = frontmatter.get("playbook_rendering")
    if raw_rendering is None:
        return {}
    rendering = _require_mapping(raw_rendering, context="playbook_rendering")
    raw_references = rendering.get("references")
    if raw_references is None:
        return {}
    references_by_rule: dict[str, tuple[PlaybookReference, ...]] = {}
    for rule_id, raw_entries in _require_mapping(raw_references, context="playbook_rendering.references").items():
        entries: list[PlaybookReference] = []
        if not isinstance(raw_entries, list):
            raise ValueError(f"playbook_rendering.references.{rule_id} must be a list")
        for index, raw_entry in enumerate(raw_entries, start=1):
            entry = _require_mapping(
                raw_entry,
                context=f"playbook_rendering.references.{rule_id}[{index}]",
            )
            entries.append(
                PlaybookReference(
                    label=_require_string(
                        entry.get("label"),
                        context=f"playbook_rendering.references.{rule_id}[{index}].label",
                    ),
                    href=_require_string(
                        entry.get("href"),
                        context=f"playbook_rendering.references.{rule_id}[{index}].href",
                    ),
                )
            )
        references_by_rule[str(rule_id)] = tuple(entries)
    return references_by_rule


def load_playbook_sections(path: Path) -> tuple[PlaybookSection, ...]:
    text = path.read_text(encoding="utf-8")
    frontmatter_raw, _ = parse_strict_yaml_frontmatter(text, require_parser=True)
    frontmatter = _require_mapping(frontmatter_raw, context=f"{path}.frontmatter")
    rules = frontmatter.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"{path} must define rules: [] in frontmatter")
    references_by_rule = _references_by_rule(frontmatter)
    sections: list[PlaybookSection] = []
    for index, raw_rule in enumerate(rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"{path}.rules[{index}] must be a mapping")
        rule = _require_mapping(raw_rule, context=f"{path}.rules[{index}]")
        raw_anchor = rule.get("playbook_anchor")
        if raw_anchor is None:
            continue
        anchor = _require_string(raw_anchor, context=f"{path}.rules[{index}].playbook_anchor")
        rule_id = _require_string(rule.get("rule_id"), context=f"{path}.rules[{index}].rule_id")
        outcome = _require_mapping(rule.get("outcome"), context=f"{path}.rules[{index}].outcome")
        guidance = _require_mapping(
            outcome.get("guidance", {}),
            context=f"{path}.rules[{index}].outcome.guidance",
        )
        meaning = _require_string(guidance.get("why"), context=f"{path}.rules[{index}].outcome.guidance.why")
        sections.append(
            PlaybookSection(
                rule_id=rule_id,
                anchor=anchor,
                meaning=meaning,
                preferred=_string_items(
                    guidance.get("prefer"),
                    context=f"{path}.rules[{index}].outcome.guidance.prefer",
                ),
                avoid=_string_items(
                    guidance.get("avoid"),
                    context=f"{path}.rules[{index}].outcome.guidance.avoid",
                ),
                references=references_by_rule.get(rule_id, ()),
            )
        )
    return tuple(sections)


def _render_reference_line(references: tuple[PlaybookReference, ...]) -> list[str]:
    if not references:
        return []
    label = "Reference" if len(references) == 1 else "References"
    joined = ", ".join(f"[{item.label}]({item.href})" for item in references)
    return [f"{label}: {joined}."]


def render_playbook_block(*, path: Path) -> str:
    sections = load_playbook_sections(path)
    rendered_sections: list[str] = []
    for section in sections:
        lines = [
            f'<a id="{section.anchor}"></a>',
            f"## `{section.rule_id}`",
            "",
            f"Meaning: {section.meaning}",
        ]
        if section.preferred:
            lines.extend(("", "Preferred response:"))
            lines.extend(f"- {item}" for item in section.preferred)
        if section.avoid:
            lines.extend(("", "Avoid:"))
            lines.extend(f"- {item}" for item in section.avoid)
        lines.extend(_render_reference_line(section.references))
        rendered_sections.append("\n".join(lines))
    body = "\n\n".join(rendered_sections)
    return "\n".join(
        (
            _PLAYBOOK_BEGIN,
            (
                "_The playbook sections below are generated from this document's `rules:` "
                "frontmatter via `mise exec -- python -m scripts.policy.render_policy_rule_playbooks`._"
            ),
            "",
            body,
            _PLAYBOOK_END,
        )
    )


def _replace_block(*, text: str, begin: str, end: str, replacement: str) -> str:
    start = text.find(begin)
    if start < 0:
        raise ValueError(f"missing marker {begin}")
    finish = text.find(end, start)
    if finish < 0:
        raise ValueError(f"missing marker {end}")
    finish += len(end)
    return text[:start] + replacement + text[finish:]


def _rewrite_doc(*, path: Path, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    rendered = _replace_block(
        text=original,
        begin=_PLAYBOOK_BEGIN,
        end=_PLAYBOOK_END,
        replacement=render_playbook_block(path=path),
    )
    if check:
        return original != rendered
    if original != rendered:
        path.write_text(rendered, encoding="utf-8")
    return False


def run(
    *,
    ambiguity_contract_doc_path: Path,
    grade_monotonicity_doc_path: Path,
    check: bool = False,
) -> int:
    changed = any(
        (
            _rewrite_doc(path=ambiguity_contract_doc_path, check=check),
            _rewrite_doc(path=grade_monotonicity_doc_path, check=check),
        )
    )
    return 1 if check and changed else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render policy-rule playbook sections from markdown frontmatter.",
    )
    parser.add_argument(
        "--ambiguity-contract-doc",
        type=Path,
        default=Path("docs/policy_rules/ambiguity_contract.md"),
    )
    parser.add_argument(
        "--grade-monotonicity-doc",
        type=Path,
        default=Path("docs/policy_rules/grade_monotonicity.md"),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        ambiguity_contract_doc_path=args.ambiguity_contract_doc,
        grade_monotonicity_doc_path=args.grade_monotonicity_doc,
        check=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
