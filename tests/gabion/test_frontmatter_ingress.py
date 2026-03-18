from __future__ import annotations

import pytest

from gabion.frontmatter import (
    FrontmatterParseError,
    parse_lenient_yaml_frontmatter,
    parse_strict_yaml_frontmatter,
)
from gabion.frontmatter_ingress import (
    FrontmatterDecompositionKind,
    FrontmatterParseMode,
    parse_frontmatter_document,
)
from gabion_governance import governance_audit_impl


# gabion:behavior primary=desired
def test_parse_frontmatter_document_exposes_typed_document_and_field_identities() -> None:
    carrier = parse_frontmatter_document(
        "\n".join(
            [
                "---",
                "doc_id: sample_policy_rules",
                "doc_revision: 1",
                "---",
                "## sample",
            ]
        ),
        source_path="docs/policy_rules/sample.md",
    )

    assert carrier.mode is FrontmatterParseMode.YAML
    assert carrier.payload == {
        "doc_id": "sample_policy_rules",
        "doc_revision": 1,
    }
    assert carrier.body == "## sample"
    assert carrier.identity.item_kind == "document"
    assert carrier.identity.item_key == "yaml"
    assert str(carrier.identity) == "docs/policy_rules/sample.md"
    assert carrier.identity.wire() != str(carrier.identity)
    assert {
        item.decomposition_kind for item in carrier.identity.decompositions
    } >= {
        FrontmatterDecompositionKind.CANONICAL,
        FrontmatterDecompositionKind.SOURCE_PATH,
        FrontmatterDecompositionKind.ITEM_KIND,
        FrontmatterDecompositionKind.ITEM_KEY,
    }
    assert tuple(field.field_name for field in carrier.fields) == (
        "doc_id",
        "doc_revision",
    )
    assert carrier.fields[0].identity.item_kind == "field"
    assert str(carrier.fields[0].identity) == "doc_id"
    assert carrier.fields[0].identity.wire() != str(carrier.fields[0].identity)


# gabion:behavior primary=desired
def test_parse_strict_yaml_frontmatter_preserves_boundary_projection_errors() -> None:
    with pytest.raises(FrontmatterParseError, match="unterminated YAML frontmatter"):
        parse_strict_yaml_frontmatter(
            "\n".join(
                [
                    "---",
                    "doc_id: sample_policy_rules",
                ]
            ),
            require_parser=True,
        )

    with pytest.raises(FrontmatterParseError, match="invalid YAML frontmatter"):
        parse_strict_yaml_frontmatter(
            "\n".join(
                [
                    "---",
                    "doc_id: [sample_policy_rules",
                    "---",
                ]
            ),
            require_parser=True,
        )

    with pytest.raises(
        FrontmatterParseError,
        match="frontmatter root must be a mapping",
    ):
        parse_strict_yaml_frontmatter(
            "\n".join(
                [
                    "---",
                    "- sample_policy_rules",
                    "---",
                ]
            ),
            require_parser=True,
        )


# gabion:behavior primary=desired
def test_parse_lenient_yaml_frontmatter_preserves_absent_unterminated_and_invalid_fallbacks() -> None:
    absent_payload, absent_body = parse_lenient_yaml_frontmatter("## sample")
    assert absent_payload == {}
    assert absent_body == "## sample"

    unterminated = "\n".join(
        [
            "---",
            "doc_id: sample_policy_rules",
        ]
    )
    unterminated_payload, unterminated_body = parse_lenient_yaml_frontmatter(
        unterminated
    )
    assert unterminated_payload == {}
    assert unterminated_body == unterminated

    invalid = "\n".join(
        [
            "---",
            "doc_id: [sample_policy_rules",
            "---",
            "## sample",
        ]
    )
    invalid_payload, invalid_body = parse_lenient_yaml_frontmatter(invalid)
    assert invalid_payload == {}
    assert invalid_body == "## sample"

    valid_payload, valid_body = parse_lenient_yaml_frontmatter(
        "\n".join(
            [
                "---",
                "doc_id: sample_policy_rules",
                "doc_revision: 1",
                "---",
                "## sample",
            ]
        )
    )
    assert valid_payload == {
        "doc_id": "sample_policy_rules",
        "doc_revision": 1,
    }
    assert valid_body == "## sample"


# gabion:behavior primary=desired
def test_governance_frontmatter_helpers_share_the_ingress_mode_surface() -> None:
    valid = "\n".join(
        [
            "---",
            "doc_id: sample_policy_rules",
            "---",
            "## sample",
        ]
    )
    invalid = "\n".join(
        [
            "---",
            "doc_id: [sample_policy_rules",
            "---",
            "## sample",
        ]
    )

    block = governance_audit_impl._frontmatter_block_from_text(valid)
    assert block == (["doc_id: sample_policy_rules"], "## sample")

    payload, body, mode, detail = governance_audit_impl._parse_frontmatter_with_mode(
        valid
    )
    assert payload == {"doc_id": "sample_policy_rules"}
    assert body == "## sample"
    assert mode == "yaml"
    assert detail is None

    invalid_payload, invalid_body, invalid_mode, invalid_detail = (
        governance_audit_impl._parse_frontmatter_with_mode(invalid)
    )
    assert invalid_payload == {}
    assert invalid_body == "## sample"
    assert invalid_mode == "yaml_parse_failed"
    assert invalid_detail
