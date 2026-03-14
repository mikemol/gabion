# gabion:decision_protocol_module
# gabion:ambiguity_boundary_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from gabion.analysis.core.prime_identity_adapter import PrimeIdentityAdapter
from gabion.analysis.core.type_fingerprints import PrimeRegistry
from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.deadline_clock import MonotonicClock


@dataclass(frozen=True)
class KernelVmObjectImage:
    object_id: int
    zone_id: int
    source_path_ids: tuple[int, ...]
    class_term_id: int
    supporting_term_ids: tuple[int, ...]
    ontology_term_ids: tuple[int, ...]
    shape_term_ids: tuple[int, ...]
    example_term_ids: tuple[int, ...]
    label: str


_REPO_ROOT = Path(__file__).resolve().parents[4]
_ONTOLOGY_PATH = "in/lg_kernel_ontology_cut_elim-1.ttl"
_SHAPES_PATH = "in/lg_kernel_shapes_cut_elim-1.ttl"
_EXAMPLE_PATH = "in/lg_kernel_example_cut_elim-1.ttl"
_SOURCE_PATHS = (_ONTOLOGY_PATH, _SHAPES_PATH, _EXAMPLE_PATH)
_SOURCE_TEXTS = {
    rel_path: (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    for rel_path in _SOURCE_PATHS
}
_TERM_PATTERN = re.compile(r"(?:lg|cat|sh|rdf|rdfs|xsd):[A-Za-z][A-Za-z0-9_]*")
_PROPERTY_BLOCKS = tuple(
    re.finditer(
        r"(?ms)^(lg:[A-Za-z][A-Za-z0-9_]*)\s+a\s+rdf:Property\s*;(.*?)^\.\s*$",
        _SOURCE_TEXTS[_ONTOLOGY_PATH],
    )
)
_LINKED_PROPERTY_TERMS: dict[str, set[str]] = {}
for match in _PROPERTY_BLOCKS:
    property_term = match.group(1)
    body = match.group(2)
    linked_classes = {
        *re.findall(r"rdfs:domain\s+(lg:[A-Za-z][A-Za-z0-9_]*)\s*;", body),
        *re.findall(r"rdfs:range\s+(lg:[A-Za-z][A-Za-z0-9_]*)\s*;", body),
    }
    for class_term in linked_classes:
        _LINKED_PROPERTY_TERMS.setdefault(class_term, set()).add(property_term)

_SOURCE_BLOCKS = {rel_path: [] for rel_path in _SOURCE_PATHS}
for rel_path, text in _SOURCE_TEXTS.items():
    _buffer: list[str] = []
    _inside_multiline = False
    for raw_line in text.splitlines():
        _buffer.append(raw_line)
        if raw_line.count('"""') % 2 == 1:
            _inside_multiline = not _inside_multiline
        if not _inside_multiline and raw_line.strip().endswith("."):
            _SOURCE_BLOCKS[rel_path].append("\n".join(_buffer))
            _buffer = []
    if _buffer:
        _SOURCE_BLOCKS[rel_path].append("\n".join(_buffer))

_EXPORTED_OBJECT_NAMES = (
    "AugmentedRule",
    "ClosedRuleCell",
    "RulePolarity",
    "WitnessDomain",
    "PredicateDomain",
    "SupportReflection",
    "SelectQuery",
    "TriplePattern",
    "JoinPattern",
    "AntiJoinPattern",
)
_OBJECT_TERM_SETS: dict[str, dict[str, tuple[str, ...]]] = {}
for object_name in _EXPORTED_OBJECT_NAMES:
    class_term = f"lg:{object_name}"
    if class_term not in set(_TERM_PATTERN.findall(_SOURCE_TEXTS[_ONTOLOGY_PATH])):
        raise RuntimeError(
            f"TTL kernel object image missing required class term for {object_name}: {class_term}"
        )
    _ontology_blocks = tuple(
        block for block in _SOURCE_BLOCKS[_ONTOLOGY_PATH] if class_term in block
    )
    _ontology_terms = tuple(
        sorted(
            {
                *(term for block in _ontology_blocks for term in _TERM_PATTERN.findall(block)),
                *_LINKED_PROPERTY_TERMS.get(class_term, set()),
                class_term,
            }
        )
    )
    _shape_blocks = tuple(
        block for block in _SOURCE_BLOCKS[_SHAPES_PATH] if class_term in block
    )
    _shape_terms = tuple(
        sorted(
            {
                term
                for block in _shape_blocks
                for term in _TERM_PATTERN.findall(block)
            }
        )
    )
    _typed_instance_pattern = re.compile(
        rf"(?m)^(lg:[A-Za-z][A-Za-z0-9_]*)\s+a\s+{re.escape(class_term)}\b"
    )
    _instance_subjects = tuple(
        sorted(set(_typed_instance_pattern.findall(_SOURCE_TEXTS[_EXAMPLE_PATH])))
    )
    _example_blocks = tuple(
        block
        for block in _SOURCE_BLOCKS[_EXAMPLE_PATH]
        if class_term in block or any(subject in block for subject in _instance_subjects)
    )
    _example_terms = tuple(
        sorted(
            {
                term
                for block in _example_blocks
                for term in _TERM_PATTERN.findall(block)
            }
        )
    )
    _supporting_terms = tuple(
        sorted({*_ontology_terms, *_shape_terms, *_example_terms})
    )
    if not _supporting_terms:
        raise RuntimeError(
            f"TTL kernel object image missing parsed support terms for {object_name}"
        )
    _OBJECT_TERM_SETS[object_name] = {
        "ontology": _ontology_terms,
        "shape": _shape_terms,
        "example": _example_terms,
        "supporting": _supporting_terms,
    }

with deadline_scope(Deadline.from_timeout_ms(30_000)):
    with deadline_clock_scope(MonotonicClock()):
        _REGISTRY = PrimeRegistry()
        _ADAPTER = PrimeIdentityAdapter(registry=_REGISTRY)
        _ZONE_ID = _ADAPTER.get_or_assign(
            namespace="ttl_kernel_vm.zone",
            token="ttl_kernel_vm",
        )
        _SOURCE_PATH_IDS = tuple(
            _ADAPTER.get_or_assign(namespace="ttl_kernel_vm.path", token=rel_path)
            for rel_path in _SOURCE_PATHS
        )
        _TERM_IDS = {
            term: _ADAPTER.get_or_assign(namespace="ttl_kernel_vm.term", token=term)
            for term in sorted(
                {
                    term
                    for parts in _OBJECT_TERM_SETS.values()
                    for term_set in parts.values()
                    for term in term_set
                }
            )
        }
        _OBJECT_IMAGES = {
            object_name: KernelVmObjectImage(
                object_id=_ADAPTER.get_or_assign(
                    namespace="ttl_kernel_vm.object_image",
                    token=object_name,
                ),
                zone_id=_ZONE_ID,
                source_path_ids=tuple(
                    path_id
                    for rel_path, path_id in zip(_SOURCE_PATHS, _SOURCE_PATH_IDS, strict=True)
                    if _OBJECT_TERM_SETS[object_name][
                        "ontology" if rel_path == _ONTOLOGY_PATH else "shape" if rel_path == _SHAPES_PATH else "example"
                    ]
                ),
                class_term_id=_TERM_IDS[f"lg:{object_name}"],
                supporting_term_ids=tuple(
                    _TERM_IDS[term] for term in _OBJECT_TERM_SETS[object_name]["supporting"]
                ),
                ontology_term_ids=tuple(
                    _TERM_IDS[term] for term in _OBJECT_TERM_SETS[object_name]["ontology"]
                ),
                shape_term_ids=tuple(
                    _TERM_IDS[term] for term in _OBJECT_TERM_SETS[object_name]["shape"]
                ),
                example_term_ids=tuple(
                    _TERM_IDS[term] for term in _OBJECT_TERM_SETS[object_name]["example"]
                ),
                label=object_name,
            )
            for object_name in _EXPORTED_OBJECT_NAMES
        }

AugmentedRule = _OBJECT_IMAGES["AugmentedRule"]
ClosedRuleCell = _OBJECT_IMAGES["ClosedRuleCell"]
RulePolarity = _OBJECT_IMAGES["RulePolarity"]
WitnessDomain = _OBJECT_IMAGES["WitnessDomain"]
PredicateDomain = _OBJECT_IMAGES["PredicateDomain"]
SupportReflection = _OBJECT_IMAGES["SupportReflection"]
SelectQuery = _OBJECT_IMAGES["SelectQuery"]
TriplePattern = _OBJECT_IMAGES["TriplePattern"]
JoinPattern = _OBJECT_IMAGES["JoinPattern"]
AntiJoinPattern = _OBJECT_IMAGES["AntiJoinPattern"]

__all__ = [
    "AntiJoinPattern",
    "AugmentedRule",
    "ClosedRuleCell",
    "JoinPattern",
    "KernelVmObjectImage",
    "PredicateDomain",
    "RulePolarity",
    "SelectQuery",
    "SupportReflection",
    "TriplePattern",
    "WitnessDomain",
]
