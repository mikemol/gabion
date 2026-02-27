from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gabion.analysis.timeout_context import Deadline, deadline_clock_scope, deadline_scope
from gabion.deadline_clock import GasMeter
from gabion.analysis.type_fingerprints import (
    EVIDENCE_KIND_NAMESPACE,
    SYNTH_NAMESPACE,
    TYPE_CTOR_NAMESPACE,
    Fingerprint,
    FingerprintDimension,
    PrimeRegistry,
    TypeConstructorRegistry,
    apply_synth_dimension,
    build_synth_registry,
    fingerprint_carrier_soundness,
    synth_registry_payload,
)


@dataclass(frozen=True)
class LabAdapterCase:
    case_id: str
    description: str
    base_atoms: tuple[str, ...]
    ctor_atoms: tuple[str, ...]
    provenance_atoms: tuple[str, ...] = ()
    synth_atoms: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabAdapterSpec:
    schema_version: str
    cases: tuple[LabAdapterCase, ...]


def _load_adapter_spec(path: Path) -> LabAdapterSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[LabAdapterCase] = []
    for raw_case in payload.get("cases", []):
        cases.append(
            LabAdapterCase(
                case_id=str(raw_case["case_id"]),
                description=str(raw_case.get("description", "")),
                base_atoms=tuple(str(atom) for atom in raw_case.get("base_atoms", [])),
                ctor_atoms=tuple(str(atom) for atom in raw_case.get("ctor_atoms", [])),
                provenance_atoms=tuple(
                    str(atom) for atom in raw_case.get("provenance_atoms", [])
                ),
                synth_atoms=tuple(str(atom) for atom in raw_case.get("synth_atoms", [])),
            )
        )
    return LabAdapterSpec(
        schema_version=str(payload.get("schema_version", "gabion-lab-adapter@1")),
        cases=tuple(cases),
    )


def _namespace_atom(namespace: str, atom: str) -> str:
    if namespace == TYPE_CTOR_NAMESPACE:
        return f"ctor:{atom}"
    if namespace == EVIDENCE_KIND_NAMESPACE:
        return f"evidence:{atom}"
    if namespace == SYNTH_NAMESPACE:
        return f"synth:{atom}"
    return atom


def _dimension_from_atoms(
    atoms: tuple[str, ...],
    registry: PrimeRegistry,
    *,
    namespace: str,
) -> FingerprintDimension:
    product = 1
    mask = 0
    exponents: dict[str, int] = {}
    for atom in atoms:
        key = _namespace_atom(namespace, atom)
        prime = registry.get_or_assign(key)
        product *= prime
        bit = registry.bit_for(key)
        if bit is not None:
            mask |= 1 << bit
        exponents[key] = exponents.get(key, 0) + 1
    return FingerprintDimension(
        product=product,
        mask=mask,
        exponents=tuple(sorted(exponents.items())),
    )


def _decode_dimension(dim: FingerprintDimension, registry: PrimeRegistry) -> dict[str, Any]:
    keys, remainder = dim.keys_with_remainder(registry)
    return {
        "product": dim.product,
        "mask": dim.mask,
        "keys_with_remainder": {
            "keys": keys,
            "remainder": remainder,
        },
    }


def _run_bridge() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    adapter_path = repo_root / "in/universal-curve-lab-bundle/python/adapter_schema.json"
    artifacts_dir = repo_root / "in/universal-curve-lab-bundle/artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_adapter_spec(adapter_path)

    registry = PrimeRegistry()
    ctor_registry = TypeConstructorRegistry(registry)

    unsynthesized_cases: list[tuple[LabAdapterCase, Fingerprint]] = []
    for case in spec.cases:
        for ctor in case.ctor_atoms:
            ctor_registry.get_or_assign(ctor)
        base_dim = _dimension_from_atoms(case.base_atoms, registry, namespace="type_base")
        ctor_dim = _dimension_from_atoms(
            case.ctor_atoms,
            registry,
            namespace=TYPE_CTOR_NAMESPACE,
        )
        provenance_dim = _dimension_from_atoms(
            case.provenance_atoms,
            registry,
            namespace=EVIDENCE_KIND_NAMESPACE,
        )
        provided_synth_dim = _dimension_from_atoms(
            case.synth_atoms,
            registry,
            namespace=SYNTH_NAMESPACE,
        )
        unsynthesized_cases.append(
            (
                case,
                Fingerprint(
                    base=base_dim,
                    ctor=ctor_dim,
                    provenance=provenance_dim,
                    synth=provided_synth_dim,
                ),
            )
        )

    synth_registry = build_synth_registry(
        (fingerprint for _, fingerprint in unsynthesized_cases),
        registry,
        min_occurrences=2,
        version="lab-bridge@synth@1",
    )

    case_outputs: list[dict[str, Any]] = []
    with_synth: list[Fingerprint] = []
    for case, raw_fingerprint in unsynthesized_cases:
        synthesized = apply_synth_dimension(raw_fingerprint, synth_registry)
        if (
            not raw_fingerprint.synth.is_empty()
            and not synthesized.synth.is_empty()
            and synthesized.synth != raw_fingerprint.synth
        ):
            merged_exponents: dict[str, int] = {}
            for key, exponent in raw_fingerprint.synth.exponents:
                merged_exponents[key] = merged_exponents.get(key, 0) + exponent
            for key, exponent in synthesized.synth.exponents:
                merged_exponents[key] = merged_exponents.get(key, 0) + exponent
            merged_synth = FingerprintDimension(
                product=raw_fingerprint.synth.product * synthesized.synth.product,
                mask=raw_fingerprint.synth.mask | synthesized.synth.mask,
                exponents=tuple(sorted(merged_exponents.items())),
            )
            synthesized = Fingerprint(
                base=synthesized.base,
                ctor=synthesized.ctor,
                provenance=synthesized.provenance,
                synth=merged_synth,
            )
        with_synth.append(synthesized)
        case_outputs.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "dimensions": {
                    "base": _decode_dimension(synthesized.base, registry),
                    "ctor": _decode_dimension(synthesized.ctor, registry),
                    "provenance": _decode_dimension(synthesized.provenance, registry),
                    "synth": _decode_dimension(synthesized.synth, registry),
                },
                "carrier_soundness": {
                    "base_vs_ctor": fingerprint_carrier_soundness(
                        synthesized.base,
                        synthesized.ctor,
                    ),
                    "base_vs_provenance": fingerprint_carrier_soundness(
                        synthesized.base,
                        synthesized.provenance,
                    ),
                    "base_vs_synth": fingerprint_carrier_soundness(
                        synthesized.base,
                        synthesized.synth,
                    ),
                    "ctor_vs_provenance": fingerprint_carrier_soundness(
                        synthesized.ctor,
                        synthesized.provenance,
                    ),
                    "ctor_vs_synth": fingerprint_carrier_soundness(
                        synthesized.ctor,
                        synthesized.synth,
                    ),
                    "provenance_vs_synth": fingerprint_carrier_soundness(
                        synthesized.provenance,
                        synthesized.synth,
                    ),
                },
            }
        )

    bridge_payload = {
        "schema_version": spec.schema_version,
        "fingerprint_bridge_version": "gabion-lab-harness@1",
        "case_count": len(case_outputs),
        "cases": case_outputs,
        "synth_registry_snapshot": synth_registry_payload(
            synth_registry,
            registry,
            min_occurrences=2,
        ),
        "registry_seed": registry.seed_payload(),
    }

    output_path = artifacts_dir / "gabion_fingerprint_bridge.json"
    output_path.write_text(json.dumps(bridge_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    with deadline_scope(Deadline.from_timeout_ms(120_000)):
        with deadline_clock_scope(GasMeter(limit=500_000)):
            _run_bridge()


if __name__ == "__main__":
    main()
