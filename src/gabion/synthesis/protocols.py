# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Set

from gabion.synthesis.model import (
    FieldSpec,
    NamingContext,
    ProtocolSpec,
    SynthesisConfig,
    SynthesisPlan,
)
from gabion.synthesis.naming import suggest_name
from gabion.analysis.timeout_context import check_deadline
from gabion.order_contract import sort_once

_EMPTY_FIELD_TYPES: Mapping[str, str] = {}
_DEFAULT_NAMING_CONTEXT = NamingContext()


@dataclass
class Synthesizer:
    config: SynthesisConfig = field(default_factory=SynthesisConfig)

    def plan(
        self,
        bundle_tiers: Mapping[frozenset[str], int],
        field_types: Mapping[str, str] = _EMPTY_FIELD_TYPES,
        naming_context: NamingContext = _DEFAULT_NAMING_CONTEXT,
    ) -> SynthesisPlan:
        check_deadline()
        field_types = field_types or {}
        protocols: List[ProtocolSpec] = []
        warnings: List[str] = []

        existing_names = set(naming_context.existing_names)
        context = NamingContext(
            existing_names=existing_names,
            frequency=naming_context.frequency,
            fallback_prefix=naming_context.fallback_prefix,
        )

        for bundle in sort_once(
            bundle_tiers,
            source="Synthesizer.plan.bundle_tiers",
            key=lambda value: (len(value), tuple(sort_once(value, source = 'src/gabion/synthesis/protocols.py:44'))),
        ):
            check_deadline()
            tier = bundle_tiers[bundle]
            bundle_set = set(bundle)
            if not self._bundle_allowed(bundle_set, tier):
                continue
            name = suggest_name(bundle_set, context)
            context.existing_names.add(name)
            fields = self._build_fields(bundle_set, field_types)
            protocols.append(
                ProtocolSpec(
                    name=name,
                    fields=fields,
                    bundle=bundle_set,
                    tier=tier,
                    rationale=f"Tier-{tier} bundle",
                )
            )

        if not protocols:
            warnings.append("No bundles qualified for synthesis.")

        return SynthesisPlan(protocols=protocols, warnings=warnings, errors=[])

    def _bundle_allowed(self, bundle: Set[str], tier: int) -> bool:
        if not bundle:
            return False
        if len(bundle) < self.config.min_bundle_size and not self.config.allow_singletons:
            return False
        return tier <= self.config.max_tier

    def _build_fields(
        self, bundle: Iterable[str], field_types: Dict[str, str]
    ) -> List[FieldSpec]:
        check_deadline()
        fields: List[FieldSpec] = []
        for name in sort_once(
            bundle,
            source="Synthesizer._build_fields.bundle",
        ):
            check_deadline()
            type_hint = field_types.get(name)
            fields.append(FieldSpec(name=name, type_hint=type_hint, source_params={name}))
        return fields
