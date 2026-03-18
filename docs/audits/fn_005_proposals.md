FN-005 review: confirmed
I reviewed src/gabion/tooling/policy_substrate/invariant_graph.py and the concern is real.

File size: ~538,766 bytes.

declared_workstream_registries() is at src/gabion/tooling/policy_substrate/invariant_graph.py:9158.

The actual assembly logic lives in _iter_declared_workstream_registries() at :9131.

The connectivity/registry injection point is currently the final line of that helper:
registries.extend(connectivity_synergy_workstream_registries()) at :9154.

That means the “where do I add/find the registry integration point?” answer is technically simple, but operationally expensive to discover because it is buried in a giant mixed-responsibility module with no top-level navigation surface.

Higher-order synthesis
Mind A convergence
AA: The immediate fix should reduce orientation cost without changing semantics. A module-level docstring and section index are the fastest high-leverage improvement.

AB critique: A docstring alone is not enough because it still leaves the registry integration point embedded in a 500KB file; the fix should expose the integration seam structurally, not just descriptively.

AA response: Agreed. Documentation should be paired with code motion that creates a smaller, named registry-catalog surface.

Mind B wedge product
The strongest synthesis is:

Add explicit navigation metadata now so humans/LLMs can orient cheaply.

Extract registry declaration/assembly into a dedicated small module so the WRD/registry integration point becomes the file, not a buried line number.

Add a machine-checked navigation contract so the orientation aid does not rot.

That combination addresses both immediate friction and long-term maintainability.

Issue 1: invariant_graph.py has no module-level navigation contract, so public entry points and major subsystems are undiscoverable without symbol-exact search
Today the top of the file is only imports and constants. There is no docstring that answers basic orientation questions like:

What are the major public APIs?

Where are graph model types defined?

Where are registry assembly points?

Where do artifact joiners live?

Where is the workstream projection pipeline?

Where are load/write helpers?

This is exactly why declared_workstream_registries() is costly to find unless the reader already knows the symbol name.

Suggested task
Add a module-level navigation docstring to invariant_graph.py

Start task
Issue 2: The registry integration point is structurally hidden because registry assembly is implemented as an inline imperative list-builder inside invariant_graph.py
_iter_declared_workstream_registries() is the actual integration seam for declared registries, including the connectivity-synergy bundle. But because it is:

private,

near line 9131,

in the middle of unrelated graph-building code,

it does not read like a first-class extension point. This is the core structural cause of the WRD/integration-point discoverability problem.

The better shape is to make registry declaration a small, dedicated surface with a filename that advertises its purpose.

Suggested task
Extract declared workstream registry assembly into a dedicated registry catalog module

Start task
Issue 3: The registry catalog is opaque because there is no typed manifest explaining why each provider is present, whether it is singleton vs family, and where to edit it
The current helper is just a sequence of local variables and if x is not None: append(x). That preserves behavior, but it does not encode orientation metadata.

A small typed manifest would make the integration seam self-describing for both humans and LLMs, and it aligns with the repo’s preference for explicit structural carriers over ad hoc branches.

Suggested task
Represent declared registry providers as a typed catalog instead of ad hoc local variables

Start task
Issue 4: Orientation aids will drift unless there is a machine-checked contract tying docs/navigation to the declared registry seam
Without a guard, a docstring or extracted module can become stale the next time someone adds a new registry provider. The repo already relies heavily on declarative and testable contracts; navigation should be treated similarly.

Suggested task
Add a lightweight test that locks the declared-registry navigation surface

Start task
Issue 5: Even after extraction, the giant file still needs visible section boundaries for cheap intra-file scanning
A docstring and extraction solve the registry seam, but invariant_graph.py will still be a very large file with many internal families. Readers need cheap visual anchors even when they stay within the module.

This is especially relevant for LLM agents, because section sentinels reduce blind-range scanning.

Suggested task
Add stable section sentinels inside invariant_graph.py for major implementation zones

Start task
Recommended implementation order
Extract the declared-registry seam into its own module.

Add the typed catalog in that module.

Add the module-level docstring and section sentinels in invariant_graph.py.

Add/adjust tests that lock the new navigation surface.

That order gives the biggest discoverability win earliest.

Net assessment
FN-005 is valid, and the best fix is not just “add a docstring.”
The comprehensive solution is:

documentation fix: top-level navigation docstring

structural fix: dedicated declared-registry catalog module

semantic-shape fix: typed provider manifest

durability fix: test-guarded navigation contract

scanability fix: stable section sentinels

That combination directly addresses the current pain around declared_workstream_registries() and makes future WRD/workstream registry integration points obvious.

Checks performed
✅ rg -n "^# |^## |^### |policy_seed|change_protocol|runtime|ambiguity|LSP|Decision Protocol|compatibility" POLICY_SEED.md | head -n 120

✅ rg -n "^# |^## |^### |contract|runtime|distinction|ladder|semantic|Decision Protocol" glossary.md | head -n 120

✅ rg --files | rg 'invariant_graph\.py$|workstream|registry'

✅ cat in/AGENTS.md

✅ wc -c src/gabion/tooling/policy_substrate/invariant_graph.py

✅ sed -n '1,140p' src/gabion/tooling/policy_substrate/invariant_graph.py

✅ rg -n "^(def|class) |declared_workstream_registries|WRD|workstream_registry|registry" src/gabion/tooling/policy_substrate/invariant_graph.py | head -n 260

✅ sed -n "${START},${END}p" src/gabion/tooling/policy_substrate/invariant_graph.py for the declared_workstream_registries region

✅ rg -n "\bWRD\b|WRD-|workstream registry definition|declared_workstream_registries|connectivity_synergy_workstream_registries|workstream_registries\(" src tests docs | head -n 240

⚠️ make test not run, per read-only QA scope and instruction not to execute code/tests for this review.

FN-005 review: invariant_graph.py is an orientation hotspot, and the current file structure confirms the report
I reviewed src/gabion/tooling/policy_substrate/invariant_graph.py directly.

What the file currently looks like
Size: 538,766 bytes (~551 KB)

Top-level surface:

159 top-level functions

48 top-level classes

The file begins with imports and constants, but no module-level docstring

declared_workstream_registries() is indeed buried near line 9158

The actual registry assembly point is very small and currently split across:

the import block at the top

_iter_declared_workstream_registries() near line 9131

declared_workstream_registries() near line 9158

So the user’s friction report is valid: the integration point exists, but it is not discoverable by structure.

Mind A
AA perspective
The problem is not just “missing docs”; it is that a small public entrypoint is embedded in a monolith whose physical layout does not match its conceptual layout.

declared_workstream_registries() is only a thin assembler:

phase5_workstream_registry()

prf_workstream_registry()

surface_contract_convergence_workstream_registry()

runtime_context_injection_workstream_registry()

boundary_ingress_convergence_workstream_registry()

unit_test_readiness_workstream_registry()

structural_anti_pattern_convergence_workstream_registry()

connectivity_synergy_workstream_registries()

That means the orientation cost is mostly self-inflicted by placement, not by logic complexity.

AA critique of a docstring-only fix
A module docstring would help, but by itself it would still leave:

the public registry integration point in the wrong file neighborhood,

the import list as the only real “table of contents,” and

the navigation burden for both humans and LLMs mostly unchanged once they need the next hop.

So I would treat the hypothesis as correct but incomplete.

AB perspective
There are really two distinct discoverability jobs that the file is failing to do:

Module orientation
“What major families live here?”

Registry entrypoint orientation
“Where do I add or inspect a workstream registry?”

The second is the sharper pain. A contributor trying to integrate a new registry does not need the whole invariant graph surface; they need a single canonical catalog location.

AB critique of a split-everything refactor
A full decomposition of a 551 KB file into many modules may be directionally right, but it is expensive and risky if done all at once. For FN-005 specifically, that would overshoot the immediate usability problem.

So the right answer is likely staged:

add strong orientation affordances immediately,

then extract the registry catalog into a dedicated module,

then continue file decomposition by concept family.

Higher-order synthesis from Mind A
The best solution is a three-layer structural fix, not one fix.

Issue 1 — The module has no navigation surface, so readers must infer structure from raw symbol search
A 551 KB module with 159 functions and 48 classes needs a top-level orientation artifact. Right now the only way to understand its conceptual zones is to skim thousands of lines or know the exact symbol ahead of time.

Suggested task
Add a module-level orientation docstring and section index to invariant_graph.py

Start task
Issue 2 — The workstream registry integration point is physically hidden inside the graph-construction monolith
The real integration point for “where do registries come from?” is _iter_declared_workstream_registries(), but it lives in the middle of unrelated graph-building helpers. That means the registry catalog is not a first-class concept in the file layout.

This is the highest-value structural change for FN-005.

Suggested task
Extract declared workstream registry assembly into a dedicated registry catalog module

Start task
Issue 3 — The registry catalog itself is opaque because the assembly list carries no inline rationale or extension protocol
Even after a reader finds _iter_declared_workstream_registries(), they still have to infer:

which registries are singleton vs multi-registry families,

whether ordering matters,

how to add a new registry correctly.

That is an orientation failure at the API boundary.

Suggested task
Make the workstream registry catalog self-describing and extension-friendly

Start task
Issue 4 — invariant_graph.py is carrying too many conceptual zones for efficient human or LLM navigation
The file currently combines at least all of these:

graph data model

scoring/cut-tradeoff logic

workstream projections

ledger projections

phase-5 touchsite scanning

registry enrichment

artifact joins

planning-chart augmentation

load/write serialization

That is a structural smell independent of FN-005. It amplifies every navigation task.

Suggested task
Plan a staged decomposition of invariant_graph.py by concept family

Start task
Mind B
BA: wedge product emerging from A
AA correctly identifies the layout mismatch.
AB correctly insists the fix be staged.

The synthesis that emerges is:

Docstring solves “I opened the file and need bearings.”

Registry catalog extraction solves “I need the WRD/workstream integration point now.”

Concept-family decomposition solves “this file will keep hurting us again.”

That wedge matters because FN-005 is not just a documentation bug or just a file-size bug. It is a navigation topology bug:

local discoverability is poor,

the public extension seam is not first-class,

the physical module boundary does not reflect the conceptual boundary.

BB: critique of the synthesis
The only caveat is prioritization:

If only one change ships soon, it should be extract the registry catalog.

If two ship, add the module docstring/index.

The broader file split should be explicitly framed as a follow-on structural remediation, not required to validate the user’s immediate hypothesis.

Recommended resolution order
Extract declared_workstream_registries() into a dedicated catalog module

Highest leverage for the reported use case

Makes the WRD/workstream integration seam obvious

Add a module-level docstring and visible section index in invariant_graph.py

Fast win for both humans and LLM agents

Reduces repeated token-cost orientation

Stage a broader concept-family split of invariant_graph.py

Longer-term fix

Prevents future orientation regressions

Bottom line
FN-005 is a valid issue.

The proposed module-level docstring is worthwhile, but it is not sufficient by itself. The concrete structural fix should make the registry assembly seam a first-class module, then add a top-level navigation surface, then progressively decompose the monolith by concept family. That combination addresses the immediate “where is the registry integration point?” problem and the deeper orientation barrier that caused it.

FN-005 review: invariant_graph.py is an orientation barrier
I reviewed src/gabion/tooling/policy_substrate/invariant_graph.py and the adjacent registry modules. The reported friction is real.

The immediate integration point for a new workstream registry is currently split across:

top-level imports in src/gabion/tooling/policy_substrate/invariant_graph.py

_iter_declared_workstream_registries() at src/gabion/tooling/policy_substrate/invariant_graph.py:9131

the public wrapper declared_workstream_registries() at :9158

That means the “where do I wire in WRD?” answer is technically present, but buried inside a 538,766-byte monolith with no module-level map.

Higher-order synthesis
Mind A / AA
The narrow diagnosis is correct: a module docstring would help. A top-of-file public API map would lower first-contact cost for both humans and LLMs.

Mind A / AB critique
A docstring alone treats the symptom, not the cause. The registry assembly seam is not just undocumented; it is misplaced inside a file that also defines graph DTOs, ledger projections, planning bundles, scanners, enrichment, loaders, writers, and comparison functions.

Mind A convergence
So the right answer is layered:

add orientation aids now,

extract the registry catalog seam next,

make registry integration discoverable and testable as a first-class contract.

Mind B wedge-product synthesis
The emergent solution is not “better comments” or “big refactor” in isolation. It is a structural split between:

navigation surface: a compact, explicit catalog of public families and integration seams

semantic surface: the actual graph-building logic

That split reduces context cost while preserving existing behavior.

Issue 1: invariant_graph.py has no module-level navigation surface despite exposing many public families
The top of src/gabion/tooling/policy_substrate/invariant_graph.py starts directly with imports and then hundreds of lines of dataclasses and helpers. The only compact list of public entrypoints is the __all__ block near the very end, which is too late to help orientation. Given the file contains at least these distinct families—

graph DTOs

workstream projections

ledger projections / deltas / alignments

touchsite scanning

registry enrichment

graph building

serialization / tracing

—the absence of a top-level map is the first barrier.

Suggested task
Add a module-level navigation docstring to invariant_graph.py

Start task
Issue 2: the WRD/workstream-registry integration seam is embedded in the wrong module
declared_workstream_registries() is a registry-catalog concern, but it currently lives inside src/gabion/tooling/policy_substrate/invariant_graph.py, a large semantic/serialization module. That makes contributors infer that registry declaration is part of graph assembly internals, when it is really a boundary-level catalog.

This is the highest-value structural fix. If someone asks “where do I add a new registry?”, the answer should be a dedicated small module, not “search inside the 9k-line graph file.”

Suggested task
Extract declared workstream registry assembly into a dedicated catalog module

Start task
Issue 3: registry assembly is encoded as repetitive append logic instead of an explicit catalog contract
The current implementation is a sequence of local variables plus if ... is not None: append(...), followed by registries.extend(connectivity_synergy_workstream_registries()). It works, but it hides structure:

which providers are singleton providers vs multi-registry providers

what ordering is contractual

where a new registry should be inserted

what metadata explains the registry family

That makes “add WRD here” a source-reading exercise instead of a declarative edit.

Suggested task
Replace hand-written append logic with an explicit registry provider catalog

Start task
Issue 4: there is no first-class discoverability contract for adding a new registry
Today, discoverability is implicit. A contributor can infer the workflow from tests in tests/gabion/tooling/runtime_policy/test_workstream_registry_definitions.py, but there is no small, named contract saying:

create a *_registry.py module,

expose *_workstream_registry() or *_workstream_registries(),

add it to the catalog,

add/update coverage in the registry definition tests.

For a repo with strong policy semantics, this is a missing boundary contract.

Suggested task
Document and test the canonical 'add a workstream registry' workflow

Start task
Issue 5: the file boundary itself is too coarse for both humans and LLMs
Even after adding a docstring, invariant_graph.py would still be doing too many jobs. The file currently mixes:

data contracts (InvariantGraph, InvariantGraphNode, etc.)

workstream projection logic

ledger projection logic

touchsite scanner internals

registry enrichment

graph build orchestration

serialization helpers

That is why the line-9200 lookup cost exists in the first place. The long-term fix is to decompose by responsibility, not just annotate the monolith.

Suggested task
Split invariant_graph.py into responsibility-oriented modules with a stable facade

Start task
Recommended rollout order
Immediate: module docstring index

Next: dedicated workstream_registry_catalog.py

Then: declarative provider catalog

Then: contributor/test contract for adding registries

Long-term: split the monolith behind a stable facade

Bottom line
FN-005 is valid, and the proposed module-level docstring is a good first patch, but it is not sufficient by itself.

The concrete structural answer is:

Yes to a module-level summary.

More importantly, move declared_workstream_registries() out of invariant_graph.py into a small dedicated catalog module that becomes the canonical WRD/workstream-registry integration point.

Back that with an explicit provider manifest and tests so discoverability becomes contractual rather than folkloric.

Checks performed
⚠️ sed -n '1,180p' POLICY_SEED.md && sed -n '1,220p' glossary.md — static document inspection only; no execution or mutation.

✅ rg --files | rg 'invariant_graph\\.py$|workstream.*registry|registry.*workstream' — identified the relevant invariant graph and registry modules.

✅ wc -c src/gabion/tooling/policy_substrate/invariant_graph.py src/gabion/tooling/runtime/invariant_graph.py && rg -n "declared_workstream_registries|WRD" src/gabion/tooling/policy_substrate/invariant_graph.py src/gabion/tooling/runtime/invariant_graph.py — confirmed the 538,766-byte target file and located declared_workstream_registries().

✅ sed -n '1,180p' src/gabion/tooling/policy_substrate/invariant_graph.py && sed -n '9090,9195p' src/gabion/tooling/policy_substrate/invariant_graph.py — inspected the file top and the registry assembly seam.

✅ sed -n '12720,12810p' src/gabion/tooling/policy_substrate/invariant_graph.py && sed -n '1,240p' src/gabion/tooling/policy_substrate/workstream_registry.py — inspected the public export surface and the registry contract types.

✅ rg -n "\\bWRD\\b|workstream_registry|declared_workstream_registries" src/gabion/tooling/policy_substrate — mapped the registry-defining modules and integration references.

✅ sed -n '740,790p' tests/gabion/tooling/runtime_policy/test_workstream_registry_definitions.py && sed -n '1,80p' tests/gabion/tooling/runtime_policy/test_workstream_registry_definitions.py — confirmed existing tests already treat declared registry aggregation as a testable contract.

Notes
No files were modified, no tests were run, and no commit or PR was created because this was a read-only QA review.

FN-005 review: confirmed
src/gabion/tooling/policy_substrate/invariant_graph.py is currently an orientation bottleneck, not just a large module. Static inspection shows:

the file is about 538,766 bytes

declared_workstream_registries() is near line 9158

the registry assembly logic is embedded inside the same monolith that also defines:

graph model dataclasses

graph build state and node/edge helpers

artifact joiners

workstream/ledger projections

serialization helpers

the top of the file has no module docstring or section map

the public export list exists only at the very end in __all__, so it does not help with first-pass navigation

the WRD/workstream-registry integration point is discoverable only if the reader already knows to search for:

_iter_declared_workstream_registries

declared_workstream_registries

or the specific imported *_workstream_registry factory names

Below is the converged review, with the “two minds” synthesis folded into each recommendation.

1. The immediate orientation failure is real: the module lacks a top-level navigation surface
AA: The user’s hypothesis is correct. A module-level docstring is the fastest, lowest-risk fix because it reduces search cost without changing semantics.

AB critique: A docstring alone helps first contact, but it does not solve the deeper issue that the file’s conceptual zones are not reflected in its structure. If the only change is prose, declared_workstream_registries() will still live ~9k lines deep in a file that mixes multiple responsibilities.

A convergence: Add the docstring, but make it an actual navigation index rather than a generic summary:

list the major function families

identify the registry assembly entrypoint explicitly

identify the graph build pipeline entrypoints

identify where artifact joiners live

identify public projection/serialization functions

include “if you are looking for WRD/workstream registry integration, start here” language

B wedge observation: The emergent synthesis is: documentation should mirror runtime phases. The top-of-file summary should be organized by operational pipeline, not alphabetically.

Suggested task
Add a real module-level navigation index to invariant_graph.py

Start task
2. The real structural problem is that registry declaration is trapped inside the graph-construction monolith
AA: declared_workstream_registries() is a small catalog function, but it is buried between _enrich_workstream_registry() and later graph-index/test-join code. That is a structural mismatch: registry declaration is a boundary/catalog concern, not graph-core logic.

AB critique: Merely moving the function inside the same file will not be enough. The import block at the top already reveals the true problem: the module is acting as both registry catalog and graph engine. New contributors looking for “where do I add a new registry?” should not have to open the graph engine at all.

A convergence: Extract the registry aggregation into a dedicated module, keeping invariant_graph.py as a consumer. This creates a stable discovery surface for both humans and LLMs.

B wedge observation: The key synthesis is a boundary normalization move consistent with repo policy: treat registry assembly as a reified input bundle to graph construction, not as incidental inline setup.

Suggested task
Extract workstream registry cataloging into a dedicated module

Start task
3. New registry integration points are hardcoded as an implicit import list instead of an explicit registry manifest
AA: Even after extraction, contributors still need to infer the allowed registry set by reading individual imports and append statements. That is brittle, especially for “where does WRD go?” questions.

AB critique: A manifest must not become a dynamic plugin mechanism unless policy explicitly wants that. This repo prefers deterministic, impossible-by-construction contracts over looser discovery patterns.

A convergence: Keep the registry set explicit and static, but reify it as a small manifest/descriptor table in one place. That gives:

one canonical list

clear insertion point for new registries

easier documentation generation

lower token cost for LLM readers

B wedge observation: The higher-order synthesis here is: replace repeated procedural append logic with a declarative catalog while preserving deterministic behavior.

Suggested task
Replace ad hoc registry append logic with a declarative registry manifest

Start task
4. The module mixes at least five conceptual zones without explicit section boundaries or package decomposition
AA: The file is not just “big”; it conflates:

graph schema/model types

graph-building primitives

registry enrichment

artifact joins

projections and serialization

This is why a reader chasing one symbol has to cross unrelated semantic terrain.

AB critique: A full split into many files could become churn-heavy if done in one pass. But the absence of section boundaries means even a staged refactor has no landmarks.

A convergence: Use a staged structural solution:

Stage 1: add section headers and a top-level map

Stage 2: split by responsibility into adjacent modules

Stage 3: keep invariant_graph.py as a façade if external imports depend on it

A practical decomposition suggested by the current symbol layout:

invariant_graph_types.py

invariant_graph_registry_ingress.py

invariant_graph_artifact_joins.py

invariant_graph_build.py

invariant_graph_projections.py

invariant_graph_io.py

B wedge observation: The strong emergent pattern is that the file already contains latent modules; the refactor is mostly a matter of making the hidden topology explicit.

Suggested task
Stage a responsibility-based decomposition of invariant_graph.py

Start task
5. There is no dedicated contributor-facing document for “how workstream registries are assembled”
AA: Even after code cleanup, contributors will still ask “what is the registry integration protocol?” because the conceptual rule is larger than one function.

AB critique: This should not become a long narrative doc if a short focused reference will do. The best doc is a minimal map tied to concrete file paths and identifiers.

A convergence: Add a short docs page specifically for the invariant graph/workstream-registry assembly path. This is especially useful for LLM agents because it externalizes the navigation index into a small file instead of forcing a 500KB module read.

B wedge observation: This is the documentation counterpart to the code split: a small stable orientation artifact that amortizes future context-window cost.

Suggested task
Add a focused developer note for invariant graph registry integration

Start task
Recommended implementation order
Add the module-level navigation docstring

Extract declared_workstream_registries() to a dedicated catalog module

Reify the registry list as a static manifest

Add the short docs page

Then do the larger module decomposition in stages

That order gives immediate relief for FN-005 while preserving a clean path to the more durable fix.

Bottom line
Your hypothesis is good, but incomplete:

Yes: a module-level docstring would materially reduce orientation cost.

But: the more important fix is to move registry assembly out of the 551KB graph monolith and make it a first-class catalog surface.

The best durable answer is a combination of:

top-level navigation

dedicated registry catalog module

declarative manifest

staged file decomposition

small contributor-facing registry map doc

Checks performed
✅ pwd && rg --files -g 'AGENTS.md' -g 'POLICY_SEED.md' -g 'glossary.md' -g 'README.md' -g 'CONTRIBUTING.md' -g 'src/**/invariant_graph.py'

✅ sed -n '1,220p' AGENTS.md

✅ sed -n '1,220p' POLICY_SEED.md && printf '\n---GLOSSARY---\n' && sed -n '1,220p' glossary.md

✅ rg -n "declared_workstream_registries|WRD|workstream_registry|registry" src/gabion/tooling/policy_substrate/invariant_graph.py src/gabion/tooling/runtime/invariant_graph.py

✅ wc -c src/gabion/tooling/policy_substrate/invariant_graph.py src/gabion/tooling/runtime/invariant_graph.py && printf '\n--- policy_substrate head ---\n' && sed -n '1,140p' src/gabion/tooling/policy_substrate/invariant_graph.py && printf '\n--- runtime head ---\n' && sed -n '1,140p' src/gabion/tooling/runtime/invariant_graph.py

✅ sed -n '9100,9188p' src/gabion/tooling/policy_substrate/invariant_graph.py && printf '\n--- tail public exports ---\n' && sed -n '12720,12820p' src/gabion/tooling/policy_substrate/invariant_graph.py

✅ rg -n '^def ' src/gabion/tooling/policy_substrate/invariant_graph.py | sed -n '1,140p'; printf '\n--- later defs ---\n'; rg -n '^def ' src/gabion/tooling/policy_substrate/invariant_graph.py | sed -n '140,260p'

✅ sed -n '1,220p' src/gabion/tooling/policy_substrate/workstream_registry.py && printf '\n--- registry modules ---\n' && rg --files src/gabion/tooling/policy_substrate | rg '_registry\\.py$'

✅ sed -n '11740,11820p' src/gabion/tooling/policy_substrate/invariant_graph.py && printf '\n--- build graph/workstreams ---\n' && sed -n '11940,12540p' src/gabion/tooling/policy_substrate/invariant_graph.py