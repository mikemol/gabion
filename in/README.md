---
doc_revision: 2
reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.
doc_change_protocol: POLICY_SEED.md#change_protocol
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
  - CONTRIBUTING.md#contributing_contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 42
  glossary.md#contract: 43
  CONTRIBUTING.md#contributing_contract: 84
doc_review_notes:
  POLICY_SEED.md#policy_seed: Re-reviewed policy seed execution controls (self-hosted constraints, pinned actions, and review discipline) and confirmed this readme still points to the canonical security contract.
  glossary.md#contract: Re-reviewed glossary contract semantics and confirmed this readme correctly frames semantic validity as co-equal with execution policy.
  CONTRIBUTING.md#contributing_contract: Re-reviewed contributor workflow guardrails and confirmed this readme still delegates operational checks to CONTRIBUTING.
doc_id: in_readme
doc_role: readme
doc_scope:
  - repo
  - documentation
  - readme
doc_authority: informative
doc_sections:
  in_readme: 2
doc_section_requires:
  in_readme:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
    - CONTRIBUTING.md#contributing_contract
doc_section_reviews:
  in_readme:
    POLICY_SEED.md#policy_seed:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: Policy seed anchor still governs execution/CI safety exactly as referenced here.
    glossary.md#contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: Glossary contract anchor remains the normative semantic companion cited by this readme.
    CONTRIBUTING.md#contributing_contract:
      dep_version: 1
      self_version_at_review: 2
      outcome: no_change
      note: Contributor contract anchor remains the correct operational workflow reference.
---

<a id="in_readme"></a>

# Prism VM

Prism VM is a small JAX-backed interpreter for a tiny IR (zero/suc/add/mul) with
deduplication, basic static optimization, and kernel dispatch, plus an
experimental BSP arena pipeline.

## Maintenance cadence for `in/` normative artifacts
- Re-run docflow anchor review after any `doc_revision` change in `POLICY_SEED.md`, `glossary.md`, `CONTRIBUTING.md`, or referenced `in/` design docs.
- Treat stale dependency entries in `out/docflow_section_reviews.md` as a same-cycle maintenance task for `in/` docs: update `doc_reviewed_as_of`, `doc_review_notes`, and section review metadata together.
- For steady-state periods, perform a lightweight monthly review sweep of `in/` normative frontmatter to prevent recurrent drift.

## Requirements
- Python via mise (`mise.toml`)
- JAX (CPU or CUDA)

## Setup
CPU-only:
```
mise exec -- python -m pip install jax jaxlib
```

CUDA 12:
```
mise exec -- python -m pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Run
Baseline REPL:
```
mise exec -- python prism_vm.py
```

Baseline program file:
```
mise exec -- python prism_vm.py tests/test_add_cache.txt
```

BSP REPL:
```
mise exec -- python prism_vm.py --mode bsp
```

BSP program file (with multiple cycles):
```
mise exec -- python prism_vm.py --mode bsp --cycles 3 tests/test_add_cache.txt
```

Disable sort in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --no-sort tests/test_add_cache.txt
```

Enable Morton ordering in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --morton tests/test_add_cache.txt
```

Enable block-local (hierarchical) sorting in BSP mode:
```
mise exec -- python prism_vm.py --mode bsp --block-size 256 tests/test_add_cache.txt
```

Swizzle backend (optional GPU acceleration, falls back to JAX on CPU; set `pallas` or `triton`):
```
PRISM_SWIZZLE_BACKEND=triton mise exec -- python prism_vm.py --mode bsp --morton tests/test_add_cache.txt
```

Benchmark compare matrix (baseline + BSP variants, CSV output):
```
mise exec -- python bench_compare.py --runs 3 --cycles 3 --out bench_results.csv
```

Benchmark with swizzle backend sweep:
```
mise exec -- python bench_compare.py --swizzle-backends jax,pallas,triton --runs 3 --cycles 3 --out bench_results.csv
```

Benchmark with hierarchical modes (L2/L1/global) enabled:
```
mise exec -- python bench_compare.py --hierarchy-l1-mult 4 --runs 3 --cycles 3 --out bench_results.csv
```

Note: hierarchy modes are included by default; use `--hierarchy-no-global` to drop the global stage, or `--hierarchy-morton` to include Morton variants.

Target hierarchy stress workloads only:
```
mise exec -- python bench_compare.py --workloads arena_hierarchy_l2256_l11024 --runs 3 --cycles 3 --out bench_results.csv
```

Override hierarchy workload sizes explicitly:
```
mise exec -- python bench_compare.py --hierarchy-workload-l2 128 --hierarchy-workload-l1 512 --runs 3 --cycles 3 --out bench_results.csv
```

Sweep arena sizes and block sizes to find inflection points:
```
mise exec -- python bench_compare.py --block-sizes 64,128,256,512 --arena-counts 8000,16000,32000 --runs 3 --cycles 3 --out bench_results.csv
```

Phoronix-style suite (CSV + Markdown + SVG/PNG plots, CPU/GPU):
```
mise exec -- python bench_phoronix.py --block-sizes 64,128,256,512 --arena-counts 8000,16000,32000 --runs 3 --cycles 3 --out-dir bench_phoronix
```

Note: `bench_phoronix.py` uses matplotlib if available; otherwise it falls back to a minimal built-in plotter.

## Testing
Install pytest (once):
```
mise exec -- python -m pip install pytest
```

Run the suite:
```
mise exec -- pytest
```

## Agda proofs
Agda checks run in a pinned container image. See `agda/README.md#repo_contract` for the
current digest and full instructions. Quick local run:
```
scripts/check_agda_container.sh
```

## Telemetry
Summarize damage metrics from CI artifacts (or local runs):
```
mise exec -- python scripts/collect_damage_metrics.py \
  --base collected_report/raw \
  --out collected_report/damage_metrics_summary.md
```

Capture host performance and memory baselines (record-only):
```
mise exec -- python scripts/audit_host_performance.py \
  --engine intrinsic --iterations 10 --warmup 1 \
  --json-out artifacts/host_perf_intrinsic.json

mise exec -- python scripts/audit_memory_stability.py \
  --engine intrinsic --iterations 10 --warmup 1 \
  --json-out artifacts/host_memory_intrinsic.json
```

Capture and analyze a CPU trace (record-only):
```
mise exec -- python scripts/capture_trace.py \
  --engine intrinsic --iterations 5 --warmup 1 \
  --out-dir /tmp/jax-trace

mise exec -- python scripts/trace_analyze.py \
  --report-only --json-out artifacts/trace_cpu_report.json
```

Summarize host telemetry + trace baselines from artifacts:
```
mise exec -- python scripts/collect_host_metrics.py \
  --base collected_report/raw \
  --out collected_report/host_metrics_summary.md

mise exec -- python scripts/collect_telemetry_baselines.py \
  --base collected_report/raw \
  --damage-summary collected_report/damage_metrics_summary.md \
  --host-summary collected_report/host_metrics_summary.md \
  --out collected_report/telemetry_baselines.md
```

Unpack a collected-report zip locally:
```
mise exec -- python scripts/unpack_collected_report.py \
  --zip "artifacts/collected-report (2).zip" \
  --out-dir artifacts
```

Push, watch CI, and download artifacts:
```
scripts/ci_watch.sh --artifacts-dir artifacts
```

## Policy
This repo uses a self-hosted runner. Read `POLICY_SEED.md#policy_seed` before changing any
workflow or CI behavior. Install advisory hooks with:
```
scripts/install_policy_hooks.sh
```

Semantic correctness is governed by `glossary.md#contract` and is a co-equal normative
contract alongside `POLICY_SEED.md#policy_seed`.

See `CONTRIBUTING.md#contributing_contract` for the guardrails and required checks.

## Milestones
m1 semantic commitments: Ledger interning uses full key-byte equality, univalence
hard-cap is enforced (overflow => corrupt), corrupt/oom are sticky stop-paths
(no further mutation), and baseline vs ledger equivalence holds on the m1 suite.
Changes to these commitments require a milestone bump and updates in
`MILESTONES.md`.

m1-only mode is deprecated. The baseline suite (see `pytest.baseline.ini`)
still runs the m1 test set, but under the current baseline milestone (from
`.pytest-milestone`, currently `m3`), not under an m1-restricted semantic mode.

## Repo layout
- `prism_vm.py` - VM, kernels, and REPL
- `tests/` - pytest suite and sample program fixtures
- `mise.toml` - Python toolchain config
- `in/` - design notes and evolution documents
