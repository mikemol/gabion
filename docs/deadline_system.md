---
doc_revision: 2
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
doc_id: deadline_system
doc_role: contract
doc_scope:
  - repo
  - analysis
  - tooling
  - semantics
doc_authority: informative
doc_requires:
  - POLICY_SEED.md#policy_seed
  - glossary.md#contract
doc_reviewed_as_of:
  POLICY_SEED.md#policy_seed: 57
  glossary.md#contract: 47
doc_review_notes:
  POLICY_SEED.md#policy_seed: "Reviewed POLICY_SEED.md rev57; the deadline system operates within deadline_scope/GasMeter bounds on all analysis paths; no execution on untrusted surfaces; consistent with policy."
  glossary.md#contract: "Reviewed glossary.md rev47; ASPF forest, never_throw_exception_protocol, and related terms are used per glossary definitions."
doc_change_protocol: "POLICY_SEED.md#change_protocol"
doc_sections:
  deadline_system: 1
doc_section_requires:
  deadline_system:
    - POLICY_SEED.md#policy_seed
    - glossary.md#contract
doc_section_reviews:
  deadline_system:
    POLICY_SEED.md#policy_seed:
      dep_version: 57
      self_version_at_review: 1
      outcome: no_change
      note: "Policy seed rev57 reviewed; deadline enforcement model is consistent with the process-relative runtime invariant and never_throw_exception_protocol."
    glossary.md#contract:
      dep_version: 47
      self_version_at_review: 1
      outcome: no_change
      note: "Glossary rev47 reviewed; ASPF and related terms used correctly."
doc_erasure:
  - formatting
  - typos
doc_owner: maintainer
---

<a id="deadline_system"></a>
# Deadline System

The deadline system is the analysis-timeout substrate for all long-running
operations in gabion. It enforces budget limits on analysis code using a
two-layer model: a **wall-clock hard bound** (a `Deadline` carrying an absolute
nanosecond timestamp) and a **logical-tick soft bound** (a `GasMeter` counting
`check_deadline()` calls). The tick count is the primary enforcement mechanism
inside analysis code; the wall clock is the secondary bound used by the LSP
client layer and by the `_DeadlineFlowBuffer` scheduling algorithm.

The system spans five source layers:

```
gabion.deadline_clock          — Clock protocol, MonotonicClock, GasMeter
gabion.analysis.foundation     — Deadline, scopes, check_deadline, profiling
gabion.runtime.deadline_policy — DeadlineBudget, canonical scope constructors
gabion.runtime.env_policy      — Runtime timeout configuration and overrides
gabion.cli_support / cli.py    — CLI integration, DEFAULT_CLI_TIMEOUT_TICKS
gabion.lsp_client              — LSP client composite deadline formula
```

---

## Clock Layer

### `DeadlineClock` protocol

```python
class DeadlineClock(Protocol):
    def consume(self, ticks: int = 1) -> None: ...
    def get_mark(self) -> int: ...
```

`consume(ticks)` advances the clock by `ticks` units and may raise
`DeadlineClockExhausted` if the budget is exhausted.
`get_mark()` returns a monotonically non-decreasing integer mark used for
profiling and delta calculations. The unit of `get_mark()` depends on the
implementation — nanoseconds for `MonotonicClock`, accumulated ticks for
`GasMeter`.

### `MonotonicClock`

```python
@dataclass(frozen=True)
class MonotonicClock:
    def consume(self, ticks: int = 1) -> None:
        return   # no-op
    def get_mark(self) -> int:
        return time.monotonic_ns()
```

`consume()` is a no-op. `MonotonicClock` never raises `DeadlineClockExhausted`.
`get_mark()` returns nanoseconds since an arbitrary epoch (system boot).
Used in unit tests and in any context where wall time is tracked but
deterministic tick enforcement is not wanted. When a `MonotonicClock` is
active, `check_deadline()` calls are observable (profiling still runs) but
not load-bearing — they will not cause the analysis to time out.

### `GasMeter`

```python
@dataclass
class GasMeter:
    limit: int      # Maximum ticks before exhaustion
    current: int = 0

    def consume(self, ticks: int = 1) -> None:
        self.current += int(ticks)
        if self.current >= self.limit:
            raise DeadlineClockExhausted(...)

    def get_mark(self) -> int:
        return self.current
```

`GasMeter` is a deterministic counter. Each call to `consume(ticks)` advances
`current` by `ticks`. When `current >= limit`, it raises
`DeadlineClockExhausted`. `get_mark()` returns the accumulated tick count.

The `GasMeter` is mutable and not thread-safe; it must not be shared across
concurrent execution contexts.

### `DeadlineClockExhausted`

```python
class DeadlineClockExhausted(RuntimeError): ...
```

Raised only by `GasMeter.consume()`. This is a raw machine exception and
**must not escape** `consume_deadline_ticks()` into call sites unless the
forest context variable is unset (test-only fallback). All call sites above
`consume_deadline_ticks()` receive `TimeoutExceeded` instead.

---

## Deadline Layer

### `TimeoutTickCarrier`

```python
@dataclass(frozen=True)
class TimeoutTickCarrier:
    ticks: int
    tick_ns: int

    @classmethod
    def from_ingress(cls, *, ticks: object, tick_ns: object) -> "TimeoutTickCarrier": ...
```

Represents a duration as `ticks` abstract units each of `tick_ns` nanoseconds.
Total duration = `ticks × tick_ns` nanoseconds. `from_ingress` accepts any
integer-coercible values and validates both are positive. The dominant
convention throughout the codebase is `tick_ns = 1_000_000` (1 ms per tick),
making `ticks` a millisecond count.

### `Deadline`

```python
@dataclass(frozen=True)
class Deadline:
    deadline_ns: int   # Absolute wall-clock nanosecond timestamp

    @classmethod
    def from_timeout_ticks(cls, carrier: TimeoutTickCarrier) -> "Deadline": ...
    @classmethod
    def from_timeout_ms(cls, milliseconds: int) -> "Deadline": ...

    def expired(self) -> bool: ...
    def check(self, builder: Callable[[], TimeoutContext]) -> None: ...
```

`deadline_ns` is an **absolute** wall-clock timestamp, computed at construction
as `time.monotonic_ns() + ticks × tick_ns`. Once created, it does not change.

**`expired()` and `check()` are wall-clock operations.** They read
`time.monotonic_ns()` at call time. `check()` raises `TimeoutExceeded` if
`expired()` is true. These are used by the LSP client and the
`_DeadlineFlowBuffer` scheduling algorithm, not by the inner
`check_deadline()` hot path.

The wall-clock deadline is a **hard bound**: if the physical clock crosses
`deadline_ns`, no further I/O wait will be granted. The GasMeter is the
**primary enforcement mechanism** inside analysis loops. Both bounds are active
simultaneously.

### `TimeoutExceeded`

```python
class TimeoutExceeded(TimeoutError):
    def __init__(self, context: TimeoutContext) -> None: ...
    context: TimeoutContext
```

The primary timeout exception raised within analysis code. Always carries a
`TimeoutContext` with:
- `call_stack` — packed, deduplicated execution site chain
- `deadline_profile` — optional heat map (site, edge, and I/O statistics)
- `forest_spec_id`, `forest_signature` — ASPF forest identity
- `progress` — classification payload (e.g.
  `"timed_out_progress_resume"` or `"timed_out_no_progress"`)

`TimeoutExceeded` is a `TimeoutError` and should be caught at the server
orchestrator stage boundary. It must not be silenced or re-raised as a
different exception type by intermediate layers.

---

## Context Management

All deadline state is held in `contextvars.ContextVar` instances. Three are
required to be active simultaneously for `check_deadline()` to function:

| ContextVar | Type | Sentinel / default |
|---|---|---|
| `_deadline_var` | `Deadline` | `None` (error if missing) |
| `_deadline_clock_var` | `DeadlineClock` | `None` (error if missing) |
| `_forest_var` | `Forest` | `_MISSING_FOREST` (error if missing unless bypassing) |

A fourth is optional:

| ContextVar | Type | Purpose |
|---|---|---|
| `_deadline_profile_var` | `_DeadlineProfileState \| None` | Heat-map collection |

### Scope managers

```python
deadline_scope(deadline: Deadline)         # sets _deadline_var
deadline_clock_scope(clock: DeadlineClock) # sets _deadline_clock_var
forest_scope(forest: Forest)               # sets _forest_var
```

Each is a **replace-with-restore** context manager using `ContextVar.set()`
and `ContextVar.reset()`. Exiting the scope restores the previous value.
Nested scopes are correctly restored in LIFO order. This means an inner
`deadline_scope` replaces (does not merge with) the outer one for the duration
of the inner block — the outer deadline is restored on exit.

**Footgun — inner scope with a longer deadline suspends the outer**: if a
callee opens a new `deadline_scope` carrying a deadline farther in the future
than the caller's outer deadline, the outer deadline is **suspended** for the
duration of the inner block. It will not fire. The callee's deadline governs
instead. The same applies to `deadline_clock_scope`: an inner scope's
`GasMeter` replaces the outer one, so `check_deadline()` calls inside the
callee deplete only the inner budget. The outer GasMeter does not accumulate
and cannot expire while suspended.

This suspension is intentional for LSP dispatch: `lsp_client.run_command()`
sets both a longer wall-clock deadline and a large logical-limit GasMeter so
that I/O-bound work is not penalised by a tight outer CLI budget. Callers that
expect an outer deadline to provide a hard cancellation guarantee must be aware
that callees can extend, not only shorten, the effective bound.

**Invariant**: `check_deadline()` will call `never()` (unconditional assertion
failure) if either `_deadline_var` or `_deadline_clock_var` is unset. Missing
the forest does not cause a `never()` assertion; instead,
`consume_deadline_ticks()` propagates the raw `DeadlineClockExhausted` rather
than wrapping it (test-only path).

### Canonical setup: `deadline_scope_from_ticks()`

```python
@contextmanager
def deadline_scope_from_ticks(
    budget: DeadlineBudget,
    *,
    gas_limit: int | None = None,
) -> Iterator[None]:
    limit = budget.ticks if gas_limit is None else int(gas_limit)
    with forest_scope(Forest()):
        with deadline_scope(Deadline.from_timeout_ticks(
            TimeoutTickCarrier.from_ingress(ticks=budget.ticks, tick_ns=budget.tick_ns)
        )):
            with deadline_clock_scope(GasMeter(limit=limit)):
                yield
```

`deadline_scope_from_ticks()` is the **canonical entry point** for all
deadline-protected analysis regions. It sets up all three required context
variables in the correct nesting order. The `gas_limit` parameter allows
decoupling the GasMeter limit from the tick budget (used when the GasMeter is
rate-limited relative to the deadline, e.g. for the LSP client's logical
progress budget vs. the analysis's tick budget).

All analysis code that calls `check_deadline()` must be invoked within a
`deadline_scope_from_ticks()` context or an equivalent manual triple-nesting
of the three scope managers.

---

## Tick Consumption

### `consume_deadline_ticks(ticks=1, ...)`

```python
def consume_deadline_ticks(ticks: int = 1, *, project_root=None,
                           forest_spec_id=None, forest_signature=None,
                           allow_frame_fallback: bool = True) -> None:
```

The fundamental tick-consumption call. Calls `clock.consume(ticks)` on the
active `DeadlineClock`. If the clock is a `GasMeter` and it raises
`DeadlineClockExhausted`:

- If `_forest_var` is set: builds a `TimeoutContext` (call stack, profile
  snapshot, forest identity) and raises `TimeoutExceeded`.
- If `_forest_var` is not set: propagates `DeadlineClockExhausted` raw (test
  environments that have not set up a forest).

### `check_deadline(...)`

```python
def check_deadline(deadline=None, *, project_root=None, forest_spec_id=None,
                   forest_signature=None, allow_frame_fallback: bool = True) -> None:
```

The primary call-site primitive for analysis code. Validates that both
`_deadline_var` and `_deadline_clock_var` are active (raises `never()` if
either is missing), then calls `consume_deadline_ticks()`, then records a
profiling sample.

`check_deadline()` does **not** check the wall-clock deadline directly. The
wall clock is checked by `_wait_readable()` in the LSP client and by
`_DeadlineFlowBuffer._poll()` when scheduling adaptive batches.

**Usage rule**: call `check_deadline()` at every meaningful analysis
checkpoint — loop iterations, recursive descent steps, file-level scan
boundaries. Do not call it in tight arithmetic loops where the overhead
exceeds the analytical value.

### `deadline_loop_iter(values: Iterable[T]) -> Iterator[T]`

```python
def deadline_loop_iter(values: Iterable[T]) -> Iterator[T]:
    flow_buffer = _DeadlineFlowBuffer.open()
    for value in values:
        flow_buffer.before_item()
        yield value
        flow_buffer.after_item()
```

Drop-in wrapper for hot iteration loops. Calls `check_deadline()` adaptively
using `_DeadlineFlowBuffer` rather than on every item. The adaptive batching
ensures deadline responsiveness without paying the profiling overhead per
iteration.

**Usage rule**: use `deadline_loop_iter()` on any iteration over a collection
whose size is not bounded at a small constant. Never call `check_deadline()`
inside the loop body if using `deadline_loop_iter()` — it double-counts ticks.

---

## Adaptive Batching (`_DeadlineFlowBuffer`)

`_DeadlineFlowBuffer` implements Vegas-style adaptive window control for
`deadline_loop_iter()`.

### Constants

| Constant | Value | Meaning |
|---|---|---|
| `_TIMEOUT_PROGRESS_CHECKS_FLOOR` | 32 | Minimum ticks before first window check |
| `_TIMEOUT_PROGRESS_SITE_FLOOR` | 4 | Minimum checks at a site before tracking per-site rate |
| `_TIMEOUT_FLOW_INITIAL_WINDOW` | 64 | Initial batch size |
| `_TIMEOUT_FLOW_EWMA_ALPHA` | 0.35 | EWMA smoothing factor |
| `_TIMEOUT_FLOW_BEST_RATE_DECAY` | 0.95 | Decay rate for best-rate tracking |
| `_TIMEOUT_FLOW_UTILIZATION_FRACTION` | 0.5 | Fraction of remaining budget to fill |
| `_TIMEOUT_FLOW_STABILITY_TOLERANCE` | 0.25 | Rate-drop threshold triggering backoff |
| `_TIMEOUT_FLOW_STABLE_GROWTH_FACTOR` | 8 | Window growth when throughput is stable |
| `_TIMEOUT_FLOW_UNSTABLE_GROWTH_FACTOR` | 4 | Window growth when throughput is variable |
| `_TIMEOUT_FLOW_BACKOFF_FACTOR` | 0.75 | Multiplicative backoff on congestion |
| `_TIMEOUT_FLOW_HARD_CAP` | 4096 | Hard cap on window size |
| `_TIMEOUT_FLOW_MIN_RATE` | 1e-18 | Throughput rate floor (avoids division by zero) |

### Mechanics

The buffer tracks two throughput metrics separately: `items_per_tick` (logical
clock rate) and `items_per_ns` (wall-clock rate). Both use EWMA smoothing with
a "best recent rate" decay tracker that resists outliers.

On each `_poll()`, the buffer:
1. Calls `check_deadline()` to consume one tick and validate the clock.
2. Measures elapsed ticks and wall time since the last poll.
3. Updates EWMA and best-rate estimates.
4. Detects throughput drops: if current rate < `(1 - stability_tolerance) ×
   best_rate`, applies the backoff factor.
5. Projects remaining budget (in ticks and nanoseconds) against estimated
   throughput to compute the next window.
6. Clamps to `[1, HARD_CAP]`.

The dual tracking (ticks + wall time) ensures the window adapts correctly under
both GasMeter (tick-bound) and MonotonicClock (wall-bound) regimes.

---

## Deadline Profiling

Profiling is optional and disabled by default for call sites that don't call
`set_deadline_profile()`. When enabled, `_record_deadline_check()` captures:

- **Sites**: call locations (file path, qualified name), check count, total
  elapsed time, maximum gap between consecutive checks.
- **Edges**: site-to-site transitions, transition count, elapsed time, max gap.
- **I/O events**: via explicit `record_deadline_io(name, elapsed_ns, bytes_count)`
  calls at I/O boundaries.

Profiling is **sampled**: `sample_interval` controls how many checks are
skipped between recorded samples. At `sample_interval=1` every check is
recorded; higher values reduce overhead proportionally.

The profiling snapshot is attached to `TimeoutExceeded.context.deadline_profile`
and is also written to `artifacts/out/deadline_profile.json` and
`artifacts/out/deadline_profile.md` by the server orchestrator on timeout.

`ticks_per_ns` in the profile snapshot is a **derived measurement**:

```
ticks_per_ns = ticks_consumed / wall_total_elapsed_ns
```

This is the empirical conversion factor between logical gas ticks and
nanoseconds of real time. It is not a configuration parameter; it is computed
post-hoc from the observed run. A high `ticks_per_ns` (many checks per ns)
means `check_deadline()` is being called very frequently — the gas budget
exhausts quickly in wall time.

---

## Budget adequacy

The deadline profile's `ticks_per_ns` (ticks consumed ÷ wall elapsed ns) is
the empirical conversion factor between logical and wall-time budgets. Use it
to validate that `budget.ticks` is adequate for the intended wall-clock budget:

```
required_ticks ≈ intended_wall_budget_ns × ticks_per_ns
```

If `required_ticks` exceeds `budget.ticks`, the GasMeter will exhaust before
the wall-clock deadline — the effective budget is shorter than intended. Remedy:
increase `ticks`, increase `tick_ns` (coarser granularity), or reduce
`check_deadline()` call density.

Practical heuristics:

- **High `ticks_per_ns` on I/O-bound paths**: at `ticks_per_ns ≈ 1.1 × 10⁻⁴`
  (server-startup import chains), 7,500 ticks exhaust in ~68 ms of wall time
  even with a 7.5 s wall budget. Paths where `check_deadline()` fires inside
  import-driven code should either carry a large `gas_limit` (decoupled from
  the tick budget) or use `MonotonicClock` if tick counting is not meaningful.

- **Hot-path site dominating the profile**: if one site accounts for > 50% of
  total checks, it is calling `check_deadline()` on every iteration of a large
  loop. Convert it to `deadline_loop_iter()`, which batches checks adaptively.

- **Decoupled `gas_limit`**: `gas_limit >> budget.ticks` is appropriate when
  the primary enforcement bound is wall time (e.g. LSP I/O loops), not logical
  progress. The large GasMeter still catches runaway loops but does not penalise
  normal I/O-bound work. `gas_limit = budget.ticks` (the default) is appropriate
  when tick density is uniform and progress-proportional exhaustion is the goal.

---

## Configuration Layer

### `DeadlineBudget`

```python
@dataclass(frozen=True)
class DeadlineBudget:
    ticks: int      # Logical tick budget
    tick_ns: int    # Nanoseconds per tick (conversion factor)
```

Both fields must be positive. Total wall-clock budget = `ticks × tick_ns`.

### Defaults

| Constant | Value | Context |
|---|---|---|
| `deadline_policy.DEFAULT_TIMEOUT_TICKS` | 120,000 | Server-side default (120 s at 1 ms/tick) |
| `deadline_policy.DEFAULT_TIMEOUT_TICK_NS` | 1,000,000 | 1 ms per tick |
| `dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICKS` | 7,500 | CLI default (7.5 s at 1 ms/tick) |
| `dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICK_NS` | 1,000,000 | 1 ms per tick |

The server default (120 s) is used for analysis dispatch when the CLI or LSP
client does not inject an explicit `analysis_timeout_ticks` payload field. The
CLI default (7.5 s) is used for CLI command wrappers and LSP client startup.

### `timeout_budget_from_lsp_env()`

```python
def timeout_budget_from_lsp_env(
    *,
    default_budget: DeadlineBudget = DEFAULT_TIMEOUT_BUDGET,
) -> DeadlineBudget:
    override = env_policy.lsp_timeout_override()
    if override is not None:
        return DeadlineBudget(ticks=override.ticks, tick_ns=override.tick_ns)
    return default_budget
```

Reads the current `LspTimeoutConfig` from `env_policy._LSP_TIMEOUT_OVERRIDE`
(a `ContextVar`). If set (by `apply_cli_timeout_flag()` or
`apply_cli_timeout_flags()`), returns it as a `DeadlineBudget`; otherwise
returns `default_budget`. This is the single integration point between the
CLI `--timeout` flag and the deadline machinery.

### Timeout input formats

`env_policy` accepts timeout specifications in four forms:

| Form | Example | Parsing |
|---|---|---|
| Duration string | `"30s"`, `"2m30s"`, `"100ns"` | `timeout_config_from_duration()` using Decimal arithmetic |
| Ticks + tick_ns | `ticks=7500, tick_ns=1_000_000` | `timeout_config_from_cli_flags()` |
| Milliseconds | `ms=7500` | `timeout_config_from_cli_flags()` |
| Seconds (float or Decimal) | `seconds=7.5` | `timeout_config_from_cli_flags()` |

All forms produce a `LspTimeoutConfig(ticks, tick_ns)` with `tick_ns =
1_000_000` (1 ms per tick) as the canonical output.

---

## CLI Integration

### `_cli_deadline_scope()`

```python
@contextmanager
def _cli_deadline_scope():
    ticks, tick_ns = _cli_timeout_ticks()
    with deadline_policy.deadline_scope_from_ticks(
        deadline_policy.DeadlineBudget(ticks=ticks, tick_ns=tick_ns),
        gas_limit=int(ticks),
    ):
        yield
```

Wraps every deadline-bearing CLI command handler. Reads `DEFAULT_CLI_TIMEOUT_TICKS`
(subject to `--timeout` override via `timeout_budget_from_lsp_env()`), then
establishes the full triple scope (forest + deadline + GasMeter). The
`gas_limit` equals the tick budget — no decoupling.

**Interaction with LSP subprocess commands**: `_cli_deadline_scope()` sets the
GasMeter limit to `DEFAULT_CLI_TIMEOUT_TICKS` (7,500 by default). For commands
that spawn an LSP subprocess (e.g. `lsp-parity-gate`), `lsp_client.run_command()`
opens its own `deadline_scope` and `deadline_clock_scope` via `ContextVar.set()`.
This **suspends** both the outer CLI deadline and the outer CLI GasMeter for the
duration of the LSP call. `check_deadline()` calls inside the LSP I/O loop
consume ticks only from the inner GasMeter (`logical_limit = max(10_000,
lsp_ticks × 1_000)`). The outer GasMeter does not accumulate. Both outer
scopes are restored when `run_command()` returns.

For CLI commands that only do short coordination work (not deep analysis),
the 7,500-tick GasMeter budget is sufficient. For commands that invoke analysis
as a subprocess, the analysis runs in a **child process** with its own deadline
scope set from the payload's `analysis_timeout_ticks` — the outer
`_cli_deadline_scope()` GasMeter does not affect the child.

---

## LSP Client Integration

### Composite deadline formula

When `lsp_client.run_command()` is called, it computes a single `lsp_total_ns`
that covers both the analysis budget and server startup overhead:

```
base_total_ns        = ticks × tick_ns
analysis_target_ns   = payload.analysis_timeout_ticks × payload.analysis_timeout_tick_ns
                       (if present in payload, else = base_total_ns)
slack_ns             = clamp(analysis_target_ns ÷ 3, min=2s, max=120s)
lsp_total_ns         = max(base_total_ns, analysis_target_ns + slack_ns)
lsp_ticks            = ceil(lsp_total_ns ÷ tick_ns)
```

The slack is the margin added to the analysis budget to allow the server time
to start up, handle LSP handshake, complete cleanup on timeout, and shut down.
The minimum 2-second slack ensures that a short analysis budget doesn't produce
a deadline tighter than server startup time. With `DEFAULT_CLI_TIMEOUT_TICKS =
7_500` (7.5 s base), `slack_ns = 2.5 s`, giving `lsp_total_ns = 10 s`.

The LSP client also creates a `GasMeter` with `logical_limit = max(10_000,
lsp_ticks × 1_000)` for the I/O loop. This is separate from the outer CLI
GasMeter and governs the `check_deadline()` calls inside `_read_rpc()` and
`_read_exact()`. It is intentionally large relative to the wall-clock deadline
— the I/O loop should not exhaust its logical budget before the wall-clock
timeout fires.

### `_wait_readable()` — the wall-clock enforcement point

```python
def _wait_readable(stream, deadline_ns: int) -> None:
    remaining_ns = deadline_ns - time.monotonic_ns()
    timeout = max(0.0, remaining_ns / 1_000_000_000)
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        raise LspClientError("LSP response timed out")
```

`_wait_readable()` is the only place in the LSP client where the wall-clock
deadline is enforced via `select.select`. If the server subprocess does not
write data to stdout within the remaining deadline, `LspClientError("LSP
response timed out")` is raised. This is a `RuntimeError`, not a
`TimeoutExceeded`, and indicates a server startup or communication failure —
not a normal analysis timeout.

---

## Server Orchestrator Integration

The server's `command_orchestrator.py` reads `analysis_timeout_ticks` and
`analysis_timeout_tick_ns` from the command payload and establishes the
analysis deadline scope via `deadline_scope_from_ticks()`. Analysis code then
calls `check_deadline()` throughout execution.

On `TimeoutExceeded`, the orchestrator:

1. Calls the registered timeout cleanup handler.
2. Invokes `timeout_classification_decision()` which reads the `classification`
   field from the progress payload (set by ASPF progress tracking):
   - `"timed_out_progress_resume"` — analysis made progress; state can be
     resumed.
   - `"timed_out_no_progress"` — no progress; retry from scratch.
3. Attaches the deadline profile to the response.
4. Returns the timeout response to the LSP client with `exit_code: 2`.

The `classification` field is written by the ASPF handoff machinery and
reflects whether any forest nodes or ASPF progress checkpoints were recorded
before the timeout fired.

---

## Exception handling contract

| Exception | Source | Correct handler |
|---|---|---|
| `DeadlineClockExhausted` | `GasMeter.consume()` | `consume_deadline_ticks()` — must not escape further |
| `TimeoutExceeded` | `consume_deadline_ticks()` (after wrapping) | Server orchestrator stage boundary |
| `LspClientError("LSP response timed out")` | `_wait_readable()` | LSP client caller (CLI) |

**Never** catch `TimeoutExceeded` and re-raise it as a different exception
type. Its `context` payload is required by the orchestrator for classification
and reporting.

**Never** catch `DeadlineClockExhausted` outside of `consume_deadline_ticks()`.
Its raw form carries no diagnostic context and violates the contract that all
timeout exits carry a `TimeoutContext`.

---

## Idiomatic usage patterns

### Setting up a deadline scope for analysis

```python
# Canonical — use this in all new analysis entry points
with deadline_policy.deadline_scope_from_ticks(
    deadline_policy.DeadlineBudget(ticks=budget_ticks, tick_ns=1_000_000)
):
    run_analysis(...)

# Equivalent alternative using env override
with deadline_policy.deadline_scope_from_lsp_env():
    run_analysis(...)
```

### Checking the deadline at a loop boundary

```python
# For bounded collections where size is known small: call directly
for item in small_list:
    check_deadline()
    process(item)

# For unbounded or large collections: use adaptive batching
for item in deadline_loop_iter(large_collection):
    process(item)
```

### Checking the deadline at a recursive descent step

```python
def descend(node):
    check_deadline()   # At the top of each recursive step
    for child in node.children:
        descend(child)
```

### Recording I/O for profiling

```python
t0 = time.monotonic_ns()
data = file.read()
record_deadline_io(name="file_read", elapsed_ns=time.monotonic_ns() - t0,
                   bytes_count=len(data))
```

### Anti-patterns

| Anti-pattern | Problem |
|---|---|
| Calling `check_deadline()` outside a `deadline_scope` and `deadline_clock_scope` | `never()` assertion failure at runtime |
| Catching `TimeoutExceeded` and discarding it | Loss of timeout context; orchestrator cannot classify |
| Catching `DeadlineClockExhausted` directly in analysis code | Bypass of context-building; violates the contract |
| Using `deadline_loop_iter()` and also calling `check_deadline()` inside the loop body | Double-counting ticks; exhausts budget prematurely |
| Creating a `GasMeter` with `limit=0` | Immediate `never()` assertion failure in `GasMeter.__post_init__` |
| Sharing a `GasMeter` across threads | Race condition on `current`; undefined exhaustion behavior |
