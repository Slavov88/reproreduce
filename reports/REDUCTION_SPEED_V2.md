# Reduction Speed v2: Search Deduplication

## Status

**COMPUTATIONALLY VERIFIED:** exact-source scheduler deduplication reduced evaluation-layer candidate requests while preserving the reductions, source hashes, fingerprints, and standalone reproducers.

**OBSERVED:** it did not reduce fresh oracle executions, because v1 in-memory memoization had already prevented duplicate sources from reaching the oracle. Wall-time measurements were noisy and unfavorable; no speedup claim is made.

## Baseline and environment

- Branch: `feat/reducer-search-speed`
- Code commit: `2497018bf19f00a59b440ce216cad18f3c886ebd`
- Environment: Windows 10, Python 3.10.11, PyTorch 2.5.1+cu121
- Candidate timeout: 10 seconds
- Three repeats per benchmark and mode
- Final full suite: 108 tests passed in 1,461 seconds at `e88ad50`
- Baseline mode: `scheduler_deduplicate=false`
- Optimized mode: `scheduler_deduplicate=true`
- Full search traces retained in compressed `reports/speed_search_traces_v2.json.gz` (baseline and optimized JSON objects)

## Metric definitions

- **scheduler request:** a generated candidate is offered to the scheduler.
- **candidate request:** the evaluation layer is asked to classify a candidate after scheduler filtering.
- **unique source candidate:** a distinct candidate SHA-256 seen by the scheduler; direct baseline/final evaluations are tracked separately.
- **oracle execution:** a fresh candidate subprocess/oracle execution (`candidate_runs`).
- **memory cache hit:** an evaluation answered by the in-process source memo.
- **SQLite cache hit:** an evaluation answered by persistent SQLite.
- **skipped duplicate:** the scheduler reused a prior exact-source outcome before session evaluation.

## Search trace analysis

The trace recorded transformation, scope, collection size, candidate size, granularity, parent hash, candidate hash, evaluation kind, acceptance, and rejection reason. The raw trace archive is compressed to keep the tracked report compact.

The baseline first-occurrence oracle execution order was identical to the optimized order for both benchmarks. The optimization only replaces later exact-source requests with the prior Boolean outcome.

## Duplicate causes

| Benchmark | Duplicate source requests | Distinct duplicated sources | Identical ddmin state | Granularity repeat | Restart/changed parent | Non-AST/final normalization | No-op |
|---|---:|---:|---:|---:|---:|---:|---:|
| Large exception | 18 | 12 | 4 | 1 | 9 | 4 | 0 |
| Nested Python | 177 | 88 | 118 | 25 | 12 | 22 | 0 |

Interpretation is based on trace-local context. “Restart/changed parent” means the same final source arose after a different parent state or reduction restart. “Non-AST/final normalization” covers transformations without statement-scope metadata, including tensor/module/final candidates. No no-op source equal to its recorded parent was observed.

## Optimization implemented

Added a generic `_SessionCandidateTest` scheduler. It hashes each generated source before calling the evaluation layer. If the same source hash was already tested in the current reduction, it reuses the prior acceptance result and records `skipped_duplicate` instead of performing another cache lookup or oracle request.

This does not assume monotonicity and does not blacklist rejected structural states globally. Subprocess isolation and oracle acceptance checks are unchanged.

## Before / after

| Benchmark | Requests Before | Requests After | Oracle Exec Before | Oracle Exec After | Time Before | Time After | Speedup | LOC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Large exception | 186 | 168 | 167 | 167 | 161.25s | 199.80s | 0.807× | 4 → 4 |
| Nested Python | 390 | 213 | 212 | 212 | 206.61s | 259.50s | 0.796× | 55 → 55 |

Times are medians of three repeats. The optimized wall times were slower in this measurement despite fewer evaluation requests; candidate execution variance dominates the small scheduler savings. This is a negative speed result, not a claimed regression in reduction semantics.

## Candidate and oracle effects

**Large exception:**

- Scheduler requests: 184 → 184
- Evaluation-layer requests: 186 → 168
- Exact duplicates skipped before evaluation: 18
- Fresh oracle executions: 167 → 167
- Final source hash: unchanged

**Nested Python:**

- Scheduler requests: 388 → 388
- Evaluation-layer requests: 390 → 213
- Exact duplicates skipped before evaluation: 177
- Fresh oracle executions: 212 → 212
- Final source hash: unchanged

The key negative finding is that all eliminated requests were already cache-resolved in v1. Search deduplication reduces evaluation-layer traffic but not expensive subprocess executions.

## Reduction quality

**COMPUTATIONALLY VERIFIED:** three repeats in each mode preserved:

- final nonblank LOC;
- reduced-source SHA-256;
- failure fingerprint;
- standalone export success;
- standalone fresh-subprocess reproduction;
- first-occurrence oracle candidate ordering.

No optimization relied on a monotonicity assumption.

## Historical Inductor result

The v1 warm-cache measurement remains the current compiler baseline:

- 445 candidate requests
- 78 fresh executions
- 367 cache hits
- 574.64 seconds candidate execution
- 577.84 seconds total wall time

A full pass-2 Inductor reduction was not run because the cold-start attempt exceeded 40 minutes. No compiler speedup claim is made.

## Limitations and next bottleneck

1. Exact source deduplication cannot reduce oracle executions when in-memory memoization already handles the duplicates.
2. Scheduler requests were unchanged, so candidates are still rendered before source-level deduplication.
3. Wall-time measurements do not support a speed improvement claim.
4. The next candidate-generation optimization would need to deduplicate canonical reduction states before source rendering, but it must preserve ordering and AST semantics.
5. Persistent workers and monotonicity-based pruning remain out of scope.

## Evidence status

- **COMPUTATIONALLY VERIFIED:** request reduction, unchanged oracle execution counts, unchanged outputs, fingerprint preservation, standalone reproducers, and deterministic oracle order.
- **OBSERVED:** duplicate-cause categories and noisy unfavorable wall-time comparison.
- **NOT TESTED:** optimized full Inductor reduction and canonical AST-state pruning before unparse.

## Addendum: canonical structural-state pass

**COMPUTATIONALLY VERIFIED:** a second scheduling pass added canonical whole-tree AST-state deduplication before candidate source generation. The original V2 results above are preserved; this addendum records the follow-up experiment.

### Exact metric definitions

- `candidate_request`: evaluation layer asked to classify a candidate after scheduler filtering.
- `unique_source_candidate`: distinct source SHA-256 encountered by the scheduler.
- `oracle_execution`: actual fresh subprocess/oracle execution.
- `memory_cache_hit`: result returned by in-process memoization.
- `sqlite_cache_hit`: result returned by persistent SQLite cache.
- `scheduler_duplicate_skip`: exact source reused before `ReductionSession.evaluate()`.
- `structural_state_skip`: canonical full-AST state reused before AST unparse/session evaluation.
- `no_op_skip`: structural transformation proven to leave the current statement state unchanged.
- `syntax_skip`: candidate rejected by local `compile(source, ..., "exec")` before subprocess launch.

The benchmark's `candidate_requests` metric excludes scheduler-level skips. `oracle_executions` is the primary expensive-work metric.

### Ddmin audit

The implementation retains classical ddmin partitioning, complement tests, granularity doubling after unsuccessful rounds, and granularity reduction after an accepted deletion. It does not assume failure monotonicity and does not prune rejected supersets or subsets.

The new reuse hook only reuses an exact canonical full-AST state whose prior Boolean outcome is known. A state key includes the complete AST dump, so an identical local subtree under a changed parent is not conflated with the prior state. Accepted cached states restore the AST before ddmin continues; rejected states restore the current parent state.

### Duplicate causes and scheduling effect

| Benchmark | Structural skips | Exact-source skips after | Scheduler requests before | Scheduler requests after |
|---|---:|---:|---:|---:|
| Large exception | 4 | 7 | 177 | 173 |
| Nested Python | 120 | 41 | 362 | 252 |

The nested benchmark's repeated states are primarily recursive/restart AST states. Canonical state reuse removed 120 pre-unparse attempts. Exact-source skips still handled 41 later duplicates. Large exception had only four structural repeats.

Both modes had one memory-cache hit and zero SQLite-cache hits in these fresh benchmark workspaces. Thus the 120 nested structural skips did not remove any oracle execution: the corresponding sources were already resolved by exact-source/session memoization in the baseline.

### Before / after benchmark

Three repeats per mode were run in the same Windows 10 / Python 3.10.11 / PyTorch 2.5.1+cu121 environment. The baseline disabled only canonical structural-state deduplication; existing exact-source scheduler deduplication remained enabled.

| Benchmark | Requests Before | Requests After | Oracle Exec Before | Oracle Exec After | Time Before | Time After | Speedup | Reduced LOC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Large exception | 168 | 168 | 167 | 167 | 88.82s | 96.29s | 0.922x | 4 |
| Nested Python | 213 | 213 | 212 | 212 | 116.34s | 115.10s | 1.011x | 55 |

Wall-time speedups are below the 1.2x claim threshold and are not headline claims. The large-exception slowdown is consistent with canonical AST-dump bookkeeping without oracle reduction. Nested Python shows a small, noise-level wall-time improvement.

### Quality and safety gate

All six benchmark reductions preserved:

- final reduced nonblank LOC: 273 → 4 and 200 → 55;
- reduced-source SHA-256;
- target failure fingerprint;
- export success;
- standalone fresh-subprocess reproduction;
- deterministic candidate and reduced-source results across three repeats.

No optimization relies on monotonicity. Candidate ordering was not intentionally changed; the ddmin ordering remains unchanged. The new optimization only suppresses exact canonical states. The first-occurrence oracle order and final source hashes remained identical.

### Compile prefilter and CompileDifferenceOracle

Generated AST candidates are locally passed through Python `compile(..., "exec")` before session/oracle evaluation. No syntax skips occurred in these benchmarks. `CompileDifferenceOracle` has no source-level eager/compiled execution pair to short-circuit in its `evaluate_source` adapter; its function-level API already captures eager execution before compilation, but it is not part of subprocess reduction.

### Explicit answers

1. Duplicate candidates came from repeated canonical AST states during recursive block traversal/restarts, plus exact-source convergence after different transformations. Earlier trace categorization measured nested causes as 118 identical ddmin states, 25 granularity repeats, 12 restart/changed-parent cases, and 22 non-AST/final-normalization cases.
2. The canonical state pass eliminated 4 large-exception and 120 nested attempts before source generation/session evaluation.
3. Actual subprocess/oracle executions saved: **0** on both benchmarks. In-memory memoization had already absorbed the duplicate sources.
4. Candidate ordering did not change.
5. Final reduced-source hashes did not change.
6. No optimization relied on monotonicity assumptions.
7. Every standalone reproducer reproduced the target failure.
8. The largest measured gain was nested scheduler-work reduction: 110 fewer scheduler attempts. No optimization produced a real oracle-execution or robust wall-time gain.

### Historical Inductor

The historical Inductor benchmark was not rerun in this pass. The preserved V1/V2 baseline remains 445 candidate requests, 78 fresh executions, and 574.64 seconds of candidate execution. No optimized Inductor claim is made.

### Follow-up evidence status

- **COMPUTATIONALLY VERIFIED:** canonical-state skip counts, unchanged oracle counts, deterministic outputs, fingerprint preservation, and standalone reproductions.
- **OBSERVED:** reduced scheduler work without expensive oracle savings; wall-time result is noisy and below claim threshold.
- **NOT TESTED:** optimized full Inductor reduction.
