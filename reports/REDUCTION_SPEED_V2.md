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
