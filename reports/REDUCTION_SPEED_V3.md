# Reduction Speed V3: Bounded Parallel Candidate Evaluation

## STATUS

**COMPUTATIONALLY VERIFIED:** opt-in bounded parallel candidate evaluation preserves deterministic final reductions while reducing wall time on the two Python benchmarks. The default remains serial (`jobs=1`).

## PR #6 / BRANCH SETUP

PR #6 was fully green and mergeable and was merged normally into `main` at `a40ed4c`. This work is on `feat/parallel-reduction`; v0.1.0 was not modified.

## PARALLEL DESIGN

At one ddmin partition/granularity step, complements are generated from the same unchanged parent. For `jobs>1`, their sources are rendered and locally compiled, then evaluated through a bounded `ThreadPoolExecutor`. Threads only orchestrate independent `run_python` calls; candidate programs remain fresh subprocesses.

After all candidates before the first serial-order success have resolved, the earliest successful candidate in original order is selected. Pending later futures are cancelled where possible. Results are committed to the session cache serially after collection. Recursive scopes and non-AST PyTorch transformations remain serial search transitions.

## DETERMINISM MODEL

Parallel completion order never determines acceptance. Candidates are examined in serial order, and the first preserving candidate is selected. `jobs=1` retains the pre-parallel ddmin path. The six benchmark runs at each job count produced the same final source hash as serial.

## IMPLEMENTATION

Added:

- `reduce(..., jobs=1)` and CLI `--jobs N`;
- ddmin batch evaluation for same-parent candidate sets;
- bounded subprocess orchestration with per-candidate timeout preservation;
- central/serial cache commits;
- cancellation and speculative-work metrics;
- deterministic batch-order tests and job validation.

No persistent workers, monotonicity assumptions, oracle semantic changes, or automatic CPU-count selection were added.

## BASELINE

Fresh three-repeat measurements were collected for `jobs=1`, `jobs=2`, and `jobs=4` on Windows 10, Python 3.10.11, PyTorch 2.5.1+cu121. The benchmark commands used timeout 10 seconds and isolated workspaces. Raw records are in `reports/parallel_v3_*.json`. The records identify the clean-base commit `a40ed4c`; they were run from the corresponding worktree with the implementation changes present before being committed as `d820e75`. No benchmark-relevant code changed between execution and that commit.

## LARGE EXCEPTION

Input: 273 nonblank LOC. Serial median: 100.30 seconds and 167 oracle executions. The best measured setting was `jobs=4`: 59.81 seconds, 1.678x speedup, with 208 median oracle executions. The additional executions were speculative; 167 were useful.

## NESTED PYTHON

Input: 200 nonblank LOC. Serial median: 123.16 seconds and 212 oracle executions. The best measured setting was `jobs=4`: 67.24 seconds, 1.832x speedup, with 229 median oracle executions. The additional executions were speculative; 212 were useful.

## INDUCTOR

Not run. Resource inspection reported 16 logical CPUs, 15.4 GiB total RAM, only 3.2 GiB available RAM, and 46.4 GiB free disk. Given the preserved historical 577.84-second baseline and known compiler resource cost, no parallel compiler run was justified in this milestone.

## RESOURCE CONTENTION

Python workloads benefited from parallel subprocesses. No compiler workload was run, so compiler contention is **NOT CHECKED**, not inferred safe. GPU/memory-sensitive workloads should remain at `jobs=1` unless the caller validates resource headroom.

## SPECULATIVE WORK

Parallelism increased submitted work because later candidates were launched before an earlier candidate's success was known. Median additional completed executions relative to serial:

- Large exception: +25 at `jobs=2`, +41 at `jobs=4`.
- Nested Python: +12 at `jobs=2`, +17 at `jobs=4`.

These are reported explicitly and are not hidden as cache hits.

## CANCELLATIONS

Pending futures were cancelled when possible. Median cancellations:

- Large exception: 34 at `jobs=2`, 19 at `jobs=4`.
- Nested Python: 9 at `jobs=2`, 4 at `jobs=4`.

Already-running subprocesses were allowed to finish under their own timeout; the implementation does not claim cancellation of running child processes.

## RESULTS TABLE

| Benchmark | Jobs | Median Time | Speedup | Oracle Executions | Speculative | Reduced LOC |
|---|---:|---:|---:|---:|---:|---:|
| Large exception | 1 | 100.30s | 1.000x | 167 | 0 | 4 |
| Large exception | 2 | 73.05s | 1.373x | 192 | 25 | 4 |
| Large exception | 4 | 59.81s | 1.678x | 208 | 41 | 4 |
| Nested Python | 1 | 123.16s | 1.000x | 212 | 0 | 55 |
| Nested Python | 2 | 79.67s | 1.546x | 224 | 12 | 55 |
| Nested Python | 4 | 67.24s | 1.832x | 229 | 17 | 55 |

Oracle execution counts vary slightly between parallel repeats because cancellation races determine which already-running futures finish. Final source output remains deterministic.

## BEST CONCURRENCY

`jobs=4` produced the best median wall time for both benchmarks. Parallel efficiency was 41.9% for large exception and 45.8% for nested Python.

## REDUCTION QUALITY

All 18 benchmark runs preserved the expected reductions:

- large exception: 273 → 4 nonblank LOC;
- nested Python: 200 → 55 nonblank LOC.

All runs exported successfully and passed standalone fresh-subprocess reproduction.

## SOURCE HASHES

- Large exception: `776c1b47de79a096437e1b9296cda2bbc7f9ea64f0a993170f1bf7049d26150e`
- Nested Python: `bedc12d08f47bd486d579f002cc879c730c18e641271333be5c332663cf7c39c`

Hashes were identical across jobs 1, 2, and 4.

## FINGERPRINTS

Preserved in every run:

- `REPROREDUCE_LARGE_TARGET`
- `REPROREDUCE_NESTED_TARGET`

## STANDALONE REPRODUCERS

All 18 exported standalone reproducers reproduced their configured target failure.

## TESTS

Full local suite: **114 tests passed**.

Focused tests cover jobs validation, CLI exposure, ordered batch results, cancellation accounting, serial execution, ddmin batch selection, and existing reduction behavior. No multi-minute benchmark was added to CI.

## COMMITS

- `d820e75` — bounded parallel candidate evaluation and tests
- `a40ed4c` — merged PR #6 baseline

## FILES / RECORDS

- `reports/REDUCTION_SPEED_V3.md`
- `reports/reduction_speed_v3.json`
- `reports/parallel_v3_large_jobs1.json`
- `reports/parallel_v3_large_jobs2.json`
- `reports/parallel_v3_large_jobs4.json`
- `reports/parallel_v3_nested_jobs1.json`
- `reports/parallel_v3_nested_jobs2.json`
- `reports/parallel_v3_nested_jobs4.json`

## PR / BRANCH

- Branch: `feat/parallel-reduction`
- PR: to be opened after pushing this branch

## README CLAIM

No README performance claim was added yet. The measured speedups are substantial, but parallelism increases speculative oracle executions and has not been validated on compiler workloads.

## LIMITATIONS

1. Historical Inductor jobs=2 was not run because available memory was low relative to compiler risk.
2. Running subprocesses cannot be forcibly cancelled by the orchestration layer; only pending futures are reliably cancelled.
3. Parallel oracle execution may be unsafe for GPU- or memory-heavy workloads; `jobs=1` remains the conservative choice.
4. Some parallel execution counts vary with cancellation timing, although final source selection remains deterministic.

## FINAL VERDICT

**KEEP AS OPT-IN.** Bounded parallelism gives robust Python benchmark wall-time reductions (1.373–1.832x medians) while preserving source hashes, fingerprints, standalone reproducers, and subprocess isolation. It should not become the default or be advertised as safe for compiler/GPU workloads without separate validation.

## NEXT STEP

Open the PR with the opt-in implementation and report. Do not start dependency-aware reduction or a parallel Inductor campaign in this milestone.
