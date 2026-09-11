# Inductor Dependency Reduction V2 — Retry

## STATUS

**STRONG_VERIFIED_IMPROVEMENT:** Dependency V2 reduced the real historical Inductor fixture from 152 to **22 nonblank LOC**, below the previous 28-line result, while preserving the normalized compiler fingerprint. The exported reproducer reproduced the target in **5/5 fresh processes**.

## PR #9 / BRANCH SETUP

PR #9 was merged into `main` at `696af47`. Work was performed on `feat/inductor-dependency-v2-validation`. V0.1.0 and the prior blocked report were preserved.

## ENVIRONMENT

- Windows 10, Python 3.10.11
- PyTorch 2.5.1+cu121; CUDA build 12.1; `torch.compile` and `torch._inductor` importable
- NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB total / 3964 MiB free at resource check
- NVIDIA driver 566.07; `nvcc` unavailable
- 16 logical CPUs; 15.42 GiB RAM total
- 1.86 GiB Windows RAM available at launch; 15.42 GiB swap total / 8.81 GiB free
- 43.39 GiB disk free at resource check
- WSL2: 7.46 GiB total / 6.74 GiB available, 2 GiB swap free, but no PyTorch/Triton installed

The machine was still constrained, so the run was serial-only and no parallel compiler experiment was attempted.

## RESOURCE CHECK

**OBSERVED:** Availability improved from the earlier 0.91 GiB to approximately 1.86 GiB. This permitted one carefully isolated serial experiment, but not jobs=2/4 or a broad rerun. No system OOM, WSL termination, or infrastructure failure occurred during reduction.

## HISTORICAL BUG REPRODUCTION

**COMPUTATIONALLY VERIFIED:** The tracked small historical fixture `examples/inductor_index_fill/bug.py` completed eager execution and then failed under stable Inductor in all three preflight repetitions. Each failure contained:

```text
AssertionError + _call_user_compiler + n=copy_
```

Preflight durations were 7.84s, 7.04s, and 7.05s. The raw repetitions are retained in `reports/inductor_dependency_v2_preflight_current.json`.

## BASELINE

Prior V1 evidence, preserved in `reports/large_benchmarks_v1.json`:

| Metric | Previous Reduction | Dependency V2 |
|---|---:|---:|
| Original LOC | 152 | 152 |
| Reduced LOC | 28 | 22 |
| Reduction % | 81.6% | 85.5% |
| Oracle executions | 78 | 399 |
| Wall time | 577.84s | 2184.73s |
| Fingerprint preserved | Yes | Yes |
| Standalone repro | Yes | Yes |

The improvement is 6 lines and 3.9 percentage points. V2 required 5.12x as many expensive oracle executions and 3.78x the wall time in this run.

## DEPENDENCY V2 CONFIGURATION

- `strategy="dependency_v2"`
- `jobs=1`
- 30-second candidate timeout
- existing SQLite candidate cache
- normalized `BenchmarkOracle` requiring `AssertionError`, `_call_user_compiler`, and `n=copy_`

## PHASE METRICS

| Phase | Candidates | Oracle Runs | Accepted | LOC Removed | Time |
|---|---:|---:|---:|---:|---:|
| V1/dependency cleanup | 104 proposals | interleaved | 16 | interleaved | interleaved |
| parameter_call | 1 | 1 | 1 | 0 | 7.65s |
| call_result | 50 | 49 | 11 | 11 | 278.44s |
| control_flow | 1 | 1 | 1 | 1 | 7.49s |
| expression | 100 | 98 | 28 | 0 | 479.92s |

The V1 cleanup and existing statement phases are interleaved by the fixed-point pipeline, so their individual LOC and elapsed-time contributions are not inferred from aggregate metrics.

## REDUCTION RESULT

**COMPUTATIONALLY VERIFIED:** `152 -> 22` nonblank LOC, 27 physical LOC. The final source hash is:

`4ecfa69e581e9ef88a0b9baa4b155fe31c125e5ffdafa67365dea558bfac77b2`

The target normalized fingerprint was preserved:

```text
AssertionError + _call_user_compiler + n=copy_
```

## OLD 28 LOC VS NEW

The V2 output is 6 nonblank lines smaller and improves the reduction percentage from 81.6% to 85.5%. It is not claimed minimal.

## FINAL SOURCE ANALYSIS

The 22 nonblank lines contain:

| Category | Count | Content |
|---|---:|---|
| Scope/binding requirement | 3 | dataclass import, torch import, dataclass decorator |
| Function signature | 5 | config, input, target, experiment, main declarations |
| Required assignment/setup | 7 | config fields, zero tensor, value, index, eager clone, compiled clone |
| Target core | 1 | transposed view followed by `index_fill_` |
| Call/argument plumbing | 4 | eager target call, compiled call/return, experiment invocation |
| Return plumbing | 1 | target model return |
| Control-flow shell | 1 | `torch.compile(..., backend='inductor')` execution wrapper |
| **Total** | **22** | |

V2 retained the essential transpose/index-fill operation and removed unrelated preprocessing, modules, metadata, auxiliary outputs, and mutation setup. Remaining bloat is primarily the config dataclass and experiment wrapper.

## KNOWN CORE COMPARISON

The known hand-triaged core is 13 nonblank LOC (`.hunt/inductor-alias-mutation-v1/minimal_transpose_index_fill.py`). Automatic V2 is 9 nonblank lines larger.

V2 automatically retained:

- a 2x3 tensor shape;
- a transpose view;
- an index tensor and `index_fill_` operation;
- the Inductor compile path.

It did not remove the remaining configuration and wrapper plumbing as aggressively as the hand-triaged core. No benchmark-specific rewrite was added.

## ORACLE COST

V2 used 399 fresh expensive compiler executions, with median candidate execution time 4.27s. The expression phase alone used 98 oracle executions and 479.92s. The call-result phase removed 11 gross LOC for 49 oracle executions; control-flow removed 1 gross LOC for 1 execution. Gross phase LOC counts are not additive to final net LOC because phases interleave and rerender the source.

## FINGERPRINT PRESERVATION

**YES:** No candidate was accepted for a generic crash. Accepted candidates had to satisfy the normalized `BenchmarkOracle`. Resource OOM, timeout, infrastructure, and unrelated compiler failures were not accepted.

## STANDALONE REPRODUCTION

**COMPUTATIONALLY VERIFIED:** The normal export path was used. The exported 22-line reproducer reproduced the normalized target failure in 5/5 fresh subprocesses, with return code 1 each time.

## FRESH-PROCESS REPEATS

| Repeat | Result | Time |
|---:|---|---:|
| 1 | TARGET_FAILURE | 7.93s |
| 2 | TARGET_FAILURE | 8.12s |
| 3 | TARGET_FAILURE | 11.56s |
| 4 | TARGET_FAILURE | 11.93s |
| 5 | TARGET_FAILURE | 9.99s |

Target failure frequency: **5/5**.

## NIGHTLY CONTROL

**NOT CHECKED:** The WSL environment lacked PyTorch/Triton, and no separate nightly environment was available. Prior evidence indicates the upstream issue is fixed on current nightly.

## PARALLEL COMPILER RESULT

**NOT ATTEMPTED:** Serial reduction was already a 36.4-minute, 399-execution workload under constrained RAM. Parallel compiler evaluation was not justified.

## TESTS

No source changes were made in this retry. The merged V2 baseline remains **122 tests and 380 subtests passing**.

## README UPDATE

Not yet updated in this validation commit. The next documentation commit should update the verified-results table to 273 -> 3, 152 -> 22, 200 -> 5, and 331 -> 3, while retaining historical baseline details in reports.

## COMMITS

- `696af47` — merged PR #9.
- `2d63d81` — validation branch base and prior resource-blocked report.

## FILES / RECORDS

- `reports/INDUCTOR_DEPENDENCY_REDUCTION_V2_RETRY.md`
- `reports/inductor_dependency_reduction_v2_retry.json`
- `reports/inductor_dependency_reduction_v2_run.json`
- `reports/inductor_dependency_reduction_v2_reduced.py`
- `reports/inductor_dependency_v2_preflight_current.json`
- preserved `reports/INDUCTOR_DEPENDENCY_REDUCTION_V2.md`
- preserved `reports/large_benchmarks_v1.json`

## LIMITATIONS

No nightly control, parallel compiler run, formal minimality result, or multi-seed reduction was performed. The automatic output remains 9 lines larger than the known 13-line hand-triaged core.

## FINAL VERDICT

**STRONG_VERIFIED_IMPROVEMENT:** Dependency V2 improved the authentic historical Inductor reduction from 28 to 22 nonblank LOC and reproduced the normalized compiler failure 5/5 times. The gain came at substantial oracle cost.

## NEXT BOTTLENECK

The remaining automatic bloat is wrapper/configuration plumbing, while expression reduction is the dominant cost. A future milestone should first add phase-cost controls or targeted generic wrapper simplification rather than more broad expression candidates.

## NEXT STEP

Update the README and open the validation PR. Keep compiler reduction serial and resource-gated.
