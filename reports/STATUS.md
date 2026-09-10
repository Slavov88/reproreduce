# ReproReduce status

## Current frontier

**RESOURCE_BLOCKED:** The requested real historical Inductor Dependency V2 validation was stopped before launch. Windows available RAM was 0.91 GiB, and the compatible WSL environment lacked PyTorch/Triton. No compiler candidate was accepted and no resource failure was counted as target preservation. See `reports/INDUCTOR_DEPENDENCY_REDUCTION_V2.md`.

**COMPUTATIONALLY VERIFIED:** Dependency-aware reduction V2 is implemented on `feat/dependency-reduction-v2` at `3913c4e`. It is opt-in as `strategy="dependency_v2"`, leaving standard mode and V1 dependency mode intact. Nested Python improved from V1's 200 -> 47 to 200 -> 5 nonblank LOC, with identical jobs=1/jobs=4 output hashes and fewer fresh oracle runs (135 vs 302). Large exception and generated PyTorch both reached 3 LOC in repeated serial runs. Historical Inductor V2 remains **NOT CHECKED** because only approximately 1.82 GiB RAM was available.

**COMPUTATIONALLY VERIFIED:** Dependency-aware reduction V1 is implemented on `feat/dependency-aware-reduction` at `a17127611bd366b8ceb3ba8ce69963b3f24b24eb`. It is opt-in (`strategy="dependency"`) and oracle-backed. On the nested benchmark it reduced 200 -> 47 nonblank LOC versus the standard 200 -> 55 baseline, with identical output across jobs 1 and 4. On the large exception regression it preserved 273 -> 4 in two repeats. See `reports/DEPENDENCY_REDUCTION_V1.md`.

**OBSERVED:** The nested quality improvement costs additional oracle work: 302 serial executions versus the standard 212 baseline. Dependency jobs=4 reduced one recorded wall time from 177.58s to 117.33s, but compiler/GPU contention remains **NOT CHECKED**.

**OBSERVED:** `feat/reducer-search-speed` adds search traces and generic exact-source scheduler deduplication. It is a performance milestone, not a new bug-hunting campaign.

**COMPUTATIONALLY VERIFIED:** pass 2 reduced evaluation-layer requests from 186 -> 168 for large exception and 390 -> 213 for nested Python, while fresh oracle executions remained 167 and 212 respectively. Reduced-source hashes, fingerprints, standalone reproducers, and first-occurrence oracle order were unchanged.

**OBSERVED:** pass 2 produced no wall-time speedup; measured medians were noisy and unfavorable. The eliminated requests were already answered by v1 memoization, so expensive oracle executions did not decrease.

**COMPUTATIONALLY VERIFIED:** canonical whole-tree AST-state deduplication removed 4 large-exception and 120 nested pre-unparse attempts, but saved **0 oracle executions**. Final reductions and hashes were unchanged; no robust wall-time claim is made. See the addendum in `reports/REDUCTION_SPEED_V2.md`.

**COMPUTATIONALLY VERIFIED:** the optimization preserved reduced-source hashes, reduced LOC, failure fingerprints, standalone repro success, and deterministic candidate counts on three-repeat large-exception and nested-Python comparisons. It produced a 1.056x nested median speedup but a 0.965x large-exception median, so no broad speedup claim is made.

**OBSERVED:** isolated candidate execution dominates the measured Python benchmark wall time at approximately 98.8% in representative runs. The historical Inductor record attributes 99.4% of wall time to candidate execution; no full optimized Inductor reduction was attempted because the cold baseline exceeded 40 minutes.

The v1 large-input benchmark suite remains the evidence base: synthetic exceptions, a historical Inductor compiler failure, nested Python structure, and generated PyTorch-heavy syntax. It is a reduction benchmark, not a new bug-hunting campaign.

**COMPUTATIONALLY VERIFIED:** the completed suite reduced 273 -> 4, 152 -> 28, 200 -> 55, and 331 -> 4 nonblank LOC respectively, with configured fingerprints preserved and fresh exported repro verification for all completed measurements. See `reports/LARGE_REDUCTION_BENCHMARKS_V1.md` and `reports/large_benchmarks_v1.json`.

**OBSERVED:** the historical Inductor benchmark required a warm-cache retry after a cold-start 40-minute timeout. Nested reduction remains limited by dependent definitions and unsimplified expressions/control flow.

The earlier hunt infrastructure remains available: deterministic generic, structured broadcasting, dynamic-shape, storage-alias/mutation, and compiler-failure triage workflows. Dynamic cases compile one callable with `dynamic=True` and reuse it over a four-shape trace. Alias cases validate storage relationships and compare returned observables plus post-mutation state.

## Confirmed results

- **COMPUTATIONALLY VERIFIED:** Dependency V1 preserved `REPROREDUCE_NESTED_TARGET` and `REPROREDUCE_LARGE_TARGET` in fresh standalone subprocesses. Nested output hash: `fed7647cc51583d7d4ac7429bc2c71efd3feb129b70041de8e6ac245bd13a214`; large-exception output hash remains `776c1b47de79a096437e1b9296cda2bbc7f9ea64f0a993170f1bf7049d26150e`.
- **COMPUTATIONALLY VERIFIED:** The full local suite passed **119 tests and 380 subtests** after the dependency implementation.
- **OBSERVED:** V1's static analysis is proposal/order machinery only; all accepted candidates passed the normal failure oracle. No minimality or universal semantic-safety claim is made.

- **COMPUTATIONALLY VERIFIED:** the pre-change baseline passed **78 tests** at commit `520d1cf`; the final dynamic-shape implementation passed **87 tests**.
- **COMPUTATIONALLY VERIFIED:** the dynamic-shape campaign completed **400 cases** and **1,600 shape executions**: 200 forward cases and 200 gradient cases.
- **COMPUTATIONALLY VERIFIED:** all dynamic cases executed eagerly and completed Inductor execution with zero compile, runtime, infrastructure, or timeout failures.
- **COMPUTATIONALLY VERIFIED:** forward produced 26 numerical discrepancies; gradient produced 8 numerical and 9 nonfinite comparisons. All were bfloat16.
- **COMPUTATIONALLY VERIFIED:** all forward/gradient numerical discrepancies reproduced in five campaign confirmation attempts. The nine nonfinite cases were separately reproduced five times in a fresh process.
- **COMPUTATIONALLY VERIFIED:** static controls matched every forward failing shape and every gradient numerical failing shape. Three nonfinite cases were dynamic-only at their first shape but remained bfloat16-only.
- **COMPUTATIONALLY VERIFIED:** representative operation families and all three dynamic-only nonfinite representatives passed float32 and float64 dynamic and static controls.
- **COMPUTATIONALLY VERIFIED:** representative forward and gradient cases were reduced to 6 LOC with minimized two-shape traces and exported dynamic harnesses.
- **INFERRED / FALSE_POSITIVE:** no float32/float64 dynamic-only semantic discrepancy survived the validation funnel. No new PyTorch correctness defect is claimed.
- **COMPUTATIONALLY VERIFIED:** the alias/mutation campaign completed 400 forward cases (seeds `5000–5399`) with 400 eager-valid cases, 355 completed comparisons, zero semantic mismatches, and 45 candidate-only compiled runtime failures.
- **COMPUTATIONALLY VERIFIED:** alias coverage included eight patterns at 50 cases each, eight mutation families, float32/float64/bfloat16 controls, and contiguous/transpose/slice layouts. All eager alias relationships were valid.
- **COMPUTATIONALLY VERIFIED:** the 45 original alias candidate-only records cluster into 42 `index_fill_` backend-wrapped assertions and 3 harness-timeout artifacts. The three timeout seeds pass fresh rechecks with an extended budget.
- **COMPUTATIONALLY VERIFIED:** the 42-case index-fill cluster minimizes to a 13-nonblank-line legal reproducer: `index_fill` on a non-contiguous transpose/slice view. Direct-base and contiguous-view controls pass; transpose/slice-view controls fail across float32, float64, and bfloat16 on stable Inductor.
- **COMPUTATIONALLY VERIFIED:** the minimal stable reproducer passes on the tested current nightly `2.15.0.dev20260907+cpu` in five fresh processes and across the 15-cell control matrix.
- **KNOWN_EXISTING_ISSUE / FIXED_IN_NIGHTLY:** the cluster matches PyTorch issue #178952. No current or novel PyTorch defect is claimed.
- **COMPUTATIONALLY VERIFIED:** the alias implementation and compiler-failure triage support passed the complete 99-test suite after the triage changes.

## Alias/mutation campaign record

See `reports/INDUCTOR_ALIAS_MUTATION_V1.md`.

Environment: WSL2 Ubuntu, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0, NVIDIA GeForce RTX 3050 Laptop GPU. Inputs were CPU tensors; the registered Inductor backend was invoked. Raw records remain in ignored `.hunt/inductor-alias-mutation-v1/`.

- Implementation commit: `fef8634a11287c0b4d075c147ea2fbde6343af8b`.
- Forward seeds: **5000–5399**.
- Eight alias patterns: 50 cases each.
- Semantic mismatches: **0**.
- Candidate-only compiled runtime failures: **45** in the recorded run; **42** in the repeat run.

## Dynamic campaign record

See `reports/INDUCTOR_DYNAMIC_V1.md`.

Environment: WSL2 Ubuntu, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0, NVIDIA GeForce RTX 3050 Laptop GPU. Inputs were CPU tensors; the registered Inductor backend was invoked. Case timeout was disabled.

- Forward seeds: **3000–3199**.
- Gradient seeds: **4000–4199**.
- Four shape instances per case; zero-sized dimensions excluded.
- Backend graph observations: 280 forward / 275 gradient graph compilations; 80 / 75 cases respectively observed more than one graph.
- Raw records and validation artifacts: ignored `.hunt/inductor-dynamic-v1/`.

## Failed directions

- **OBSERVED:** bfloat16 dynamic broadcasts, reductions, and gradient cases produce many reproducible low-precision or nonfinite comparisons, but static controls and higher-precision controls do not support a dynamic-shape defect claim.
- **OBSERVED:** recompilation is common enough to be diagnostically useful, but it is not itself a correctness failure.
- **OBSERVED:** the original 30-second in-process campaign budget wrapped three cold-compilation timeouts as `BackendCompilerFailed`; this was a ReproReduce classification bug and is now fixed. The remaining index-fill failure is an AOTAutograd functional-graph assertion in the Inductor path.

## Unresolved bottlenecks

- Nightly PyTorch validation was not run because no float32/float64 dynamic-only discrepancy survived controls.
- Guard/recompilation counts are secondary observations and rely on a compatibility wrapper around private backend lookup.
- The dynamic v1 grammar excludes zero-sized dimensions, mutation, explicit aliasing, and view/write interactions.
- Alias/mutation v1 is forward-only and excludes zero-sized dimensions, autograd aliasing, and nightly validation.
- The original campaign does not retain full tracebacks; fresh standalone reproductions provide exact AOTAutograd stage evidence.
- CUDA-nightly validation of the minimized reproducer remains untested; current nightly validation used a CPU-only wheel.
- The control matrix excludes empty/negative indices, alternate index dtypes, gradients, dynamic shapes, and GPU tensors.

## Next highest-value experiment

Keep Dependency V1 opt-in and do not redesign the scheduler. The next useful check is a separately resourced, isolated ablation of dependency components versus expression candidates; historical Inductor and GPU parallel runs remain deferred until adequate resource controls are available.
