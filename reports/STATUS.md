# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` now supports deterministic generic, structured broadcasting, dynamic-shape, and storage-alias/mutation PyTorch Inductor campaigns. Dynamic cases compile one callable with `dynamic=True` and reuse it over a four-shape trace. Alias cases validate storage relationships and compare returned observables plus post-mutation state.

## Confirmed results

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
- **OBSERVED:** all 45 recorded candidate-only failures were `index_fill_` `BackendCompilerFailed` outcomes. They are compiler-support diagnostics, not correctness findings; a repeat run had 42 such failures and also had zero mismatches.
- **COMPUTATIONALLY VERIFIED:** the alias implementation and campaign support passed the complete 92-test suite at commit `fef8634`.

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
- **OBSERVED:** Inductor `index_fill_` lowering failed for some generated alias/view-write graphs across all three tested dtypes. These candidate-only errors require a separate backend-support investigation; they are not semantic mismatches.

## Unresolved bottlenecks

- Nightly PyTorch validation was not run because no float32/float64 dynamic-only discrepancy survived controls.
- Guard/recompilation counts are secondary observations and rely on a compatibility wrapper around private backend lookup.
- The dynamic v1 grammar excludes zero-sized dimensions, mutation, explicit aliasing, and view/write interactions.
- Alias/mutation v1 is forward-only and excludes zero-sized dimensions, autograd aliasing, and nightly validation.
- The alias campaign has no minimized semantic finding because no semantic mismatch survived; `index_fill_` compiler-only failures remain untriaged beyond classification.

## Next highest-value experiment

Extend alias/mutation coverage to zero-sized dimensions, autograd, and explicit view/write interactions in a separate milestone. Retain the eager/static/dynamic validation funnel and do not treat compiler-only failures or graph-count changes as correctness defects.
