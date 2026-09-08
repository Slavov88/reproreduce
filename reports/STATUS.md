# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` now supports deterministic generic, structured broadcasting, and dynamic-shape PyTorch Inductor campaigns. Dynamic cases compile one callable with `dynamic=True` and reuse it over a four-shape trace, with static per-shape controls, shape-level metadata, dynamic-aware reduction, and dynamic harness export.

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

## Unresolved bottlenecks

- Nightly PyTorch validation was not run because no float32/float64 dynamic-only discrepancy survived controls.
- Guard/recompilation counts are secondary observations and rely on a compatibility wrapper around private backend lookup.
- The v1 grammar excludes zero-sized dimensions, mutation, explicit aliasing, and view/write interactions.

## Next highest-value experiment

Do not start it in this milestone. The next separate frontier is aliasing + mutation + view/write interactions, with float32/float64 controls and the same eager/static/dynamic validation funnel.
