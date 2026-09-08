# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` includes a deterministic PyTorch tensor-program search frontend, explicit backend tracking, checkpointed experiment records, dtype-aware tolerances, discrepancy confirmation, ReproReduce minimization, standalone reproducer export, and structured broadcasting coverage.

## Confirmed results

- **COMPUTATIONALLY VERIFIED:** the full test suite passed **78 tests** on WSL2 Ubuntu.
- **COMPUTATIONALLY VERIFIED:** the targeted broadcasting campaign completed **500 deterministic cases**: 250 forward cases and 250 gradient cases.
- **COMPUTATIONALLY VERIFIED:** all 500 cases executed eagerly and completed Inductor execution; no compile failures, runtime failures, infrastructure failures, or timeouts were recorded.
- **COMPUTATIONALLY VERIFIED:** forward cases covered six broadcasting patterns, three dtypes, six post-operation cells, and contiguous, sliced, and transposed inputs. Gradient cases used the same structured dimensions.
- **COMPUTATIONALLY VERIFIED:** the forward half produced 4 persistent discrepancies; the gradient half produced 17 persistent discrepancies. Every candidate reproduced in all five confirmation attempts.
- **OBSERVED:** the 4 forward candidates were three bfloat16 rounding cases and one float32 reduction-order case. The 17 gradient candidates were all bfloat16 cases.
- **COMPUTATIONALLY VERIFIED:** fresh-process float32/float64 controls passed for representative bfloat16 forward and gradient candidates; a float64 control also passed for the float32 reduction-order candidate.
- **COMPUTATIONALLY VERIFIED:** representative forward and gradient candidates reduced from 17 LOC to 6 LOC, with failure preservation, and were exported with runnable harnesses under `.hunt/inductor-broadcast-v1/export-*`.
- **INFERRED / FALSE POSITIVE:** the observed candidates are low-precision rounding or legal reduction-order effects, not semantic incorrectness. No new PyTorch defect is claimed.

## Targeted campaign record

Environment: WSL2 Ubuntu, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0, NVIDIA GeForce RTX 3050 Laptop GPU. Inputs were CPU tensors; the registered Inductor backend was invoked. Case timeout was disabled.

- Forward seeds: **1500–1749**.
- Gradient seeds: **2500–2749**.
- Confirmation policy: five repeated runs with fixed generated inputs.
- Raw records: ignored `.hunt/inductor-broadcast-v1/`.
- Coverage reports: `forward.coverage.md` and `gradient.coverage.md`.

## Failed directions

- **OBSERVED:** treating all reproducible bfloat16 discrepancies as compiler defects overcalls expected fusion/reassociation differences.
- **OBSERVED:** per-case Inductor compilation makes broad campaigns slow; the forward campaign was completed in segments after the initial process exceeded the interactive execution budget.

## Unresolved bottlenecks

- Nightly PyTorch validation was not run because every candidate was eliminated by dtype controls and semantic review; no higher-precision semantic discrepancy remained.
- The current grammar does not generate dynamic shapes, mutation, explicit aliasing, or view/write interactions.
- Search throughput remains limited by per-case Inductor compilation.

## Next highest-value experiment

Add a small mutation/view/aliasing pilot with float32 and float64 controls. Kill the direction if it produces only low-precision or expected layout-dependent numerical differences; retain it only if a stable higher-precision semantic discrepancy survives fresh-process validation.
