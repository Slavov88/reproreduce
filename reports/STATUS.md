# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` now includes a deterministic PyTorch tensor-program search frontend, explicit Inductor preflight, checkpointed experiment records, dtype-aware tolerances, discrepancy confirmation, ReproReduce minimization, and standalone reproducer export.

## Confirmed results

- **COMPUTATIONALLY VERIFIED:** the final Linux/WSL Inductor preflight invoked the registered Inductor backend and produced matching output.
- **COMPUTATIONALLY VERIFIED:** the final full suite passed **72 tests**.
- **COMPUTATIONALLY VERIFIED:** a total of **1,000 deterministic cases** completed: 500 forward cases and 500 gradient cases.
- **COMPUTATIONALLY VERIFIED:** all 1,000 cases executed eagerly; 999 completed compiled execution; one cache-related infrastructure failure was isolated and excluded from correctness counts.
- **COMPUTATIONALLY VERIFIED:** the campaign produced two persistent forward discrepancies and one persistent gradient discrepancy under the initial dtype-aware oracle. All three were automatically reduced and exported.
- **MANUALLY REPRODUCED:** reduced forward discrepancies were reproduced three times on stable PyTorch 2.5.1+cu124.
- **MANUALLY REPRODUCED:** the reduced gradient discrepancy was reproduced three times on stable PyTorch 2.5.1+cu124.
- **INFERRED / FALSE POSITIVE:** all three discrepancies are explained by low-precision bfloat16 rounding and compiler fusion/reassociation. Float32 and float64 variants match; higher-precision reference checks show the compiled forward result can match the single-rounding reference more closely than eager execution.
- **COMPUTATIONALLY VERIFIED:** the reduced gradient expression passes float64 `gradcheck`.
- **OBSERVED:** no new PyTorch defect is claimed.

## Main campaign record

Environment: WSL2 Ubuntu, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0, NVIDIA GeForce RTX 3050 Laptop GPU. Inputs were CPU tensors; Inductor and Triton were available and the C++ toolchain was installed.

- Forward seeds: 1000–1499.
- Gradient seeds: 2000–2499.
- Confirmation policy: five repeated runs with fixed generated inputs.
- Case timeout: disabled.
- Raw records: ignored `.hunt/inductor-main/`.

## Failed directions

- **OBSERVED:** the first gradient attempt used a temporary 120-second case budget and was interrupted after 166 cases. Its timeout-contaminated case was rerun without a budget and passed; the final 500-case gradient record uses the rerun.
- **OBSERVED:** an initial bfloat16 gradient smoke discrepancy was eliminated by dtype-aware tolerances.

## Unresolved bottlenecks

- Nightly PyTorch was not installed because every persistent candidate was eliminated as a low-precision numerical false positive before nightly validation was necessary.
- The current grammar does not generate broadcasting, dynamic shapes, mutation, or explicit aliasing patterns.
- Search throughput is limited by per-case Inductor compilation; the 500-case forward campaign took approximately 74 minutes.

## Next highest-value experiment

Add only structured broadcasting cases to the existing grammar, retain float32/float64 controls alongside bfloat16, and run a smaller targeted Inductor campaign. Broadcasting is the clearest measured coverage gap before adding mutation or dynamic-shape complexity.
