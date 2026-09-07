# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` adds a small PyTorch tensor-program search frontend. It generates deterministic short programs and configurations, runs eager versus `torch.compile`, confirms and deduplicates discrepancy candidates, delegates confirmed source to ReproReduce, and exports issue-ready artifacts.

## Confirmed results

- **COMPUTATIONALLY VERIFIED:** 22 hunt-focused tests pass locally.
- **COMPUTATIONALLY VERIFIED:** generated programs in the deterministic test sample execute eagerly, including configured non-contiguous layouts.
- **COMPUTATIONALLY VERIFIED:** controlled forward and gradient discrepancies complete the generate → detect → confirm → deduplicate → reduce → export pipeline.
- **OBSERVED:** a two-case local `aot_eager` smoke campaign produced two `PASS` outcomes and no discrepancy.
- **OBSERVED:** the historical PyTorch #91468 reduction remains the only real correctness benchmark; this branch makes no new-bug discovery claim.

## Failed directions

None in this branch.

## Unresolved bottlenecks

- Default Windows Inductor remains unavailable because MSVC `cl` is not installed.
- Hunt-specific automatic reduction currently receives source through the existing explicit oracle-adapter interface; a fully automatic runtime adapter for every generated configuration is still limited.
- Search throughput and operator/configuration coverage are not yet characterized on Linux Inductor.

## Next highest-value experiment

Run a modest Linux/WSL Inductor campaign, archive seed/configuration/commit/environment/raw counts, and manually triage any reproducible forward or gradient mismatch before increasing the search budget.
