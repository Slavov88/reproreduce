# ReproReduce status

## Current frontier

**OBSERVED:** `feat/bug-hunter` adds a small PyTorch tensor-program search frontend. It generates deterministic short programs and configurations, runs eager versus `torch.compile`, confirms and deduplicates discrepancy candidates, delegates confirmed source to ReproReduce, and exports issue-ready artifacts.

## Confirmed results

- **COMPUTATIONALLY VERIFIED:** 24 hunt-focused tests pass locally.
- **COMPUTATIONALLY VERIFIED:** generated programs in the deterministic test sample execute eagerly, including configured non-contiguous layouts.
- **COMPUTATIONALLY VERIFIED:** controlled forward and gradient discrepancies complete the generate → detect → confirm → deduplicate → reduce → export pipeline.
- **OBSERVED:** a 20-case local `aot_eager` forward pilot and a separate 20-case gradient pilot each produced 20 valid eager cases, 20 compiled cases, zero candidate-only errors, zero mismatches, and zero deduplicated findings. These are pipeline pilots, not an Inductor discovery campaign.
- **OBSERVED:** pilot environment was Windows 10, Python 3.10.11, PyTorch 2.5.1+cu121, commit `cbbde29`; raw JSON remains in the ignored `.hunt/` directory.
- **OBSERVED:** the historical PyTorch #91468 reduction remains the only real correctness benchmark; this branch makes no new-bug discovery claim.

## Failed directions

None in this branch.

## Unresolved bottlenecks

- Default Windows Inductor remains unavailable because MSVC `cl` is not installed.
- Hunt-specific automatic reduction currently receives source through the existing explicit oracle-adapter interface; a fully automatic runtime adapter for every generated configuration is still limited.
- Search throughput and operator/configuration coverage are not yet characterized on Linux Inductor.

## Next highest-value experiment

Run a modest Linux/WSL Inductor campaign, archive seed/configuration/commit/environment/raw counts, and manually triage any reproducible forward or gradient mismatch before increasing the search budget.
