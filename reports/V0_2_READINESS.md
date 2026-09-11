# ReproReduce v0.2.0 Release Readiness

## Status

**READY_FOR_V0.2.** Release-branch and merged-main CI pass. PyPI publication is explicitly out of scope.

- Version: `0.2.0`
- Release commit candidate: `e6a990b5f497c986c0844995b7b69c8660522b68`
- PR #11 merge commit: `2c2df44`
- Prior release: `v0.1.0` unchanged

## Scope since v0.1.0

- Opt-in dependency-aware reduction strategies through `dependency_v3`.
- Coordinated parameter/call and call-result reduction.
- Control-flow, wrapper, main-guard, alias, constant, and configuration propagation.
- Bounded parallel candidate evaluation with serial default.
- Richer metrics and normalized compiler-failure fingerprints.
- Large-input benchmark coverage and standalone export evidence.

## Strongest verified evidence

| Case | Original | Reduced | Evidence |
|---|---:|---:|---|
| Large synthetic Python exception | 273 | 1 nonblank LOC | Failure preserved |
| Nested Python | 200 | 1 nonblank LOC | Failure preserved |
| Generated realistic PyTorch | 331 | 1 nonblank LOC | Failure preserved |
| Historical real PyTorch Inductor | 152 | 12 nonblank LOC | Fingerprint preserved; standalone 5/5 |

The historical compiler fingerprint is `AssertionError + _call_user_compiler + n=copy_`. The reduction is computationally verified, not claimed formally minimal. Full evidence is in `reports/PLUMBING_REDUCTION_V3.md` and `reports/inductor_plumbing_v3.json`.

## Verification

- Clean editable installation in a fresh temporary virtual environment: passed.
- `import reproreduce`: passed; version `0.2.0`.
- `reproreduce --version`: passed; `0.2.0`.
- `reproreduce --help`: passed.
- `reproreduce reduce --help`: passed; documents `dependency_v3` and `--jobs`.
- Dependency V3 exception smoke reduction and exported `repro.py`: passed.
- Exported smoke reproducer: passed with the expected target exception.
- Final local suite: **127 passed, 380 subtests passed**.
- `compileall`: passed before release preparation.
- PR #11 required CI: all checks passed across core Python 3.10–3.12, PyTorch, and hunt jobs.
- Merged-main CI: passed on exact commit `e6a990b5f497c986c0844995b7b69c8660522b68` (workflow run `34590422004`).

## Compatibility and defaults

`standard` remains the default. `dependency`, `dependency_v2`, and `dependency_v3` remain opt-in for compatibility and reproducibility. Compiler/GPU-heavy reductions should normally use `jobs=1`; parallelism is not claimed universally safe.

## Limitations

- Supports self-contained scripts, not arbitrary multi-file projects.
- No formal minimality guarantee.
- Aggressive strategies can be expensive.
- Parallel compiler reduction was not validated.
- Reflection-heavy and dynamic Python limits static guidance.
- Historical Inductor evidence is specific to the tested stable environment; current nightly was previously observed to pass the upstream-fixed issue.
- PyPI publication is not performed.

## Release blockers

None identified.
