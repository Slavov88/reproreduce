# Changelog

## 0.2.0

### Added

- Added opt-in dependency-aware reduction strategies, including the strongest current `dependency_v3` plumbing strategy.
- Added bounded parallel candidate evaluation with `--jobs` and expanded large-input benchmark coverage.
- Added richer reduction metrics, normalized compiler-failure fingerprints, and coordinated parameter/call reduction.

### Improved

- Improved nested and control-flow reduction with call-result, assignment, wrapper, main-guard, and literal/configuration simplification.
- Added cost-aware budgeting for expression search with expensive oracles.
- Preserved serial-by-default behavior and subprocess isolation for failure validation.

### Verified results

- 273 → 1 nonblank LOC — large synthetic Python exception.
- 200 → 1 nonblank LOC — nested Python failure.
- 331 → 1 nonblank LOC — generated realistic PyTorch failure.
- 152 → 12 nonblank LOC — historical real PyTorch Inductor failure, preserving `AssertionError + _call_user_compiler + n=copy_` and reproducing 5/5 in fresh processes.

These are computationally verified reductions, not formal minimality claims.

## 0.1.0

ReproReduce is prepared as an installable v0.1 developer tool.

- Documented installation, exception reduction, PyTorch/Inductor workflows, failure statuses, and limitations.
- Added a tracked historical `index_fill_`/non-contiguous-view fixture with a reduction walkthrough.
- Added failure-stage diagnostics, normalized compiler-failure fingerprints, and `reproreduce summarize`.
- Added concise architecture and release verification documentation.
- Added package version reporting and CLI smoke coverage.
- Preserved the verified alias/mutation, dynamic-shape, broadcast, and historical benchmark evidence.

This is not a PyPI release and no Git tag is created by this change.
