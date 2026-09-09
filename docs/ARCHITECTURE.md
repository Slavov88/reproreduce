# Architecture

ReproReduce keeps the reduction engine and PyTorch hunting frontend separate but uses the same preservation-oracle idea.

## Reduction pipeline

```text
source program
    ↓
subprocess runner + timeout
    ↓
failure oracle and normalized fingerprint
    ↓
SQLite candidate cache
    ↓
AST / tensor / module transformations
    ↓
repeated validation of the preserved failure
    ↓
ReductionResult and export package
```

`ReductionSession` owns the loop. `RunResult` records isolated execution. An `FailureOracle` decides whether a candidate preserves the baseline. The cache avoids rerunning identical candidates. Reducers are conservative: they accept a transformation only when the oracle and fingerprint still agree.

## PyTorch differential pipeline

```text
structured generator
    ↓
eager and torch.compile execution
    ↓
value / gradient / state comparison
    ↓
classification and failure-stage diagnostics
    ↓
confirmation and fingerprint clustering
    ↓
reduction or standalone export
```

`CompileDifferenceOracle` and `GradientDifferenceOracle` compare eager and compiled outcomes. The hunt frontend adds deterministic tensor programs, dtype/layout controls, confirmation runs, and JSON coverage records. Alias/mutation cases additionally validate storage relationships and post-mutation state.

Compiler-only exceptions are recorded separately from semantic mismatches. `reproreduce summarize` clusters saved campaign records using normalized exception fingerprints; raw records remain the source of truth.

## Public entry points

- `reproreduce reduce ...` reduces a self-contained script.
- `reproreduce hunt ...` runs a structured PyTorch campaign.
- `reproreduce summarize ...` clusters saved campaign failures.
- `from reproreduce import reduce` exposes the core Python API.

The system does not promise global minimality, arbitrary-project reduction, universal PyTorch support, or automatic upstream issue filing.
