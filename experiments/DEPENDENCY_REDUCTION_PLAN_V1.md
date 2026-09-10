# Dependency-aware Reduction V1 Plan

## Hypothesis

A conservative AST definition/use graph can prioritize removable dependency islands and a small set of oracle-validated expression simplifications, allowing the nested benchmark to pass the standard reducer's 55 nonblank LOC floor without trusting static analysis as a correctness oracle.

## Mechanism

Static analysis identifies definitions, uses, direct calls, imports, assignments, and dynamic-Python risk. The dependency strategy proposes candidates in deterministic low-risk order. Every accepted source is still evaluated by the configured ReproReduce oracle.

## Strong baselines

- Standard strategy, serial: nested Python 200 -> 55 LOC, 212 oracle executions, source SHA-256 `bedc12d08f47bd486d579f002cc879c730c18e641271333be5c332663cf7c39c`.
- Standard strategy, serial: large exception 273 -> 4 LOC.
- Historical Inductor evidence is preserved at 152 -> 28 LOC and is not part of the initial pilot.

## Smallest discriminating experiment

Run the dependency strategy serially on the reduced nested source and on the original nested fixture, first with definition/use candidate generation only. Verify that candidates are oracle-approved, the output is executable, and the result is no larger than the standard output. Then add expression candidates only if the definition/use pass does not lower the floor.

## Ablation order

A. standard strategy;
B. dependency definition/use ordering;
C. unused definition/import/assignment proposals;
D. dependency component grouping;
E. small expression candidates;
F. dependency strategy with jobs=4 after serial quality is stable.

## Kill criteria

Stop a component if it changes the target fingerprint, breaks standalone reproduction, produces nondeterministic output, or adds substantial oracle work without lowering LOC. Do not claim minimality. Do not infer static semantic safety.

## Resource policy

Self-contained Python fixtures only for V1. No new bug hunt and no parallel Inductor run unless separately authorized after a resource pilot.

## Result labels

Use OBSERVED for measurements, COMPUTATIONALLY VERIFIED for repeated oracle/fingerprint/standalone checks, and NOT CHECKED for deferred compiler/GPU cases.
