# Dependency-aware reduction V1

## Status

**COMPUTATIONALLY VERIFIED:** The opt-in `strategy="dependency"` implementation preserved the configured exception fingerprints and produced a deterministic 47-nonblank-line nested reproducer, versus the standard strategy's 55-line result. It also preserved the large-exception 4-line result. This is not a minimality or universal-safety claim.

## Scope and design

The default remains `strategy="standard"`; `jobs=1` remains the default. `v0.1.0` was not modified.

The dependency strategy performs conservative static AST analysis of definitions, uses, direct calls, imports, assignments, and dynamic-Python markers. It proposes:

- unused definition/import/assignment removals;
- same-scope unused-definition dependency components;
- individual unused import aliases; and
- small oracle-validated expression replacements for binary, Boolean, and conditional expressions.

Static analysis only proposes or orders candidates. Every candidate that can be parsed is evaluated by the normal failure oracle, with the same failure-fingerprint comparison as the standard reducer. No monotonicity, semantic-safety, or minimality assumption is made. Parameter and call-argument reduction are deliberately not implemented in V1.

## Pre-run contract

The smallest pilot was run on the existing 55-line nested export before the full fixture. It reached 47 lines while preserving `REPROREDUCE_NESTED_TARGET`. The kill criteria were fingerprint change, standalone reproduction failure, nondeterminism, or extra work without a useful quality result. The implementation was then tested on the original nested fixture and the large exception regression fixture.

## Results

| Benchmark | Strategy | Jobs | Original LOC | Reduced LOC | Wall time | Oracle executions | Output SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Nested Python | standard baseline (V3 median) | 1 | 200 | 55 | 123.16s | 212 | `bedc12d08f47bd486d579f002cc879c730c18e641271333be5c332663cf7c39c` |
| Nested Python | dependency V1 repeat 2 | 1 | 200 | 47 | 177.58s | 302 | `fed7647cc51583d7d4ac7429bc2c71efd3feb129b70041de8e6ac245bd13a214` |
| Nested Python | dependency V1 repeat 2 | 4 | 200 | 47 | 117.33s | 316 | `fed7647cc51583d7d4ac7429bc2c71efd3feb129b70041de8e6ac245bd13a214` |
| Large exception | standard baseline (V3 median) | 1 | 273 | 4 | 100.30s | 167 | `776c1b47de79a096437e1b9296cda2bbc7f9ea64f0a993170f1bf7049d26150e` |
| Large exception | dependency V1 repeat 1 | 1 | 273 | 4 | 68.10s | 118 | `776c1b47de79a096437e1b9296cda2bbc7f9ea64f0a993170f1bf7049d26150e` |
| Large exception | dependency V1 repeat 2 | 1 | 273 | 4 | 69.01s | 118 | `776c1b47de79a096437e1b9296cda2bbc7f9ea64f0a993170f1bf7049d26150e` |

The standard baselines are the three-repeat measurements in `reports/REDUCTION_SPEED_V3.md`; dependency records include commit, command, environment, configuration, metrics, and raw reduction metadata in the JSON files alongside this report.

## Dependency candidate accounting

For the nested serial repeat, V1 proposed 33 dependency candidates: 3 dependency components, 23 unused assignments, 1 unused import, and 6 expression candidates. It accepted 3 components, 2 assignments, 1 import, and 6 expressions. The final 47-line output preserved the target in a fresh subprocess.

The nested jobs=4 run submitted 323 candidate executions, completed 316, cancelled 7 pending futures, and reached peak concurrency 4. The final source hash was identical to serial. As with the existing parallel reducer, already-running child subprocesses were not forcibly cancelled.

The large-exception repeats proposed 15 dependency candidates and accepted one dependency component, two assignments, and one import. Both runs reproduced the same 4-line output and hash.

## Ablations and controls

- **Standard vs dependency:** standard is the unchanged default path; dependency lowers nested output by 8 nonblank lines but increases serial oracle executions from 212 to 302 on this fixture.
- **Dependency jobs 1 vs 4:** output and hash are unchanged; jobs=4 reduced dependency wall time from 177.58s to 117.33s in the recorded repeats, with additional speculative executions.
- **Large-exception regression:** dependency preserved the standard 4-line result in two independent serial repeats.
- **Full regression suite:** 119 tests passed, including 380 subtests. No multi-minute benchmark was added to CI.
- **Standalone checks:** exported nested and large reproducers exited nonzero and emitted their exact configured target messages in fresh subprocesses.

The component, assignment, import, and expression counts are instrumentation for V1 rather than causal attribution of the 8-line improvement. A fully isolated per-transform ablation remains future work.

## Limitations

Compiler/GPU contention and historical Inductor dependency-aware reduction are **NOT CHECKED**. The implementation does not claim universal ReproReduce quality, planner-like semantic reasoning, or minimality. Dynamic Python is only flagged; static analysis is not treated as complete in the presence of reflection, `eval`, `exec`, or indirect name resolution.

## Reproducibility

Implementation commit: `a17127611bd366b8ceb3ba8ce69963b3f24b24eb`.

Raw records:

- `reports/dependency_v1_original_nested_repeat2_jobs1.json`
- `reports/dependency_v1_original_nested_repeat2_jobs4.json`
- `reports/dependency_v1_large_exception_jobs1.json`
- `reports/dependency_v1_large_exception_jobs1_repeat2.json`

The earlier pilot records are retained separately as `reports/dependency_v1_original_nested.json` and `reports/dependency_v1_original_nested_jobs4.json`; they are not overwritten.
