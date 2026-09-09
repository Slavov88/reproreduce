# Large-Input Reduction Benchmarks v1

## Objective

**OBSERVED / COMPUTATIONALLY VERIFIED:** measure whether ReproReduce can reduce realistically large failing Python and PyTorch programs to small standalone reproducers while preserving the configured failure fingerprint.

This is a reduction benchmark, not a new PyTorch hunt campaign. The fixtures have known causal failures so the measurement isolates reduction behavior.

## Provenance and environment

- Code commit: `1aacfb9796830a23fd3edd875955958aac86a95c`
- Baseline release commit: `e194923c053206f181613ed381259f8f536d3219`
- Branch: `feat/large-reduction-benchmarks`
- Host: Windows 10, Python 3.10.11
- PyTorch: `2.5.1+cu121`, CUDA available
- Command: `python -m reproreduce.bench`
- Candidate timeout: 10 seconds for Python benchmarks and 30 seconds for PyTorch benchmarks
- Baseline full suite: 102 tests passed in 1,162 seconds at `e194923`
- Final full suite: 105 tests passed in 1,200 seconds at `1aacfb9`

The historical Inductor fixture was freshly executed in this environment. The exact PyTorch `2.5.1+cu124` evidence for issue #178952 remains in the earlier triage report; this benchmark run used the installed `2.5.1+cu121` wheel.

## Benchmark definitions

| Benchmark | Classification | Failure | Input structure |
|---|---|---|---|
| `large_exception` | SYNTHETIC | `RuntimeError("REPROREDUCE_LARGE_TARGET")` | 24 varied helper stages, a dataclass ledger, transformations, diagnostics, and a buried target path |
| `large_inductor_index_fill` | HISTORICAL_REAL_BUG | Inductor `AssertionError` containing `n=copy_` | PyTorch preprocessing, modules, metadata, tensor branches, and the known non-contiguous-view `index_fill_` core |
| `nested_python` | SYNTHETIC | `RuntimeError("REPROREDUCE_NESTED_TARGET")` | Nested function, `if`, loop, `try/except`, dataclass records, and dependent helper calls |
| `generated_pytorch` | GENERATED_REALISTIC | `RuntimeError("REPROREDUCE_GENERATED_TORCH_TARGET")` | 28 tensor helpers, 12 `nn.Module` classes, a generated model, batch construction, and statistics |

The generated Python and PyTorch fixtures are reproducible from `benchmarks/large_reduction_v1/generate.py`. The checked-in generated sources were parsed and executed before reduction.

## Results table

The primary size metric is nonblank LOC. Physical LOC is also reported.

| Benchmark | Type | Original LOC | Reduced LOC | Reduction % | Candidate Runs | Wall Time | Fingerprint Preserved | Standalone Repro |
|---|---|---:|---:|---:|---:|---:|---|---|
| Large exception | SYNTHETIC | 273 (308 physical) | 4 (4 physical) | 98.5% | 167 | 189.66–219.55 s | yes | yes |
| Large Inductor `index_fill_` | HISTORICAL_REAL_BUG | 152 (228 physical) | 28 (33 physical) | 81.6% | 78 fresh / 445 evaluations | 577.84 s warm-cache retry | yes, normalized | yes |
| Nested Python | SYNTHETIC | 200 (289 physical) | 55 (65 physical) | 72.5% | 212 | 216.36–220.11 s | yes | yes |
| Generated PyTorch | GENERATED_REALISTIC | 331 (392 physical) | 4 (4 physical) | 98.8% | 154 | 889.91 s | yes | yes |

The table reports one completed measurement for the PyTorch cases. The exception and nested benchmarks each have two completed repeats; their reduced source hashes and candidate counts matched exactly.

## Large exception result

**COMPUTATIONALLY VERIFIED:** `273 -> 4` nonblank LOC, or 98.5% reduction. Both repeats produced the same reduced-source SHA-256 and the same 167 fresh candidate runs. The reduced and exported scripts raised `RuntimeError` with the exact configured `REPROREDUCE_LARGE_TARGET` message.

This is strong evidence for large synthetic Python reduction. It is not evidence about arbitrary multi-file applications.

## Large PyTorch / Inductor result

**COMPUTATIONALLY VERIFIED:** the 152-nonblank-line wrapper reduced to 28 nonblank lines while retaining the historical Inductor failure. The exported `repro.py` was executed in a fresh subprocess and reproduced an `AssertionError` whose compiler message contained `n=copy_`.

The raw Inductor graph text changed placeholder ordering between some executions. Therefore exact full-message equality was not used as the benchmark identity. The recorded normalized fingerprint is:

```text
AssertionError + target n=copy_ + stage _call_user_compiler
```

The raw before/after fingerprints remain in `reports/large_benchmarks_v1.json`. This is a known historical failure corresponding to PyTorch issue #178952, not a new defect claim.

The first cold-start reduction attempt reached the externally imposed 40-minute budget with 298 committed cache rows and did not terminate normally. A warm-cache retry completed in 577.84 seconds. The timeout is retained as evidence of the real Inductor reduction cost rather than being hidden.

## Nested result

**COMPUTATIONALLY VERIFIED:** `200 -> 55` nonblank LOC, or 72.5% reduction. The two repeats matched exactly.

This is a useful limitation result rather than a failure of correctness. The existing v0.1 reducer already traverses nested statement lists recursively, but it does not perform general expression/control-flow simplification. The reduced source consequently retains a substantial dependent path: the `Record` dataclass, record construction, normalization, scoring, formatting, context construction, and the nested `try`/loop/branch path.

No reducer improvement was made before or after this measurement. The benchmark therefore records current v0.1-era behavior rather than an improved result.

## Generated PyTorch result

**COMPUTATIONALLY VERIFIED:** `331 -> 4` nonblank LOC, or 98.8% reduction. The PyTorch-heavy source was generated deterministically and executed with PyTorch `2.5.1+cu121`. The target was a configured exception after the model path, so the result measures removal of large PyTorch syntax and helper definitions rather than compiler reduction.

The single run took 889.91 seconds. A second run was not performed because the first reduction was already approximately 15 minutes; this is explicitly not a repeated determinism result.

## Reduction cost

| Benchmark | Candidate evaluations | Cache hits | Hit ratio | Candidate execution time | Notes |
|---|---:|---:|---:|---:|---|
| Large exception | 186 | 19 | 10.2% | 215.67 s (repeat 1) | Two repeats; candidate counts matched |
| Large Inductor | 445 | 367 | 82.5% | 574.64 s | Warm-cache retry; cold start timed out at 40 minutes |
| Nested Python | 390 | 178 | 45.6% | 208.33 s (repeat 1) | Two repeats; candidate counts matched |
| Generated PyTorch | 173 | 19 | 11.0% | 887.46 s | One run |

`candidate_runs` counts fresh subprocess executions. `candidate evaluations` includes fresh executions plus cache hits. The full machine-readable records contain both values, median candidate duration, accepted/rejected transformations, hashes, timeout state, and stderr tails.

## Fingerprint preservation

- **Large exception:** exact exception type and message preserved.
- **Nested Python:** exact exception type and message preserved; relevant call frames remained stable.
- **Generated PyTorch:** exact exception type and message preserved.
- **Large Inductor:** normalized compiler identity preserved: `AssertionError`, `_call_user_compiler`, and `n=copy_`. Full raw graph messages differed in placeholder ordering, so exact raw-message equality would be an overly strict and unstable identity for this compiler failure.

A candidate that merely exited nonzero was not accepted as successful. Each benchmark used the configured exception oracle, and each exported script was run in a fresh subprocess.

## Standalone repro verification

All four completed benchmark reductions exported a package containing `repro.py`, `README.md`, `environment.json`, and `reduction.json`. All four exported scripts reproduced their configured failure in a fresh subprocess. The exported Inductor script reproduced the normalized historical compiler failure under the available PyTorch installation.

## Determinism

**COMPUTATIONALLY VERIFIED:**

- `large_exception`: two repeats, reduced LOC 4/4, candidate runs 167/167, identical reduced-source hash.
- `nested_python`: two repeats, reduced LOC 55/55, candidate runs 212/212, identical reduced-source hash.
- `large_inductor_index_fill`: one completed warm-cache measurement; no repeat claim.
- `generated_pytorch`: one measurement; no repeat claim.

## Limitations discovered

1. Large exception reduction is effective but expensive: about 190–220 seconds for 273 nonblank LOC.
2. PyTorch process startup and compiler work dominate cost. The 152-line historical wrapper required a cold-start timeout and a warm-cache retry.
3. Recursive statement reduction is not general semantic simplification. The nested benchmark stalled at 55 nonblank LOC because dependent definitions, calls, and control-flow expressions remain necessary.
4. Compiler error text can contain unstable graph details such as placeholder ordering. A benchmark fingerprint must identify stable exception/stage/target components rather than blindly compare the entire traceback message.
5. Generated PyTorch reduction to four lines is a synthetic exception result, not evidence that arbitrary PyTorch compiler graphs reduce to four lines.
6. These results do not establish global minimality, arbitrary-project support, universal `torch.compile` support, or performance beyond the measured Windows/Python/PyTorch environment.

## Optional reducer improvement

None. The baseline already included recursive nested statement-list traversal. The nested benchmark was measured without modifying the reducer, and its 200-to-55 result is retained as the current limitation baseline. No second-phase reducer change was justified within this milestone.

## Reproduction

Run a selected benchmark:

```bash
python -m reproreduce.bench --benchmark large_exception --repeats 2
```

Run the suite (the PyTorch cases can take many minutes):

```bash
python -m reproreduce.bench --repeats 1 --timeout 30
```

The runner writes ignored exports and SQLite caches below `.benchmarks/`. The compact structured record is `reports/large_benchmarks_v1.json`.

## Evidence status

- **COMPUTATIONALLY VERIFIED:** completed reductions, fingerprint checks, exported repro execution, and repeated deterministic results stated above.
- **OBSERVED:** the cold-start Inductor timeout and raw compiler-message placeholder-order variation.
- **NOT TESTED:** CUDA-nightly validation, multi-file project reduction, arbitrary generated programs, and a second generated-PyTorch repeat.
