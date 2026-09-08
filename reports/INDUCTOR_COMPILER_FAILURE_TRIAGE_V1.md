# Inductor compiler-failure triage v1

**Status: COMPUTATIONALLY VERIFIED for the recorded controls. The dominant failure is a known issue that passes on the latest tested nightly; no current novel PyTorch defect is claimed.**

## SOURCE CAMPAIGN

Source: `.hunt/inductor-alias-mutation-v1/campaign-v3.json`.

- Implementation commit: `fef8634a11287c0b4d075c147ea2fbde6343af8b`
- Backend: PyTorch Inductor
- Stable version: PyTorch `2.5.1+cu124`, CUDA `12.4`, Triton `3.1.0`
- Inputs: CPU tensors
- Cases: 400, seeds `5000–5399`
- Original candidate-only records: 45
- Original classification: 45 `COMPILED_RUNTIME_FAILURE`
- Eager-valid cases: 400
- Original semantic mismatches: 0

The original 30-second per-case budget was applied inside the same process as compilation. Three records contain `BackendCompilerFailed` wrappers whose inner message is `HuntTimeout`; they are not stable compiler failures. The original campaign record is preserved unchanged.

## FAILURE CLUSTERS

Failure fingerprints were computed with path, address, generated-node-suffix, placeholder-order, and line-number normalization. The reusable command is:

```bash
reproreduce summarize .hunt/inductor-alias-mutation-v1/campaign-v3.json
```

The processed machine-readable triage record is `.hunt/inductor-alias-mutation-v1/triage-v1.json`. The raw record yields two clusters:

| Cluster | Original cases | Raw fingerprint/cause | Stage from raw record | Final classification |
|---|---:|---|---|---|
| A | 42 | `BackendCompilerFailed` + `AssertionError: n=copy_, n.args[0]=permute...` | `backend_wrapped_unknown` because the campaign did not retain a traceback | AOTAutograd functionalization failure; `KNOWN_EXISTING_ISSUE`, `FIXED_IN_NIGHTLY` |
| B | 3 | `BackendCompilerFailed` wrapping `HuntTimeout: case exceeded 30.0 seconds` | `harness_timeout` | `FALSE_POSITIVE` / harness artifact |

### Cluster A composition

- Mutation: `index_fill_` in all 42 cases.
- Dtypes: bfloat16 21, float64 14, float32 7.
- Layouts: slice 16, contiguous 14, transpose 12.
- Patterns: sibling views 6, overlapping views 6, chained views 6, mutation reduction 6, mutation view transform 6, mutate-view/observe-base 4, sequential mutations 4, mutate-base/observe-view 4.
- The fresh traceback points to `torch/_functorch/_aot_autograd/functional_utils.py`, `assert_functional_graph`, called from Inductor's `compile_fx`/AOTAutograd path. The failing graph contains `aten.index_put`, an `empty`/`permute` copy-back target, and `aten.copy_`.

### Cluster B composition

Seeds `5000`, `5144`, and `5206`; mutations `sub_`, `fill_`, and `mul_`. Fresh-process reruns with a 300-second budget returned `PASS` for all three. These were cold-compilation budget artifacts, not stable compiler failures.

## INDEX_FILL_ ANALYSIS

The smallest stable reproducer is `.hunt/inductor-alias-mutation-v1/minimal_transpose_index_fill.py`:

```python
import torch


def f(x, index):
    view = x.transpose(0, 1)
    view.index_fill_(-1, index, 0.5)
    return x
```

The complete standalone evidence script is 13 nonblank lines; the trigger is a two-line mutation through a `3 x 2` transpose view. Inputs are a `2 x 3` float32 tensor and valid int64 indices `[0, 1]`. Eager execution changes the base storage successfully.

The failure does **not** require the generated sibling/overlap alias topology:

- direct base `index_fill_`: passes;
- transpose view, in-place: fails;
- transpose view, out-of-place `index_fill`: fails;
- transpose followed by `.contiguous()`, then in-place: passes;
- sliced non-contiguous view, in-place: fails;
- returning only the view or only the base does not remove the failure.

Therefore the minimal trigger is a non-contiguous view target for `index_fill`, not the presence of multiple returned aliases or a second aliased graph input. Empty indices, negative indices, and non-long index dtypes were not part of this v1 control matrix.

## MINIMIZED REPRODUCERS

### Cluster A

- Original cases: 42.
- Minimal reproducer: 13 nonblank lines in `minimal_transpose_index_fill.py`.
- Eager: succeeds and propagates the write to the base.
- Stable Inductor: fails reproducibly in five independent fresh processes.
- Latest nightly: passes in five independent fresh processes.

The minimal reproducer preserves legal non-overlapping view semantics and valid bounds. The failure remains when alias observables are removed, so aliasing is not necessary for this cluster.

### Cluster B

- Original cases: 3.
- No compiler reproducer was retained because all three passed in fresh processes with an extended budget.
- Final classification: harness-timeout artifact.

## BACKEND CONTROLS

Stable control matrix (`2 x 3` tensors, fresh process per cell):

| Program form | `torch.compile(..., backend="eager")` | `aot_eager` | Inductor |
|---|---:|---:|---:|
| Direct base, in-place | PASS | PASS | PASS |
| Transpose view, in-place | PASS | PASS | FAIL |
| Transpose view, out-of-place | PASS | PASS | FAIL |
| Transpose then contiguous, in-place | PASS | PASS | PASS |
| Sliced non-contiguous view, in-place | PASS | PASS | FAIL |

The stable control records are in `.hunt/inductor-alias-mutation-v1/controls-stable-v3.jsonl` and `.hunt/inductor-alias-mutation-v1/minimal-matrix.jsonl`; the nightly matrix is `.hunt/inductor-alias-mutation-v1/minimal-matrix-nightly.jsonl`. The traceback establishes that the failure occurs in AOTAutograd's functional-graph assertion while invoked by Inductor, before generated-kernel code generation. This is not a Triton kernel runtime failure.

## DTYPE CONTROLS

The minimized stable matrix tested every listed form at float32, float64, and bfloat16:

| Form | float32 | float64 | bfloat16 |
|---|---:|---:|---:|
| Direct base, in-place | PASS | PASS | PASS |
| Transpose view, in-place | FAIL | FAIL | FAIL |
| Transpose view, out-of-place | FAIL | FAIL | FAIL |
| Transpose then contiguous, in-place | PASS | PASS | PASS |
| Sliced view, in-place | FAIL | FAIL | FAIL |

The result is therefore a view-contiguity condition, not a float32-only or bfloat16-only phenomenon.

## STABLE RESULTS

**COMPUTATIONALLY VERIFIED:** the 42 index-fill records have a small legal reproducer that fails under stable Inductor and passes under eager and `aot_eager`. Five fresh stable processes reproduced the minimal float32 failure. The three timeout records passed on fresh rechecks with a longer budget.

The raw campaign's 45 count must not be interpreted as 45 stable compiler defects: 42 are the index-fill cluster and 3 are harness timeouts.

## NIGHTLY RESULTS

Two isolated environments were retained:

1. `2.7.0.dev20250310+cu124`, CUDA 12.4 nightly wheel: the minimal reproducer failed in five fresh processes with the same `copy_`/`permute` assertion. This is an older nightly and is retained as historical evidence.
2. `2.15.0.dev20260907+cpu`, git `f2ac3133587499ab2dfe33d5f4009bf89accebb0`, CPU-only nightly wheel: the minimal reproducer passed in five fresh processes. The complete 15-cell dtype/form matrix also passed.

The latest nightly result is **FIXED_IN_NIGHTLY**. The CPU-only nightly differs from the stable CUDA-enabled build, but both executions use CPU tensors and CPU Inductor; this environment difference is recorded rather than hidden.

## KNOWN-ISSUE SEARCH

GitHub API search found the exact existing PyTorch issue:

- [pytorch/pytorch#178952](https://github.com/pytorch/pytorch/issues/178952), “`torch.compile` fails on `index_fill` after `permute`: functionalization assertion (`copy_`) while eager works”.
- The issue's reproducer, exception text, FX shape, and AOTAutograd `assert_functional_graph` location match this cluster.
- The issue was closed after a report that current `main` passed the reproducer.

This cluster is therefore **KNOWN_EXISTING_ISSUE** and **FIXED_IN_NIGHTLY**, not potentially novel and not a current confirmed defect. No upstream issue was opened.

## CLASSIFICATION

- **Cluster A — 42 cases:** `KNOWN_EXISTING_ISSUE` + `FIXED_IN_NIGHTLY`. Stable result: reproducible Inductor/AOTAutograd compiler failure. Latest nightly result: pass. Minimal reproducer: 13 nonblank lines. The failure is a non-contiguous-view `index_fill` functionalization assertion, not an alias-specific numerical mismatch.
- **Cluster B — 3 cases:** `FALSE_POSITIVE` as a compiler-failure finding; specifically a harness timeout artifact. Stable extended-budget result: pass. Nightly: not applicable. No upstream status.

No cluster meets `CONFIRMED_CURRENT_DEFECT`: Cluster A is already reported upstream and passes on the latest tested nightly; Cluster B is not a stable compiler failure.

## IMPLEMENTATION CHANGES

- Added `hunt.failures` with normalized failure messages, stage diagnostics, stable fingerprints, and backward-compatible campaign clustering.
- Added `reproreduce summarize <campaign.json>`.
- Preserved raw exception messages and, for new executions, full exception tracebacks and relevant frames.
- Fixed `HuntTimeout` propagation in alias execution so harness timeouts are not reclassified as compiler failures.
- Added tests for stage classification, fingerprint normalization, clustering, serialization, backward compatibility, and timeout propagation.

## TESTS

- Focused diagnostics/alias/execution tests: passed.
- **COMPUTATIONALLY VERIFIED:** the final complete repository suite passed 99 tests in 1130.758 seconds.

## LIMITATIONS

- The original campaign records do not contain complete tracebacks; exact stage attribution comes from fresh standalone reproductions.
- Nightly validation used a CPU-only wheel because it was the available current nightly; it did not test CUDA Inductor.
- The control matrix did not cover empty indices, negative indices, alternate index dtypes, dynamic shapes, gradients, or GPU tensors.
- No upstream issue was opened automatically.

## NEXT STEP

Stop this triage milestone. Do not start another broad campaign. If desired later, add targeted regression tests around non-contiguous `index_fill` views and verify the latest CUDA nightly separately, with explicit user approval before any upstream report.
