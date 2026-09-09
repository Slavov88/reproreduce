# Storage-aliasing and mutation Inductor campaign

**Status: COMPUTATIONALLY VERIFIED for the recorded run. No PyTorch correctness defect is claimed.**

## Question and pilot decision

- **Hypothesis:** Inductor functionalization or lowering may fail to preserve storage aliasing, view relationships, in-place mutation, or write propagation.
- **Expected mechanism:** a mutation through a view should update the base and all overlapping views; a base mutation should be visible through existing views; returned values and post-mutation input state should agree with eager execution.
- **Strongest baseline:** eager PyTorch, with storage-alias checks performed before comparing compiled results.
- **Smallest discriminating pilot:** 16 structured forward cases covering all eight patterns and the available mutation/layout cells. The pilot found no semantic mismatch; three `index_fill_` candidate-only compiler/runtime failures were retained as diagnostics.
- **Kill criterion:** stop a correctness direction when no float32/float64 semantic or state discrepancy survives the eager/compiled comparison. This campaign met that criterion.

## Protocol

The alias family extends the existing hunt IR rather than adding a separate runner. Each case:

1. materializes independent eager and compiled inputs from the same seed;
2. validates expected storage relationships on eager outputs;
3. executes one statically compiled Inductor callable;
4. compares returned observables and cloned post-mutation input state;
5. classifies compiler-only failures separately from correctness mismatches.

The eight structured patterns are `mutate_view_observe_base`, `mutate_base_observe_view`, `sibling_views`, `overlapping_views`, `chained_views`, `mutation_reduction`, `mutation_view_transform`, and `sequential_mutations`. Mutation families are `add_`, `sub_`, `mul_`, `fill_`, `zero_`, `copy_`, `index_fill_`, and slice assignment. v1 is forward-only and excludes zero-sized dimensions, autograd aliasing, and explicit mutation of user-created aliases beyond generated views.

## Configuration and provenance

- Implementation commit: `fef8634a11287c0b4d075c147ea2fbde6343af8b`
- Branch: `feat/bug-hunter`
- Backend: `torch.compile(..., backend="inductor")`
- Mode: forward
- Cases: 400, seeds `5000–5399`
- Confirmation attempts: 5 for semantic mismatches (none were produced)
- Case timeout: 30 seconds
- Cache: `/tmp/reproreduce-inductor-alias-mutation-v3`
- Raw record: `.hunt/inductor-alias-mutation-v1/campaign-v3.json`
- Coverage record: `.hunt/inductor-alias-mutation-v1/campaign-v3.coverage.md`

Environment: WSL2 Ubuntu, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA 12.4, Triton 3.1.0, NVIDIA GeForce RTX 3050 Laptop GPU. Inputs were CPU tensors; the registered Inductor backend was invoked.

Command:

```bash
PYTHONPATH=src TORCHINDUCTOR_CACHE_DIR=/tmp/reproreduce-inductor-alias-mutation-v3 \
python -m reproreduce.cli.main hunt --backend inductor \
  --family alias_mutation --mode forward --cases 400 --seed 5000 \
  --confirm-runs 5 --case-timeout 30 \
  --output .hunt/inductor-alias-mutation-v1/campaign-v3.json \
  --coverage-output .hunt/inductor-alias-mutation-v1/campaign-v3.coverage.md
```

## Results

| Outcome | Count |
|---|---:|
| Eager-valid cases | 400 |
| PASS | 355 |
| Candidate-only compiled runtime failures | 45 |
| Compile failures | 0 |
| Forward mismatches | 0 |
| Nonfinite comparisons | 0 |
| Infrastructure failures | 0 |
| Timeouts | 0 |
| Invalid generated cases | 0 |
| Backend graph compilations | 400 |
| Observed recompilations | 0 |

All 400 eager cases passed storage-alias validation. No returned-value, post-mutation-state, alias-relationship, numerical, or nonfinite discrepancy survived the compiled comparison. The 45 candidate-only failures were `BackendCompilerFailed` exceptions; 42 involved `index_fill_`, and three additional failures involved `sub_`, `fill_`, or `mul_` on individual view/layout cases. They were not treated as correctness findings. A repeat campaign from the same commit retained the same zero-mismatch result, with 42 candidate-only failures rather than 45, so the compiler-failure count is diagnostic rather than a new-best claim.

The candidate-only failures were distributed across float32 (7), bfloat16 (21), and float64 (14) in the recorded run. Most occurred for `index_fill_` cases. Fresh-process spot checks reproduced a float32 `index_fill_` compiler failure (seed 5040) and a bfloat16 failure (seed 5105); a float64 `index_fill_` control (seed 5296) passed. This confirms the failures are separate backend-support/error outcomes, not hidden eager-vs-compiled semantic mismatches.

## Coverage

- Eight patterns: 50 cases each.
- Dtypes: float32 120, float64 136, bfloat16 144.
- Layouts: contiguous 192, transpose 128, slice 128.
- Overlapping-view pattern: 50 cases; other patterns: 350 cases.
- Every case observed post-mutation state and returned view observables.
- Mutation counts: `sub_` 56, `mul_` 56, `add_` 48, `fill_` 48, `zero_` 48, `copy_` 48, `index_fill_` 48, and slice assignment 48.

## Reduction and export support

Alias-aware reduction now preserves the same return/state comparison and alias checks through the existing reduction adapter. Reproducer export includes eager and compiled return values, cloned input states, and per-observable and state differences. No semantic discrepancy existed to minimize or export in this campaign.

## Software validation

- **COMPUTATIONALLY VERIFIED:** focused alias tests passed 5/5.
- **COMPUTATIONALLY VERIFIED:** the complete repository suite passed 92 tests in 852.905 seconds.
- **COMPUTATIONALLY VERIFIED:** the 400-case campaign had zero eager-invalid and zero infrastructure cases.
- **NOT TESTED:** PyTorch nightly, because no float32/float64 semantic discrepancy survived the primary comparison and no correctness finding remained to validate.

## Interpretation and next step

**OBSERVED:** the generated aliasing and mutation programs were eager-valid, and all completed Inductor comparisons agreed on returned values and post-mutation state. **OBSERVED:** `index_fill_` dominated compiler-only failures, with three additional isolated mutation/layout failures. **CONJECTURED:** those failures reflect an Inductor lowering/support boundary for these generated functionalized view-write graphs; this report does not assert a PyTorch bug. The next separate milestone should extend aliasing to zero-sized dimensions, autograd, and more explicit view/write interactions while retaining the same validation funnel.
