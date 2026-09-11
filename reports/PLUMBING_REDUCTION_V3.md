# Plumbing Reduction V3

## STATUS

**COMPUTATIONALLY VERIFIED — VERIFIED_HAND_CORE_LEVEL.** Generic `strategy="dependency_v3"` reduced the historical 152-nonblank-line Inductor fixture to **12 nonblank LOC**, below the known 13-line hand-triaged reference. This is not a formal minimality result.

PR #10 was merged normally into `main` at `3850216`; work then started from `feat/plumbing-reduction-v3`. The released `v0.1.0` was not modified. The compiler run was validated at `c96d2f2`; the final tree adds only stricter mutation/escape rejection guards after that run.

## V2 BASELINE PRESERVATION

The V2 records remain unchanged, including `152 -> 22`, 399 oracle executions, 2184.732 seconds, source hash `4ecfa69e581e9ef88a0b9baa4b155fe31c125e5ffdafa67365dea558bfac77b2`, and 5/5 standalone reproduction.

## EXACT 22-LINE V2 ANALYSIS

The V2 source was `reports/inductor_dependency_reduction_v2_reduced.py`. The following is an exhaustive nonblank-line classification; “wrapper function” includes retained function-signature plumbing because the requested taxonomy has no separate signature category.

| Category | Count | Nonblank content |
|---|---:|---|
| TRUE_FAILURE_CORE | 1 | `value.transpose(0, 1).index_fill_(1, index, 0.5)` |
| CONFIG_OBJECT | 1 | `class ExperimentConfig` |
| DATACLASS / CLASS PLUMBING | 1 | `@dataclass` |
| CONSTANT / LITERAL PLUMBING | 3 | `rows`, `columns`, `index_value` fields |
| ALIAS ASSIGNMENT | 0 | none |
| WRAPPER_FUNCTION / FUNCTION SIGNATURE | 4 | `build_input`, `target_model`, `run_experiment`, `main` definitions |
| COMPILE_WRAPPER | 1 | `compiled = torch.compile(...)` |
| EXECUTION_WRAPPER | 5 | input construction/calls, print, and `main()` entry plumbing |
| RETURN PLUMBING | 2 | `return value`, compiled return |
| REQUIRED_IMPORT | 2 | `dataclasses` and `torch` imports |
| REQUIRED_TENSOR_SETUP | 2 | zero tensor and index construction |
| REDUCER_LIMITATION | 0 | none identified |
| UNKNOWN | 0 | none |
| **Total** | **22** | |

The main V2 limitation was configuration and execution scaffolding, not failure-core discovery.

The nine-line V2 gap is a structural size difference, not a canonical textual set difference: six dataclass/config lines, two `build_input` wrapper lines, and the remaining entry-wrapper shell, offset by two hand-repro diagnostic/output lines. V3 removes this gap without a bug-specific rewrite.

## V3 DESIGN

V3 preserves V1/V2 behavior and adds an opt-in fixed-point sequence:

1. V1 dependency cleanup;
2. coordinated parameter/call cleanup;
3. strict dataclass/config attribute propagation;
4. local alias and immutable-literal propagation;
5. V2 call-result inlining;
6. trivial local wrapper and `__name__` guard reduction;
7. V2 control-flow candidates;
8. V1 cleanup and existing statement cleanup;
9. bounded expression search.

Only local, structurally eligible transformations are proposed. Every accepted candidate is evaluated by the original oracle in a fresh subprocess through the normal scheduler.

## ALIAS PROPAGATION

Implemented conservative single-use local `Name -> Name` replacement. The compiler run accepted one alias candidate and removed one nonblank line. Rebinding, multiple uses, and uncertain scopes are rejected.

## CONSTANT / CONFIG PROPAGATION

Implemented immutable local constants, literal-container lookups, and strict local dataclass configuration propagation. Config models require simple literal defaults, no inheritance or methods, and no dynamic behavior. Attribute replacement is oracle-validated; dependency cleanup removes the now-unused class and import.

The compiler run accepted all three config-field candidates. They had zero direct LOC credit but enabled subsequent configuration cleanup.

## ASSIGNMENT INLINING

V3 composes the existing conservative V2 call-result/assignment inliner. On the compiler fixture it used 55 oracle executions, accepted 11 candidates, and removed 11 direct nonblank LOC.

## WRAPPER REDUCTION

Implemented trivial local wrapper candidates and zero-argument straight-line wrapper flattening. The compiler run accepted three wrapper candidates; direct LOC credit was zero because cleanup and rerendering interleaved, but the transformations exposed later dead definitions and calls.

## MAIN / EXECUTION PLUMBING

Implemented normal oracle-validated `if __name__ == "__main__":` flattening. The compiler run accepted one guard candidate and removed one direct nonblank line. The final source has a direct `print(run_experiment())` entry call.

## EXPRESSION COST POLICY

V2 used 98 expression oracle executions and approximately 479.92 seconds with zero direct nonblank LOC removal. V3 observes recent candidate duration generically rather than checking for a compiler name. The observed median was 4.009 seconds, above the 2-second expensive-oracle threshold, so V3 applied a deterministic **global 16-candidate expression budget**. It used 16 expression oracle executions, accepted 10, and removed zero direct LOC. The measured expression-phase saving against the V2 expression phase was approximately **361.46 seconds**.

The policy is global across the reduction, not reset per fixed-point round. Expression enablement was not causally isolated; no stronger claim is made.

## ABLATION / HISTORICAL INDUCTOR PHASES

| Transformation family | Oracle runs | Accepted | Direct LOC removed | Eventually enabled LOC removal |
|---|---:|---:|---:|---|
| Parameter/call coordination | 2 | 2 | 0 | Yes, later cleanup |
| Config/attribute propagation | 3 | 3 | 0 | Yes, class/import cleanup |
| Alias propagation | 1 | 1 | 1 | Yes |
| Call-result/assignment inlining | 55 | 11 | 11 | Yes; largest direct contributor |
| Wrapper reduction | 3 | 3 | 0 | Yes, fixed-point cleanup |
| Main/guard reduction | 1 | 1 | 1 | Yes |
| Expression reduction | 16 | 10 | 0 | Not separately attributable |
| Interleaved dependency cleanup | Not isolated | 17 | Interleaved | Yes |

## RESULTS

| Benchmark | V2 LOC | V3 LOC | V2 Oracle Runs | V3 Oracle Runs | V2 Time | V3 Time | Fingerprint |
|---|---:|---:|---:|---:|---:|---:|---|
| Historical Inductor | 22 | **12** | 399 | **325** | 2184.732s | **1632.187s** | Preserved: AssertionError + `_call_user_compiler` + `n=copy_` |
| Nested Python | 5 | **1** | 135 | 149 | 79.54s | 91.862s | Preserved |
| Large exception | 3 | **1** | 98 | 104 | ~54.44s | 65.328s | Preserved |
| Generated PyTorch | 3 | **1** | 109 | 115 | ~387.135s | 374.535s | Preserved |

The V3 source hashes and complete metrics are in `reports/plumbing_reduction_v3.json`, `reports/inductor_plumbing_v3.json`, and `reports/plumbing_reduction_v3_regressions.json`.

## FINAL SOURCE ANALYSIS

The automatic V3 result is 12 nonblank LOC:

```python
import torch

def target_model(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    value.transpose(0, 1).index_fill_(-1, index, 0.5)
    return value

def run_experiment() -> torch.Tensor:
    value = torch.zeros((2, 3), dtype=torch.float64)
    index = torch.tensor((-1,), dtype=torch.long)
    eager_output = target_model(value.clone(), index)
    compiled = torch.compile(target_model, backend='inductor')
    compiled_input = eager_output.clone()
    return compiled(compiled_input, index)
print(run_experiment())
```

The 12-line result is structurally different from the 13-line human reference and is not claimed minimal. It retains only generic tensor setup, the transpose/index-fill operation, and the compile/execute path.

## FINGERPRINT AND STANDALONE REPRODUCTION

**YES.** The normal exported source reproduced the target in **5/5 fresh processes**. All five runs returned a compiler failure containing `AssertionError`, `_call_user_compiler`, and `n=copy_`. Details are in `reports/inductor_plumbing_v3_standalone_repeats.json`; the exported source is `reports/inductor_plumbing_v3_export/repro.py`.

## TESTS

**COMPUTATIONALLY VERIFIED:** The final full suite passed: **127 tests and 380 subtests** in 642.99 seconds. The increase from the 122-test baseline is the five focused V3 tests; the subtest count is unchanged. After compiler validation, the final tree added only stricter rejection guards for mutated/escaped config objects; no eligible historical target object uses those cases.

## LIMITATIONS

No formal minimality, universal semantic-equivalence, universal reduction-quality, nightly-control, or parallel-compiler claim is made. Compiler validation remained serial because the observed candidate median and available RAM did not justify parallel Inductor execution. The JaxOOM ATT-2 status is unchanged and no planner correction was applied.

## VERDICT

V3 is **VERIFIED_HAND_CORE_LEVEL**: it reaches 12 LOC, below both the 15-line strong target and the 13-line hand reference, while using fewer compiler oracle executions than V2 and preserving the exact historical fingerprint.
