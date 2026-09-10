# Dependency-aware reduction V2

## STATUS

**COMPUTATIONALLY VERIFIED:** V2 adds an opt-in `strategy="dependency_v2"` while leaving `strategy="standard"` and the V1 `strategy="dependency"` path unchanged. Every accepted V2 candidate was evaluated by the original failure oracle.

On the nested benchmark, V2 reduced 200 -> 5 nonblank LOC, improving the V1 dependency result of 200 -> 47. The jobs=1 and jobs=4 outputs had the same SHA-256. The large exception and generated PyTorch regressions reduced to 3 LOC each and preserved their exact target fingerprints in two large-exception repeats and two generated-PyTorch repeats.

This is not a minimality claim and does not establish universal Python, PyTorch, or compiler support.

## PR #8 / BRANCH SETUP

PR #8 was green and mergeable and was merged normally into `main` at `4a106eb`. Work continued on `feat/dependency-reduction-v2`. V0.1.0 was not modified and V1 reports/records were not overwritten.

Implementation commit: `3913c4e8beb7f92f65286df8432497c716fe9c62`. The recorded benchmark runs were performed on the exact working tree that was committed at this hash; no benchmark-relevant code changed between execution and commit.

## 47-LINE ANALYSIS

The exact V1 nested export was inspected before V2. The following mutually exclusive classification uses **nonblank-line ordinals** in the 47-line file:

| Category | Count | Nonblank ordinals |
|---|---:|---|
| TARGET_CORE | 1 | 39 |
| FUNCTION_SIGNATURE | 8 | 8, 11, 17, 20, 23, 25, 32, 42 |
| CALL_ARGUMENT_PLUMBING | 11 | 9, 10, 15, 19, 26, 29, 30, 33, 34, 45, 47 |
| REQUIRED_ASSIGNMENT | 3 | 13, 27, 43 |
| CONTROL_FLOW_SHELL | 9 | 14, 28, 35–37, 40–41, 44, 46 |
| EXPRESSION_COMPLEXITY | 5 | 12, 18, 21, 24, 38 |
| RETURN_PLUMBING | 3 | 16, 22, 31 |
| SCOPE/BINDING_REQUIREMENT | 7 | 1–7 |
| REDUCER_LIMITATION | 0 | — |
| UNKNOWN | 0 | — |
| **Total** | **47** | |

The floor was dominated by semantic plumbing: seven helper/function definitions, record construction and formatting, context/payload construction, and loop/branch/exception shells. V2 therefore targeted coordinated transformations rather than additional unused-definition heuristics.

## V2 DESIGN

V2 uses a bounded deterministic fixed-point pipeline:

1. V1 oracle-backed dependency cleanup;
2. coordinated local-function parameter/call candidates;
3. call-result and assignment-result simplification;
4. control-flow branch/body candidates;
5. bounded expression, sequence, dictionary, comparison, subscript, and unary candidates;
6. V1 dependency cleanup again through the next fixed-point round;
7. existing statement/module cleanup phases.

Candidates are structurally deduplicated before evaluation. The normal oracle remains authoritative; static analysis only proposes candidates. No static slicer or semantic bypass was added.

## PARAMETER / CALL REDUCTION

V2 supports conservative module-local direct functions with simple positional or explicit keyword calls. It rejects decorators, varargs, kwargs, keyword-only parameters, defaults, uncertain call shapes, escapes as values, and class methods. Signature and all known direct call sites are edited atomically.

Nested accepted: 1 candidate, 1 oracle run, **0 nonblank LOC removed**. The transformation changed semantic plumbing without changing line count. It remains enabled because it can expose later simplifications and is cheap; it is not the source of the nested size improvement.

## CONTROL-FLOW REDUCTION

V2 proposes oracle-validated body/else replacements for `if`, body/handler replacements for `try`, and only conservative loop-body candidates when loop bindings are unused and no break/continue/yield hazard is present.

Nested: 10 candidates, 10 oracle runs, 8 accepted, **16 LOC removed**.

## EXPRESSION REDUCTION

V2 extends the existing expression candidates with comparison operands, subscripts, unary operands, tuple/list/set element removal, dictionary item removal, and return-container element candidates. Candidate generation is bounded and source-hash deduplicated. Arbitrary constants and general attribute replacement were intentionally not added.

Nested: 90 candidates, 90 oracle runs, 31 accepted, **0 nonblank LOC removed**. These candidates materially simplify expressions and enable later cleanup, but the line-count metric does not credit character-level simplification.

## FIXED-POINT PIPELINE

The nested run reached the same 5-line fixed point from jobs=1 and jobs=4. Accepted transformations exposed further dead definitions and assignments; V1 cleanup was rerun after V2 phases. The final source is:

```python
def reproduce() -> None:
    raise RuntimeError('REPROREDUCE_NESTED_TARGET')

def main() -> None:
    reproduce()
main()
```

## ABLATION

The primary ablation is V1 dependency versus V2 dependency and the V2 phase accounting:

| Nested configuration | LOC | Oracle runs | Wall time |
|---|---:|---:|---:|
| Standard, jobs=1 | 55 | 212 | 123.16s |
| Dependency V1, jobs=1 | 47 | 302 | 177.58s |
| Dependency V2, jobs=1 | 5 | 135 | 79.54s |
| Dependency V2, jobs=4 | 5 | 138 | 79.54s |

The V2 transformation-family table is:

| Phase | Candidates | Oracle runs | Accepted | LOC removed |
|---|---:|---:|---:|---:|
| parameter_call | 1 | 1 | 1 | 0 |
| call_result | 8 | 8 | 5 | 5 |
| control_flow | 10 | 10 | 8 | 16 |
| expression | 90 | 90 | 31 | 0 |

The best measured nonblank-LOC/oracle-run ratio was **control_flow: 16/10 = 1.6 LOC/run**. `call_result` yielded 0.625 LOC/run. Expression reduction yielded 0 line-count/run, despite enabling second-order cleanup. V1 dependency cleanup is not assigned a V2 phase LOC total because it is interleaved with the fixed-point pipeline.

## NESTED RESULT

**COMPUTATIONALLY VERIFIED:** 200 -> 5 nonblank LOC. Jobs=1 and jobs=4 produced identical source hash `c91367cbc2662510e0a89ab812e6e06485270796fd22c137e7c8e388f7097ebc`. The target `REPROREDUCE_NESTED_TARGET` was preserved.

## LARGE EXCEPTION

**COMPUTATIONALLY VERIFIED:** Two independent serial V2 runs produced 273 -> 3 LOC, 98 oracle runs, and identical hash `8f549d61f7a45b096acc1fb26607dd8cef7f91278516767ad848e5d779bcdef4`. Both preserved `REPROREDUCE_LARGE_TARGET`. This improves the V1 dependency result of 273 -> 4 LOC without a quality regression.

## GENERATED PYTORCH

**COMPUTATIONALLY VERIFIED:** Two independent serial V2 runs produced 331 -> 3 LOC, 109 oracle runs, and identical hash `8f847a3b568ea9fcc56fa47fc0c63765ac793ee24afc3c5c02e78b21b6c139ce`. Both preserved `REPROREDUCE_GENERATED_TORCH_TARGET`. The timeout was 30 seconds per candidate and the fixture seed was 0.

## HISTORICAL INDUCTOR

**NOT CHECKED:** The historical 152 -> 28 Inductor case was not run. Current resource inspection reported approximately 1.82 GiB available RAM on a 16-CPU Windows host; the prior V1 evidence already required a warm-cache 577.84-second run and a cold-start timeout. No risky compiler/GPU experiment was launched.

## PARALLEL COMPOSITION

**COMPUTATIONALLY VERIFIED:** Nested jobs=4 produced the same V2 source hash and LOC as jobs=1. It submitted 138 candidates, completed 138, cancelled none, and recorded 3 speculative executions with peak concurrency 4. V2's phases are largely sequential, so this run did not materially improve wall time over jobs=1. No parallel compiler run was attempted.

## ORACLE COST

V2 is substantially cheaper than V1 on the nested benchmark while producing a much smaller result: 135 versus 302 fresh oracle executions. Expression candidates are the dominant new phase cost (90 oracle runs) and should remain bounded. A phase with no line-count gain can still unlock fixed-point cleanup; expression reduction should be re-evaluated on broader benchmarks before expansion.

## RESULTS TABLE

| Benchmark | V1 LOC | V2 LOC | V1 Oracle Runs | V2 Oracle Runs | V1 Time | V2 Time | Fingerprint |
|---|---:|---:|---:|---:|---:|---:|---|
| Nested Python | 47 | 5 | 302 | 135/138 | 177.58s / 117.33s jobs=4 | 79.54s | preserved; hash identical jobs 1/4 |
| Large exception | 4 | 3 | 118 | 98 | ~68–69s | 52.17–56.71s | preserved; two repeats |
| Generated PyTorch | 4 | 3 | not repeated V1 | 109 | 889.91s historical V1 | 373.42–400.85s | preserved; two repeats |
| Historical Inductor | 28 | NOT CHECKED | 78 fresh / 445 evals | NOT CHECKED | 577.84s warm-cache V1 | NOT CHECKED | NOT CHECKED |

## REMAINING-CORE ANALYSIS

The nested result now contains only a target raise and the minimal direct call structure retained by the oracle-backed transformations. The remaining `main` wrapper and call are syntactic execution plumbing for the original entry path; V2 does not claim they are globally necessary under all allowed source transformations.

The parameter/call family did not remove line count on this fixture. The strongest measured contributors were control-flow removal and call-result/assignment-result inlining. Expression simplification had high acceptance but zero direct nonblank-LOC credit; its cost should be monitored.

## FINGERPRINT PRESERVATION

Every accepted transformation in the recorded V2 runs was evaluated through the configured original failure oracle. The exact target strings remained present in reduced subprocess stderr for all checked synthetic/generated fixtures.

## STANDALONE REPRODUCERS

Fresh subprocess checks reproduced:

- `REPROREDUCE_NESTED_TARGET`;
- `REPROREDUCE_LARGE_TARGET`; and
- `REPROREDUCE_GENERATED_TORCH_TARGET`.

Reduced source artifacts are retained beside the JSON records.

## TESTS

**COMPUTATIONALLY VERIFIED:** Full local suite passed **122 tests and 380 subtests**. Focused V2 tests cover coordinated positional parameter/call removal, control-flow and expression candidate validity, oracle preservation, strategy validation, and CLI exposure.

## COMMITS

- `4a106eb` — merged PR #8 into `main`.
- `3913c4e` — V2 implementation and tests.

## FILES / RECORDS

- `reports/dependency_reduction_v2.json`
- `reports/dependency_reduction_v2_nested_jobs1.json`
- `reports/dependency_reduction_v2_nested_jobs4.json`
- `reports/dependency_reduction_v2_large_exception_jobs1.json`
- `reports/dependency_reduction_v2_large_exception_jobs1_repeat2.json`
- `reports/dependency_reduction_v2_generated_pytorch_jobs1.json`
- `reports/dependency_reduction_v2_generated_pytorch_jobs1_repeat2.json`
- `reports/dependency_reduction_v2_*_reduced.py`
- `src/reproreduce/reduce/dependency_v2.py`
- `tests/test_dependency.py`

## README UPDATE

Not updated. The new verified results are documented in this report; the README's release-facing V0.1.0 material was left unchanged.

## PR / BRANCH

The V2 branch is `feat/dependency-reduction-v2`. A PR will be opened after the report and records are committed and the branch is pushed.

## LIMITATIONS

No formal minimality claim, general Python inliner, multi-file reduction, dynamic-reflection support, CUDA/GPU validation, or historical Inductor V2 result is claimed. The generated PyTorch result remains an exception-oracle reduction rather than evidence of general compiler-graph minimization.

## FINAL VERDICT

**COMPUTATIONALLY VERIFIED:** V2 successfully moved the nested benchmark from statement deletion toward semantic-plumbing minimization: 47 -> 5 LOC, with fewer oracle executions than V1 and deterministic serial/parallel output.

## NEXT BOTTLENECK

The next bottleneck is not unused-definition discovery. It is validating whether the aggressive control-flow/call-result simplifications generalize to real compiler failures without excessive oracle cost. The highest-value next step is a separately resourced serial V2 Inductor pilot, followed by targeted parameter/call and expression ablations.

## NEXT STEP

Open the V2 pull request after committing the report and records. Keep `dependency_v2` opt-in and do not release a new version.
