# Dynamic-shape campaign plan

**Status:** CONJECTURED before implementation.

## Question

Does reusing one `torch.compile(..., dynamic=True)` callable across a deterministic compatible shape trace expose a correctness discrepancy that is absent from fresh static compilation for the same shape?

## Expected mechanism

Symbolic dimensions, shape guards, graph reuse, dynamic reductions, broadcasting, and dynamic reshape/stride handling may expose a defect that is not exercised when each shape receives an independent static compilation.

## Strongest controls

1. Eager execution at every trace shape.
2. One dynamic compiled callable reused across the full trace.
3. A fresh static compilation for each failing shape.
4. Float32 and float64 controls; bfloat16 remains exploratory.

## Smallest discriminating pilot

Generate a small deterministic mixture of one varying dimension, two varying dimensions, dynamic broadcasting, reduction, and reshape cases. Require at least two distinct shapes per trace and verify the compiler callback is invoked through one compiled callable in the evaluator. Stop expansion if cases are invalid, the callable is recompiled from scratch per shape, or all discrepancies disappear under the dynamic/static distinction and precision controls.

## Main campaign

- 200 forward cases, seeds `3000–3199`.
- 200 gradient cases, seeds `4000–4199`.
- Four shape executions per case.
- No zero-sized dimensions in v1.
- Case timeout disabled; per-case dynamic compilation remains observable through graph-count metadata.

## Success gate

A candidate is high value only if it is reproducible, survives a fresh process, is float32 or float64, and satisfies `eager == static Inductor != dynamic Inductor` at a specific shape transition. Recompilation, graph breaks, and guard specialization are diagnostic observations rather than correctness findings.
