# Structured dynamic-shape Inductor campaign

**Status:** COMPUTATIONALLY VERIFIED for the recorded campaign; all observed discrepancies were classified as FALSE_POSITIVE after controls. No new PyTorch correctness defect is claimed.

## Question and protocol

The experiment compiled one generated function once with `torch.compile(..., dynamic=True)` and reused that callable across a deterministic four-shape trace. It did not compile a fresh dynamic callable for each shape. Each trace shape used independently materialized eager and compiled inputs from the same seed. Gradient inputs and autograd state were kept separate between the eager and compiled executions.

A static diagnostic control compiled a fresh `dynamic=False` callable for each shape. Recompilation and graph-count observations were recorded as secondary diagnostics, never as correctness failures.

## Configuration

- Implementation commit at campaign start: `3acc747bc320e3ebf36429e6a65843a148528ad1`
- Backend: `torch.compile(..., backend="inductor", dynamic=True)`
- Modes: forward and gradient
- Cases: 200 forward and 200 gradient
- Forward seeds: `3000–3199`
- Gradient seeds: `4000–4199`
- Trace length: 4 shapes per case
- Shape executions: 800 per mode, 1,600 total
- Confirmation: five attempts for forward/gradient mismatches
- Case timeout: disabled
- Zero-sized dimensions: excluded in v1
- Cache: `/tmp/reproreduce-inductor-dynamic-v1`

The 200-case runs were executed as four non-overlapping 50-case segments per mode and merged into the final records. Segment records remain in the campaign directory.

## Environment

- WSL2 Ubuntu
- Python 3.12.3
- PyTorch 2.5.1+cu124
- CUDA 12.4
- Triton 3.1.0
- NVIDIA GeForce RTX 3050 Laptop GPU
- Inputs: CPU tensors; the registered Inductor backend was invoked

## Commands

Each mode used the following command with seeds `3000`, `3050`, `3100`, `3150` or `4000`, `4050`, `4100`, `4150` and `--cases 50`:

```bash
PYTHONPATH=src TORCHINDUCTOR_CACHE_DIR=/tmp/reproreduce-inductor-dynamic-v1 \
python -m reproreduce.cli.main hunt --backend inductor --family dynamic \
  --mode forward --cases 50 --seed 3000 --confirm-runs 5 \
  --output .hunt/inductor-dynamic-v1/forward-a.json
```

The gradient command changes `--mode forward` to `--mode gradient` and uses seed `4000`.

## Processed results

| Mode | Cases | Shape executions | PASS | Numerical discrepancies | Compiler/runtime/infrastructure failures |
|---|---:|---:|---:|---:|---:|
| Forward | 200 | 800 | 174 | 26 `FORWARD_MISMATCH` | 0 |
| Gradient | 200 | 800 | 183 | 8 `GRADIENT_MISMATCH` + 9 `NONFINITE_COMPARISON` | 0 |

All 26 forward discrepancies and all 8 gradient discrepancies were bfloat16 cases and reproduced in all five campaign confirmation attempts. The 9 nonfinite gradient cases were also bfloat16; a separate fresh-process five-repeat check reproduced each one.

## Dynamic-shape coverage

Each mode covered 25 cases in each of these eight symbolic patterns:

- `vary_dim0`
- `vary_dim0_broadcast_reduction`
- `vary_dim1_broadcast_reduction`
- `two_dims_broadcast_transform`
- `batch_broadcast_permute`
- `dynamic_reduction`
- `dynamic_slice_reduction`
- `dynamic_concat_reshape`

Operation-family counts per mode were 25 for each family, except `broadcast_reduction` at 50. Each mode covered 125 one-varying-dimension cases and 75 two-varying-dimension cases. The traces used moderate positive dimensions only.

Dtype counts were:

| Mode | float32 | float64 | bfloat16 |
|---|---:|---:|---:|
| Forward | 72 | 64 | 64 |
| Gradient | 64 | 64 | 72 |

The layout generator exercised contiguous, sliced, and transposed inputs. Full machine-readable coverage is in `forward.json`, `gradient.json`, and their derived `.coverage.md` files.

## Validation funnel

- **COMPUTATIONALLY VERIFIED:** all campaign forward and gradient numerical discrepancies were repeated by fresh-process dynamic/static validation at the recorded failing shape.
- **COMPUTATIONALLY VERIFIED:** all 26 forward candidates had the same failing-shape discrepancy under dynamic and fresh static compilation.
- **COMPUTATIONALLY VERIFIED:** the 8 gradient candidates had the same failing-shape discrepancy under dynamic and fresh static compilation.
- **COMPUTATIONALLY VERIFIED:** the 9 nonfinite gradient candidates were reproduced five times in a fresh process. Most also appeared in static controls; three were dynamic-only at the first shape but remained bfloat16-only.
- **COMPUTATIONALLY VERIFIED:** representative forward operation families (`dynamic_elementwise`, `broadcast_reduction`, `broadcast_permute_reduction`, `slice_reduction`) and gradient families (`dynamic_elementwise`, `dynamic_reduction`) passed float32 and float64 dynamic and static controls.
- **COMPUTATIONALLY VERIFIED:** the three dynamic-only nonfinite representatives also passed float32 and float64 dynamic and static controls.
- **INFERRED / FALSE_POSITIVE:** the discrepancies are low-precision bfloat16 rounding or nonfinite-gradient behavior associated with the reduced-precision operation, not a surviving dynamic-shape semantic defect. No float32 or float64 candidate survived the controls.
- **NOT TESTED:** PyTorch nightly was not run because no higher-precision dynamic-only discrepancy remained.

## Minimization and export

Dynamic-aware reduction and trace minimization were added to the hunt frontend.

- Forward seed `3016`: trace reduced from four shapes to original indices `(0, 2)`; program reduced from 16 LOC to 6 LOC.
- Gradient seed `4000`: trace reduced to original indices `(0, 1)`; program reduced from 17 LOC to 6 LOC.

Both reduced dynamic traces preserved the observed discrepancy under the dynamic source adapter. Dynamic runnable harnesses were exported and executed under:

- `.hunt/inductor-dynamic-v1/export-forward-3016/`
- `.hunt/inductor-dynamic-v1/export-gradient-4000/`

## Recompilation and guard observations

The evaluator compiled one callable per case, while the backend callback observed multiple compiled graphs when shape guards or specialization required it:

| Mode | Cases | Backend graph compilations | Cases with more than one graph |
|---|---:|---:|---:|
| Forward | 200 | 280 | 80 |
| Gradient | 200 | 275 | 75 |

These counts are diagnostic only. The campaign did not treat recompilation, graph breaks, or guard specialization as correctness failures. The graph-count wrapper uses existing private backend lookup infrastructure; inability to observe counts is non-fatal to correctness evaluation.

## Software validation

- **COMPUTATIONALLY VERIFIED:** the final repository test suite passed **87 tests** after the dynamic evaluator, reducer, exporter, and campaign-report changes.

## Raw records

Complete per-case records, shape traces, shape-level results, confirmations, validation JSONL files, reductions, and exported harnesses are retained in the ignored directory `.hunt/inductor-dynamic-v1/`. Previous `.hunt/inductor-main/` and `.hunt/inductor-broadcast-v1/` evidence was not overwritten.
