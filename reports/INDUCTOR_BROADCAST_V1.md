# Structured broadcasting × layout Inductor campaign

**Status:** COMPUTATIONALLY VERIFIED for the recorded campaign; candidate interpretation is INFERRED / FALSE POSITIVE.

## Configuration

- Repository commit at campaign start: `70a129e98e6fa6ad58a326c6a7341e5d0d33ba45`
- Backend: `torch.compile(..., backend="inductor")`
- Family: `broadcast`
- Confirmation: 5 attempts, 5 required successes
- Case timeout: disabled
- Forward seeds: 1500–1749 (250 cases)
- Gradient seeds: 2500–2749 (250 cases)
- Input seed: case seed, with positional offsets for inputs
- Cache: `/tmp/reproreduce-inductor-broadcast-v1`

## Environment

- WSL2 Ubuntu
- Python 3.12.3
- PyTorch 2.5.1+cu124
- CUDA 12.4
- Triton 3.1.0
- NVIDIA GeForce RTX 3050 Laptop GPU
- Inputs: CPU tensors; registered Inductor backend invocation was checked

## Commands

```bash
PYTHONPATH=src TORCHINDUCTOR_CACHE_DIR=/tmp/reproreduce-inductor-broadcast-v1 \
python -m reproreduce.cli.main hunt --backend inductor --family broadcast \
  --mode forward --cases 250 --seed 1500 --confirm-runs 5 \
  --output .hunt/inductor-broadcast-v1/forward.json

PYTHONPATH=src TORCHINDUCTOR_CACHE_DIR=/tmp/reproreduce-inductor-broadcast-v1 \
python -m reproreduce.cli.main hunt --backend inductor --family broadcast \
  --mode gradient --cases 250 --seed 2500 --confirm-runs 5 \
  --output .hunt/inductor-broadcast-v1/gradient.json
```

The long runs were completed in non-overlapping seed segments and merged into the two final JSON records. The segment records remain beside the merged records.

## Processed results

| Mode | Cases | PASS | Persistent mismatch | Compile/runtime/infrastructure failures | Timeout |
|---|---:|---:|---:|---:|---:|
| Forward | 250 | 246 | 4 | 0 | 0 |
| Gradient | 250 | 233 | 17 | 0 | 0 |

The forward cases covered six structured broadcast patterns, three dtypes, six post-operation choices, and contiguous/slice/transpose layouts. The gradient candidates were all bfloat16. The forward candidates consisted of three bfloat16 cases and one float32 reduction-order case.

## Validation

- Every campaign candidate reproduced in all five confirmation attempts.
- Fresh-process reruns reproduced all 21 original-dtype candidates.
- Fresh-process float32 and float64 controls passed for representative bfloat16 forward and gradient candidates.
- A fresh-process float64 control passed for the float32 reduction-order candidate.
- Representative forward seed 1580 and gradient seed 2568 reduced from 17 LOC to 6 LOC with failure preservation.
- Exported runnable harnesses were checked under `.hunt/inductor-broadcast-v1/export-forward-1580/` and `export-gradient-2568/`.

No candidate survived the dtype controls as a higher-precision semantic discrepancy. Nightly PyTorch validation was therefore not required for an upstream defect report, and no new PyTorch defect is claimed.

## Raw and derived artifacts

The complete per-case records, confirmations, metadata, and coverage are retained locally in the ignored directory `.hunt/inductor-broadcast-v1/`. The derived coverage reports are `forward.coverage.md` and `gradient.coverage.md` in that directory; this report and `reports/STATUS.md` are the tracked summaries.
