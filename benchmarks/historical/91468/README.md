# PyTorch issue 91468

- Upstream issue: https://github.com/pytorch/pytorch/issues/91468
- Failure class: `torch.compile` / AOTAutograd gradient correctness
- Reported symptom: `Tensor.retain_grad()` on intermediates does not populate gradients in the compiled path.
- Backend: `aot_eager`
- Local verification: reproduced with PyTorch `2.5.1+cu121` on CPU.
- Reduced artifact: `reduced.py`

## Run

```bash
python original.py
```

The script compares eager and compiled gradients for the same small scalar input. The eager path returns gradients for both retained intermediates; the compiled path returns `None` for those gradients.

## ReproReduce result

The checked-in reduced artifact preserves the failure for the retained `z` intermediate:

```text
original: eager [1.0, 1.0] / aot_eager [None, None]
reduced:  eager [1.0, None] / aot_eager [None, None]
```

Observed reduction metrics on the local PyTorch `2.5.1+cu121` environment:

```text
Python LOC:          39 → 28
Candidate runs:      97
Cache hits:          105
Reduction wall time: 436.41 s
Median candidate:    3.75 s
```

The second retained intermediate was removed by a valid reduction; the original `z` gradient discrepancy remains. This is a known-bug reduction, not a claim of bug discovery.

The issue was selected because it has a small public reproducer, does not require a GPU, and exposes a correctness discrepancy rather than only an unsupported operation.
