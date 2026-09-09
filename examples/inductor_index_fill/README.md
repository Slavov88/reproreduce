# Historical Inductor `index_fill_` failure

This is a small, intentionally bloated reproducer for [PyTorch issue #178952](https://github.com/pytorch/pytorch/issues/178952). It exercises an eager-valid `index_fill_` through a non-contiguous transpose view.

## Run the example

With the historical environment (PyTorch 2.5.1 and CPU tensors):

```bash
python bug.py
```

Expected behavior:

- eager tensor operations succeed;
- `torch.compile(..., backend="inductor")` raises a `BackendCompilerFailed` exception containing the `copy_` functionalization assertion.

The issue is fixed in the latest nightly tested by ReproReduce, so a newer PyTorch may pass instead. A passing current version is not a failure of this example.

## Reduce it

Install ReproReduce from the repository root, then run:

```bash
reproreduce reduce examples/inductor_index_fill/bug.py \
  --exception-type AssertionError \
  --message "n=copy_" \
  --timeout 60 \
  --output .hunt/inductor-index-fill-reduced
```

The reducer writes `repro.py`, `README.md`, `environment.json`, and `reduction.json`. Run the exported script with:

```bash
python .hunt/inductor-index-fill-reduced/repro.py
```

This fixture is historical evidence and a product smoke test; it does not claim a new or currently unfixed PyTorch bug.
