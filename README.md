# ReproReduce

ReproReduce is a framework-aware reducer for Python and PyTorch bug reproducers. It repeatedly simplifies a failing program while checking that the original failure is preserved.

The current release supports self-contained Python scripts, exception matching, recursive statement reduction, conservative PyTorch tensor reduction, repeated-module reduction, eager-vs-compiled discrepancy checks, and a small PyTorch correctness-search frontend.

## Why

A useful bug report is often much smaller than the program that exposed the bug. ReproReduce searches for a smaller reproducer without accepting unrelated failures such as `NameError` or a different exception message.

## Installation

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Unix:    source .venv/bin/activate
python -m pip install -e .
```

For the PyTorch integration tests:

```bash
python -m pip install -e ".[test]"
```

## Quickstart

Given a failing script:

```bash
reproreduce reduce bug.py \
  --exception-type RuntimeError \
  --message "illegal memory access" \
  --output repro
```

The input file is never modified. The output directory contains:

```text
repro/
├── repro.py
├── README.md
├── environment.json
└── reduction.json
```

Run the generated reproducer with:

```bash
python repro/repro.py
```

A Python API is also available:

```python
from reproreduce import reduce
from reproreduce.oracle import ExceptionOracle

result = reduce(
    "bug.py",
    oracle=ExceptionOracle(
        exception_type="RuntimeError",
        message_regex="illegal memory access",
    ),
)
result.export("repro")
```

## What it reduces

- Python statement lists, including nested function, branch, loop, context-manager, and `try` blocks;
- explicit `torch.randn`, `torch.zeros`, `torch.ones`, `torch.empty`, and `torch.tensor` constructors;
- tensor shapes, selected values, selected dtypes, and simple contiguous-layout attempts;
- repeated `nn.Sequential` modules;
- iteration-based `nn.ModuleList` modules; statically indexed lists are left unchanged;
- eager-vs-`torch.compile` tensor value, shape, dtype, NaN, Inf, and exception discrepancies;
- candidates in isolated subprocesses with timeouts and a SQLite evaluation cache.

## Failure matching

`ExceptionOracle` matches the configured exception type and optional message pattern. Structured fingerprints normalize temporary paths, line numbers, and hexadecimal addresses. A candidate must preserve the baseline exception type, normalized message signature, and signal when applicable.

`CompileDifferenceOracle` compares eager and compiled function outputs using absolute and relative tolerances. It reports mismatch counts, maximum errors, NaN/Inf mismatches, shape and dtype mismatches, nested output paths, and one-sided exceptions.

`GradientDifferenceOracle` compares input gradients after explicit output scalarization. It distinguishes numerical, shape, dtype, `None`, NaN, Inf, and execution discrepancies.

## Compile discrepancy check

For a direct function comparison:

```python
from reproreduce import CompileDifferenceOracle

oracle = CompileDifferenceOracle(atol=1e-5, rtol=1e-5)
result = oracle.evaluate_function(model, inputs)
print(result.interesting, result.metadata)
```

## PyTorch correctness search

The hunt frontend generates short, shape-valid PyTorch tensor programs, compares eager execution with a selected `torch.compile` backend, confirms and deduplicates numerical or gradient discrepancies, and can hand confirmed source to the reducer. It deliberately treats compiler-only errors as unsupported/error outcomes rather than correctness bugs.

```bash
reproreduce hunt \
  --backend aot_eager \
  --mode gradient \
  --cases 1000 \
  --seed 42 \
  --output .hunt/campaign.json

# Structured broadcasting × layout coverage (Linux/WSL Inductor)
reproreduce hunt \
  --backend inductor \
  --family broadcast \
  --mode forward \
  --cases 250 \
  --seed 1500 \
  --output .hunt/broadcast-forward.json

# One dynamic callable reused across four compatible shapes
reproreduce hunt \
  --backend inductor \
  --family dynamic \
  --mode gradient \
  --cases 200 \
  --seed 4000 \
  --output .hunt/dynamic-gradient.json

# Storage aliasing, views, and in-place write propagation (forward v1)
reproreduce hunt \
  --backend inductor \
  --family alias_mutation \
  --mode forward \
  --cases 400 \
  --seed 5000 \
  --output .hunt/alias-mutation.json
```

Supported initial operations include elementwise arithmetic, `sin`, `cos`, `exp`, `relu`, reductions, reshape, transpose, permute, slicing, and concatenation. Configurations cover 1D–3D edge-case shapes, `float32`, `bfloat16`, `float64`, gradients, and contiguous or derived non-contiguous layouts. `--family broadcast` selects structured scalar/tensor, row/matrix, column/matrix, singleton-middle, multi-axis, and chained broadcasting cases. `--family dynamic` generates a four-shape trace and compiles one callable with `dynamic=True`; it records shape-level outcomes and secondary graph-count observations. `--family alias_mutation` records returned views plus post-mutation input state, validates storage-alias relationships, and covers eight structured view/write patterns and eight in-place mutation families; v1 is forward-only. All families write a coverage report beside `--output`. Use Inductor on Linux/WSL or CI; a local Windows missing-MSVC failure is not a correctness finding.

Search results are candidates only. Re-run findings, test stable and nightly PyTorch where practical, inspect semantics, and search upstream issues before calling one a new bug. The frontend does not file issues automatically.

## Example

The included example contains irrelevant imports, assignments, and a function. It reduces while preserving `RuntimeError: REPROREDUCE_TARGET`:

```bash
reproreduce reduce examples/exception_bug/bug.py \
  --exception-type RuntimeError \
  --message REPROREDUCE_TARGET \
  --output repro
```

## Historical benchmark

The repository includes a reduced, CPU reproducer for [PyTorch issue #91468](https://github.com/pytorch/pytorch/issues/91468), an AOTAutograd gradient correctness issue involving `Tensor.retain_grad()`.

```text
Original: 39 LOC
Reduced:  28 LOC
Failure:  eager [1.0, 1.0] vs aot_eager [None, None]
```

This demonstrates reduction of a known bug; it is not a claim of automated bug discovery.

## Current limitations

- Input programs should be self-contained Python scripts.
- AST output is regenerated with `ast.unparse`; comments and formatting are not preserved.
- `ModuleList` reduction is conservative and skips statically indexed containers.
- Compile-discrepancy source reduction currently uses an explicit source adapter.
- Gradient source reduction currently uses an explicit source adapter.
- Historical benchmarks are illustrative and may require pinned framework versions or backends.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall src
```

## Citation

No paper is associated with this project yet.

## License

MIT. See `LICENSE`.
