# ReproReduce

ReproReduce is a framework-aware reducer for Python and PyTorch bug reproducers. It repeatedly simplifies a failing program while checking that the original failure is preserved.

The current release supports self-contained Python scripts, exception matching, recursive statement reduction, conservative PyTorch tensor reduction, repeated-module reduction, and eager-vs-compiled numerical discrepancy checks.

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

`CompileDifferenceOracle` compares eager and compiled function outputs using absolute and relative tolerances. It reports mismatch counts, maximum errors, NaN/Inf mismatches, shape and dtype mismatches, and one-sided exceptions.

## Compile discrepancy check

For a direct function comparison:

```python
from reproreduce import CompileDifferenceOracle

oracle = CompileDifferenceOracle(atol=1e-5, rtol=1e-5)
result = oracle.evaluate_function(model, inputs)
print(result.interesting, result.metadata)
```

## Example

The included example contains irrelevant imports, assignments, and a function. It reduces while preserving `RuntimeError: REPROREDUCE_TARGET`:

```bash
reproreduce reduce examples/exception_bug/bug.py \
  --exception-type RuntimeError \
  --message REPROREDUCE_TARGET \
  --output repro
```

## Current limitations

- Input programs should be self-contained Python scripts.
- AST output is regenerated with `ast.unparse`; comments and formatting are not preserved.
- Input programs should be self-contained Python scripts.
- AST output is regenerated with `ast.unparse`; comments and formatting are not preserved.
- `ModuleList` reduction is conservative and skips statically indexed containers.
- Compile-discrepancy source reduction currently uses an explicit source adapter; nested output structures and gradient discrepancies are not implemented.
- Historical PyTorch regression benchmarks and performance reduction are not included yet.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall src
```

## Citation

No paper is associated with this project yet.

## License

MIT. See `LICENSE`.
