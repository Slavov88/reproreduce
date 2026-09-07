# ReproReduce

ReproReduce is a framework-aware reducer for Python and PyTorch bug reproducers. It repeatedly simplifies a failing program while checking that the original failure is preserved.

The current release supports self-contained Python scripts, exception matching, recursive statement reduction, and conservative reduction of explicit PyTorch tensor constructors.

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
- candidates in isolated subprocesses with timeouts and a SQLite evaluation cache.

## Failure matching

`ExceptionOracle` matches the configured exception type and optional message pattern. Structured fingerprints normalize temporary paths, line numbers, and hexadecimal addresses. A candidate must preserve the baseline exception type, normalized message signature, and signal when applicable.

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
- PyTorch support currently targets explicit tensor constructors only.
- Repeated module reduction, `torch.compile` discrepancy oracles, gradient oracles, performance reduction, and historical regression benchmarks are not included yet.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall src
```

## Citation

No paper is associated with this project yet.

## License

MIT. See `LICENSE`.
