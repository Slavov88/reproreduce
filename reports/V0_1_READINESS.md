# ReproReduce v0.1 readiness

**Target state: READY_FOR_V0.1**

## Scope

ReproReduce v0.1 is an installable developer tool for reducing self-contained Python failures and inspecting PyTorch eager/compiled failures. It supports isolated execution, exception-preserving AST/tensor/module reduction, eager/compiled differential oracles, repeated validation, failure clustering, structured Inductor hunts, and export packages.

It does not promise arbitrary-project reduction, global minimality, universal `torch.compile` support, distributed/GPU portability, or automatic upstream issue submission.

## Installation verification

Fresh WSL virtual environment:

```text
python3 -m venv /tmp/reproreduce-v01-venv
python -m pip install -e .
reproreduce --version        # reproreduce 0.1.0
```

The installed package imported successfully without `PYTHONPATH`, and the installed console entry point responded to `--help`, `reduce --help`, and `summarize --help`. The core package installed without PyTorch.

## Flagship demo

Fixture: `examples/inductor_index_fill/bug.py`.

- Original source: 26 nonblank lines.
- Reduced source: 13 nonblank lines.
- Stable environment: PyTorch 2.5.1+cu124, CPU tensors, Inductor.
- Eager behavior: succeeds.
- Target behavior: stable Inductor raises the `AssertionError: n=copy_` functionalization assertion.
- Exported `.hunt/inductor-index-fill-reduced/repro.py`: reproduced the same fingerprint in a fresh process with exit code 1.
- Reduction: 115 candidate runs, 32 cache hits, 772.36 seconds.
- Current nightly behavior: passes; the issue is known and fixed upstream.

The fixture documents PyTorch issue #178952 and does not imply that current PyTorch remains broken.

## Export quality

Core exports contain `repro.py`, `README.md`, `environment.json`, and `reduction.json`. `reduction.json` now includes original/reduced return codes, timeout state, commands, durations, and stderr tails. Hunt exports additionally include backend and finding metadata.

## CLI and public API

Verified:

- `reproreduce --version`
- `reproreduce --help`
- `reproreduce reduce --help`
- `reproreduce summarize --help`
- `from reproreduce import reduce, __version__`
- `from reproreduce.oracle import ExceptionOracle, CompileDifferenceOracle`

## CI

CI installs the package before testing. Core CI now smoke-tests the version and CLI help; PyTorch/hunt CI smoke-tests the summarize command and retains the existing small hunt run. No GPU or expensive research campaign was added.

## Tests

The final complete suite passed **102 tests** in 1258.019 seconds after the productization changes. Productization-focused tests and the clean-install smoke test also passed.

## Release blockers

None identified. No PyPI publication or Git tag is created by this milestone. Generated `.hunt/` records, temporary environments, and reduction exports remain ignored.
