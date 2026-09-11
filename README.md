# ReproReduce

ReproReduce is a failure-preserving reducer for self-contained Python programs and PyTorch compiler reproducers. It runs candidates in isolated subprocesses, keeps only transformations that preserve the configured failure, and exports a small standalone reproducer.

## Why use it?

Compiler failures are often buried under irrelevant setup. ReproReduce turns a large failing script into a smaller artifact that is easier to debug or attach to an issue.

**ReproReduce has been computationally verified on failing programs containing hundreds of lines, including 273 → 3 nonblank-LOC Python, 200 → 5 nonblank-LOC nested Python, 331 → 3 nonblank-LOC generated PyTorch, and 152 → 22 nonblank-LOC historical PyTorch Inductor reductions, with target failures preserved in standalone reproducers.**

## Verified reductions

| Case | Type | Original | Reduced | Reduction | Preserved |
|---|---|---:|---:|---:|---|
| Large Python exception | Synthetic | 273 nonblank LOC | 3 LOC | 98.9% | Yes |
| Historical PyTorch Inductor bug | Historical real bug | 152 nonblank LOC | 22 LOC | 85.5% | Yes |
| Nested Python | Synthetic | 200 nonblank LOC | 5 LOC | 97.5% | Yes |
| Generated PyTorch program | Generated realistic | 331 nonblank LOC | 3 LOC | 99.1% | Yes |

> **Historical Inductor result:** Dependency V2 reduced the 152-line historical fixture to 22 nonblank LOC while preserving the stable Inductor `AssertionError` at `_call_user_compiler` containing `n=copy_`. The exported standalone reproducer reproduced the target in 5/5 fresh processes. This corresponds to [PyTorch #178952](https://github.com/pytorch/pytorch/issues/178952); the latest nightly tested by this project fixes it. The prior 28-line result and V2 cost details remain in the reports. This is a historical reduction, not a claim that ReproReduce discovered the issue or that it is currently unfixed.

Large reductions currently take minutes rather than seconds. Compiler-backed cases can be substantially slower because candidate evaluation invokes compiler work; performance optimization is an active development area.

See the [large-reduction baseline report](reports/LARGE_REDUCTION_BENCHMARKS_V1.md) and [Dependency V2 validation report](reports/INDUCTOR_DEPENDENCY_REDUCTION_V2_RETRY.md) for environments, costs, fingerprint details, and limitations.

## Install

Core reduction has no runtime dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:   source .venv/bin/activate
python -m pip install -e .
```

Install the optional PyTorch test dependency only when needed:

```bash
python -m pip install -e ".[test]"
```

## Quickstart: reduce an exception

```bash
reproreduce reduce examples/exception_bug/bug.py \
  --exception-type RuntimeError \
  --message REPROREDUCE_TARGET \
  --output repro
python repro/repro.py
```

The input is never modified. An export contains:

```text
repro/
├── repro.py
├── README.md
├── environment.json
└── reduction.json
```

## Real PyTorch / Inductor example

The tracked fixture in `examples/inductor_index_fill/bug.py` reproduces a historical PyTorch compiler failure:

```bash
reproreduce reduce examples/inductor_index_fill/bug.py \
  --exception-type AssertionError \
  --message "n=copy_" \
  --timeout 60 \
  --output .hunt/inductor-index-fill-reduced
python .hunt/inductor-index-fill-reduced/repro.py
```

In the verified PyTorch 2.5.1+cu124 environment, the 26-line fixture reduced to 13 nonblank lines while preserving the `n=copy_` assertion fingerprint. Eager execution succeeds; the compiled Inductor path fails. The issue is known upstream as [PyTorch #178952](https://github.com/pytorch/pytorch/issues/178952) and passes on the latest nightly tested by this project, so this example does not claim a current unfixed bug.

## What ReproReduce validates

- repeated failure matching with a configurable oracle;
- isolated subprocess execution, timeouts, and SQLite candidate caching;
- exception fingerprints that normalize paths, line numbers, addresses, and relevant frames;
- eager-versus-compiled tensor and gradient comparisons;
- structured PyTorch Inductor hunts for broadcasting, dynamic shapes, and aliasing/mutation;
- compiler-failure fingerprint clustering with:

  ```bash
  reproreduce summarize .hunt/campaign.json
  ```

- standalone export with environment and reduction metadata.

A compiler crash is not automatically a novel bug, and a numerical discrepancy is not automatically a correctness defect. Re-run candidates, inspect the failure stage, compare controls, and check upstream reports.

## Python API

```python
from reproreduce import reduce
from reproreduce.oracle import ExceptionOracle

result = reduce(
    "bug.py",
    oracle=ExceptionOracle(
        exception_type="RuntimeError",
        message_regex="REPROREDUCE_TARGET",
    ),
)
print(result.original_loc, result.reduced_loc)
result.export("repro")
```

For direct eager/compiled comparisons:

```python
from reproreduce import CompileDifferenceOracle

result = CompileDifferenceOracle(atol=1e-5, rtol=1e-5).evaluate_function(model, inputs)
print(result.interesting, result.metadata)
```

## Structured PyTorch hunts

These commands are exploratory research tools, not required for ordinary reduction:

```bash
# Differential forward/gradient search
reproreduce hunt --backend aot_eager --mode gradient --cases 100 --seed 42 \
  --output .hunt/campaign.json

# Structured Inductor families
reproreduce hunt --backend inductor --family broadcast --cases 250 --seed 1500 \
  --output .hunt/broadcast-forward.json
reproreduce hunt --backend inductor --family dynamic --mode gradient --cases 200 --seed 4000 \
  --output .hunt/dynamic-gradient.json
reproreduce hunt --backend inductor --family alias_mutation --cases 400 --seed 5000 \
  --output .hunt/alias-mutation.json
```

## Failure statuses

Execution statuses include `PASS`, `COMPILE_FAILURE`, `COMPILED_RUNTIME_FAILURE`, `EAGER_ERROR`, `FORWARD_MISMATCH`, `GRADIENT_MISMATCH`, `NONFINITE_COMPARISON`, `INFRASTRUCTURE_ERROR`, and `TIMEOUT`. Cluster-level research labels include `FALSE_POSITIVE`, `KNOWN_UNSUPPORTED`, `KNOWN_EXISTING_ISSUE`, `FIXED_IN_NIGHTLY`, `POTENTIALLY_NOVEL`, and `CONFIRMED_CURRENT_DEFECT`.

These labels describe evidence, not certainty. In particular, compiler failure does not imply a PyTorch defect, and a fixed nightly result does not imply that the stable version was never defective.

## v0.1 scope and limitations

ReproReduce v0.1 supports self-contained Python scripts, exception-preserving AST reduction, selected PyTorch tensor/module reductions, eager/compiled differential checks, repeated validation, failure clustering, structured Inductor hunts, and standalone export.

It does **not** promise arbitrary-project reduction, global minimality, universal `torch.compile` support, automatic upstream issue filing, distributed or multi-GPU support, or support for every tensor/operator/dynamic-shape combination. Generated hunt artifacts and caches belong in ignored `.hunt/`; historical research evidence is documented under `reports/`.

## Development

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall src
```

See `docs/ARCHITECTURE.md` for the pipeline and `docs/RELEASE_CHECKLIST.md` for the v0.1 verification sequence. Research results and limitations are summarized in `reports/STATUS.md`.

## License

MIT. See `LICENSE`.
