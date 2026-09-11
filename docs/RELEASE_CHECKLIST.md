# v0.2 release verification

This checklist verifies the source release and GitHub publication. It does not publish to PyPI.

## Clean installation

From the repository root:

```bash
python -m venv .venv-test
# Windows: .venv-test\Scripts\activate
# Unix: source .venv-test/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
reproreduce --version
reproreduce --help
reproreduce reduce --help
reproreduce summarize --help
```

## Core smoke test

```bash
reproreduce reduce examples/exception_bug/bug.py \
  --strategy dependency_v3 \
  --jobs 1 \
  --exception-type RuntimeError \
  --message REPROREDUCE_TARGET \
  --output repro-smoke
python repro-smoke/repro.py
```

Verify that `repro.py`, `README.md`, `environment.json`, and `reduction.json` exist.

## Historical PyTorch fixture

In a stable environment known to reproduce the historical issue:

```bash
reproreduce reduce examples/inductor_index_fill/bug.py \
  --strategy dependency_v3 \
  --jobs 1 \
  --exception-type AssertionError \
  --message "n=copy_" \
  --timeout 60 \
  --output .hunt/inductor-index-fill-reduced
python .hunt/inductor-index-fill-reduced/repro.py
```

A newer PyTorch may pass because the referenced upstream issue is fixed. Record the exact version instead of treating that pass as a failed smoke test.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall src
```

## Review before commit

- `git diff --check`
- `git status --short`
- confirm `.hunt/`, virtual environments, caches, and generated exports are ignored;
- confirm README commands match the installed CLI;
- confirm `reproreduce --version` reports `0.2.0`;
- confirm `dependency_v3` is accepted and `--jobs 1` remains the safe compiler setting;
- confirm no PyTorch issue, GitHub release, or PyPI publication is performed automatically.
