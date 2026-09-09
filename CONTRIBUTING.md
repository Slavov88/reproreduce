# Contributing

Keep changes small and tied to a milestone. Every reducer transformation must be validated by the failure oracle and covered by a deterministic test fixture.

Before opening a change:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall src
reproreduce --help
```

For the complete v0.1 smoke sequence, see `docs/RELEASE_CHECKLIST.md`.

Do not add datasets, checkpoints, generated reproductions, or machine-specific artifacts to the repository.
