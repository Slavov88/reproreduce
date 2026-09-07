# Contributing

Keep changes small and tied to a milestone. Every reducer transformation must be validated by the failure oracle and covered by a deterministic test fixture.

Before opening a change:

```bash
python -m unittest discover -s tests -v
```

Do not add datasets, checkpoints, generated reproductions, or machine-specific artifacts to the repository.
