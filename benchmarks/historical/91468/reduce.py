from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from reproreduce import GradientDifferenceOracle, reduce


HERE = Path(__file__).parent


def make_adapter():
    def evaluate(source: str):
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix=".reproreduce-",
            dir=HERE,
            delete=False,
        ) as handle:
            handle.write(source)
            candidate_path = Path(handle.name)
        try:
            completed = subprocess.run(
                [sys.executable, str(candidate_path), "--json"],
                cwd=HERE,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        finally:
            candidate_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "candidate exited unsuccessfully")
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        reference = tuple(
            None if value is None else torch.tensor(value)
            for value in payload["eager"]
        )
        candidate = tuple(
            None if value is None else torch.tensor(value)
            for value in payload["compiled"]
        )
        return reference, candidate

    return evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reduce(
        HERE / "original.py",
        oracle=GradientDifferenceOracle(source_evaluator=make_adapter()),
        timeout=35,
    )
    print(result.summary())
    print(json.dumps(result.metrics, indent=2))
    if args.output:
        result.export(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
