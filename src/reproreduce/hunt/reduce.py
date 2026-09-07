from __future__ import annotations

import tempfile
from pathlib import Path

from ..api import reduce as reduce_program
from ..core.result import ReductionResult
from ..oracle.base import FailureOracle
from .program import Program


def reduce_confirmed(
    program: Program,
    oracle: FailureOracle,
    *,
    timeout: float = 30.0,
    output: str | Path | None = None,
) -> ReductionResult:
    """Hand one confirmed generated program to the existing ReproReduce pipeline."""
    with tempfile.TemporaryDirectory(prefix="reproreduce-hunt-") as directory:
        source_path = Path(directory) / "generated.py"
        source_path.write_text(program.to_source(), encoding="utf-8")
        result = reduce_program(source_path, oracle=oracle, timeout=timeout)
        if output is not None:
            result.export(output)
        return result
