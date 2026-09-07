from __future__ import annotations

import tempfile
from pathlib import Path

from ..api import reduce as reduce_program
from ..core.result import ReductionResult
from ..oracle.base import FailureOracle
from .config import TensorConfig
from .execute import ProgramExecutor
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


def reduce_execution(
    program: Program,
    configs: tuple[TensorConfig, ...],
    executor: ProgramExecutor,
    *,
    mode: str,
    timeout: float = 30.0,
    output: str | Path | None = None,
) -> ReductionResult:
    """Reduce a confirmed execution using the same backend/configuration."""
    return reduce_confirmed(
        program,
        executor.source_oracle(configs, mode=mode),
        timeout=timeout,
        output=output,
    )
