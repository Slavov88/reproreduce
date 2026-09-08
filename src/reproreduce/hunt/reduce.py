from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from ..api import reduce as reduce_program
from ..core.result import ReductionResult
from ..core.run import RunResult
from ..oracle import CompileDifferenceOracle, GradientDifferenceOracle, OracleResult
from ..oracle.base import FailureOracle
from .config import TensorConfig
from .execute import OutcomeClass, ProgramExecutor
from .program import Program


class DynamicTraceOracle:
    """Source adapter that preserves one dynamic callable across a trace."""

    def __init__(
        self,
        executor: ProgramExecutor,
        config_trace: Sequence[tuple[TensorConfig, ...]],
        *,
        mode: str,
        trace_seeds: Sequence[int | None],
    ):
        self.executor = executor
        self.config_trace = tuple(tuple(step) for step in config_trace)
        self.mode = mode
        self.trace_seeds = tuple(trace_seeds)
        if len(self.config_trace) != len(self.trace_seeds):
            raise ValueError("trace_seeds must match config_trace")

    def evaluate_source(self, source: str, *, cwd: Path, timeout: float) -> tuple[RunResult, OracleResult]:
        started = time.perf_counter()
        try:
            namespace: dict[str, Any] = {}
            exec(compile(source, "reduced_dynamic.py", "exec"), namespace)
            function = namespace["generated_program"]
            compiler = self.executor.compiler or (
                lambda fn: self.executor._compile(fn, dynamic=True)
            )
            compiled = compiler(function)
            atol, rtol = self.executor._oracle_tolerances(self.config_trace[0])
            for index, (configs, seed) in enumerate(zip(self.config_trace, self.trace_seeds)):
                eager_inputs = self.executor._materialize_inputs(configs, seed)
                compiled_inputs = self.executor._materialize_inputs(configs, seed)
                if self.mode == "forward":
                    oracle = CompileDifferenceOracle(atol=atol, rtol=rtol)
                    reference = oracle._capture(function, *eager_inputs)
                    candidate = oracle._capture(compiled, *compiled_inputs)
                    comparison = oracle.compare(reference, candidate)
                else:
                    oracle = GradientDifferenceOracle(atol=atol, rtol=rtol)
                    reference = oracle._capture_gradients(function, eager_inputs, {})
                    candidate = oracle._capture_gradients(compiled, compiled_inputs, {})
                    if reference.exception is not None or candidate.exception is not None:
                        comparison = oracle._compare_exceptions(reference, candidate)
                    else:
                        comparison = oracle._compare_gradients(reference.value, candidate.value)
                if comparison.interesting:
                    metadata = dict(comparison.metadata)
                    metadata.update({"dynamic": True, "shape_index": index})
                    result = OracleResult(
                        True,
                        comparison.fingerprint,
                        comparison.score,
                        metadata,
                    )
                    return RunResult(
                        command=("dynamic-source-adapter",),
                        returncode=1,
                        stdout="",
                        stderr="",
                        duration_seconds=time.perf_counter() - started,
                    ), result
            result = OracleResult(False, "dynamic:match", metadata={"kind": "match"})
            return RunResult(
                command=("dynamic-source-adapter",),
                returncode=0,
                stdout="",
                stderr="",
                duration_seconds=time.perf_counter() - started,
            ), result
        except BaseException as error:
            result = OracleResult(
                False,
                None,
                metadata={
                    "kind": "source_adapter_exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            )
            return RunResult(
                command=("dynamic-source-adapter",),
                returncode=1,
                stdout="",
                stderr=str(error),
                duration_seconds=time.perf_counter() - started,
            ), result

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        return (
            baseline.interesting
            and candidate.interesting
            and baseline.fingerprint == candidate.fingerprint
            and baseline.metadata.get("shape_index") == candidate.metadata.get("shape_index")
        )


@dataclass(frozen=True)
class DynamicTraceReduction:
    config_trace: tuple[tuple[TensorConfig, ...], ...]
    selected_indices: tuple[int, ...]
    original_length: int

    @property
    def reduced_length(self) -> int:
        return len(self.config_trace)


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


def reduce_dynamic_execution(
    program: Program,
    config_trace: Sequence[tuple[TensorConfig, ...]],
    executor: ProgramExecutor,
    *,
    mode: str,
    input_seed: int = 0,
    trace_seeds: Sequence[int | None] | None = None,
    timeout: float = 30.0,
    output: str | Path | None = None,
) -> ReductionResult:
    """Reduce program source while preserving a dynamic-trace discrepancy."""
    trace = tuple(tuple(step) for step in config_trace)
    seeds = tuple(trace_seeds) if trace_seeds is not None else tuple(
        executor._trace_seed(input_seed, index) for index in range(len(trace))
    )
    oracle = DynamicTraceOracle(executor, trace, mode=mode, trace_seeds=seeds)
    return reduce_confirmed(program, oracle, timeout=timeout, output=output)


def minimize_dynamic_trace(
    program: Program,
    config_trace: Sequence[tuple[TensorConfig, ...]],
    executor: ProgramExecutor,
    *,
    mode: str,
    input_seed: int = 0,
) -> DynamicTraceReduction:
    """Find the shortest subsequence that preserves the dynamic discrepancy."""
    trace = tuple(tuple(step) for step in config_trace)
    if len(trace) < 2:
        raise ValueError("dynamic trace minimization requires at least two shapes")
    baseline = executor.run_dynamic(program, trace, mode=mode, input_seed=input_seed)
    if baseline.classification not in {
        OutcomeClass.FORWARD_MISMATCH,
        OutcomeClass.GRADIENT_MISMATCH,
        OutcomeClass.NONFINITE_COMPARISON,
    }:
        raise ValueError("baseline dynamic execution is not a discrepancy")
    baseline_kind = baseline.oracle_result.metadata.get("kind") if baseline.oracle_result else None
    all_seeds = tuple(executor._trace_seed(input_seed, index) for index in range(len(trace)))
    for size in range(2, len(trace) + 1):
        for selected in combinations(range(len(trace)), size):
            candidate_trace = tuple(trace[index] for index in selected)
            candidate_seeds = tuple(all_seeds[index] for index in selected)
            candidate = executor.run_dynamic(
                program,
                candidate_trace,
                mode=mode,
                trace_seeds=candidate_seeds,
            )
            candidate_kind = candidate.oracle_result.metadata.get("kind") if candidate.oracle_result else None
            if (
                candidate.classification == baseline.classification
                and candidate.oracle_result is not None
                and baseline.oracle_result is not None
                and candidate.oracle_result.fingerprint == baseline.oracle_result.fingerprint
                and candidate_kind == baseline_kind
            ):
                return DynamicTraceReduction(candidate_trace, selected, len(trace))
    return DynamicTraceReduction(trace, tuple(range(len(trace))), len(trace))


def reduce_execution(
    program: Program,
    configs: tuple[TensorConfig, ...],
    executor: ProgramExecutor,
    *,
    mode: str,
    input_seed: int | None = None,
    timeout: float = 30.0,
    output: str | Path | None = None,
) -> ReductionResult:
    """Reduce a confirmed execution using the same backend/configuration."""
    return reduce_confirmed(
        program,
        executor.source_oracle(configs, mode=mode, input_seed=input_seed),
        timeout=timeout,
        output=output,
    )
