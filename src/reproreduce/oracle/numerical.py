from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.run import RunResult
from .base import OracleResult


@dataclass(frozen=True)
class ExecutionOutcome:
    value: Any = None
    exception: BaseException | None = None


@dataclass(frozen=True)
class CompileDifferenceOracle:
    """Compare eager output with a compiled output using explicit tolerances."""

    atol: float = 1e-5
    rtol: float = 1e-5
    compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None
    source_evaluator: Callable[[str], tuple[Any, Any]] | None = None

    def evaluate(self, run: RunResult) -> OracleResult:
        raise TypeError(
            "CompileDifferenceOracle evaluates function pairs; use evaluate_function or compare"
        )

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        return baseline.interesting and candidate.interesting and baseline.fingerprint == candidate.fingerprint

    def evaluate_source(self, source: str, *, cwd: Path, timeout: float) -> tuple[RunResult, OracleResult]:
        """Evaluate a source candidate through an injected deterministic adapter.

        The adapter is intended for controlled reduction harnesses. Real function
        comparisons should use :meth:`evaluate_function`, which invokes
        ``torch.compile`` unless a compiler is injected.
        """
        if self.source_evaluator is None:
            raise TypeError("source_evaluator is required for source reduction")
        started = time.perf_counter()
        reference, candidate = self.source_evaluator(source)
        result = self.compare(reference, candidate)
        run = RunResult(
            command=("compile-difference-adapter",),
            returncode=1 if result.interesting else 0,
            stdout="",
            stderr="",
            duration_seconds=time.perf_counter() - started,
        )
        return run, result

    def evaluate_function(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> OracleResult:
        reference = self._capture(function, *args, **kwargs)
        try:
            compiler = self.compiler or self._torch_compile
            compiled = compiler(function)
        except BaseException as error:  # compiler failures are part of the discrepancy
            compiled = None
            candidate = ExecutionOutcome(exception=error)
        else:
            candidate = self._capture(compiled, *args, **kwargs)
        return self.compare(reference, candidate)

    def compare(self, reference: ExecutionOutcome | Any, candidate: ExecutionOutcome | Any) -> OracleResult:
        reference = self._as_outcome(reference)
        candidate = self._as_outcome(candidate)

        if reference.exception is not None or candidate.exception is not None:
            return self._compare_exceptions(reference, candidate)
        return self._compare_values(reference.value, candidate.value)

    @staticmethod
    def _as_outcome(value: ExecutionOutcome | Any) -> ExecutionOutcome:
        return value if isinstance(value, ExecutionOutcome) else ExecutionOutcome(value=value)

    @staticmethod
    def _capture(function: Callable[..., Any], *args: Any, **kwargs: Any) -> ExecutionOutcome:
        try:
            return ExecutionOutcome(value=function(*args, **kwargs))
        except BaseException as error:
            return ExecutionOutcome(exception=error)

    @staticmethod
    def _torch_compile(function: Callable[..., Any]) -> Callable[..., Any]:
        import torch

        return torch.compile(function)

    def _compare_exceptions(
        self,
        reference: ExecutionOutcome,
        candidate: ExecutionOutcome,
    ) -> OracleResult:
        if reference.exception is None:
            return self._exception_result("candidate_only_exception", candidate.exception)
        if candidate.exception is None:
            return self._exception_result("reference_only_exception", reference.exception)

        same = (
            type(reference.exception) is type(candidate.exception)
            and str(reference.exception) == str(candidate.exception)
        )
        if same:
            return OracleResult(
                interesting=False,
                fingerprint="compile:matching_exception",
                metadata={"kind": "matching_exception"},
            )
        return OracleResult(
            interesting=True,
            fingerprint="compile:exception_mismatch",
            metadata={
                "kind": "exception_mismatch",
                "reference_exception": type(reference.exception).__name__,
                "candidate_exception": type(candidate.exception).__name__,
            },
        )

    @staticmethod
    def _exception_result(kind: str, error: BaseException | None) -> OracleResult:
        return OracleResult(
            interesting=True,
            fingerprint=f"compile:{kind}",
            metadata={
                "kind": kind,
                "exception_type": type(error).__name__ if error is not None else None,
                "exception_message": str(error) if error is not None else None,
            },
        )

    def _compare_values(self, reference: Any, candidate: Any) -> OracleResult:
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None and (torch.is_tensor(reference) or torch.is_tensor(candidate)):
            if not (torch.is_tensor(reference) and torch.is_tensor(candidate)):
                return self._simple_result("type_mismatch", reference, candidate)
            return self._compare_tensors(reference, candidate, torch)

        if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
            return self._compare_scalars(reference, candidate)
        if type(reference) is not type(candidate) or reference != candidate:
            return self._simple_result("value_mismatch", reference, candidate)
        return OracleResult(False, "compile:match", metadata={"kind": "match"})

    def _compare_tensors(self, reference: Any, candidate: Any, torch: Any) -> OracleResult:
        if reference.shape != candidate.shape:
            return OracleResult(
                True,
                "compile:shape_mismatch",
                metadata={
                    "kind": "shape_mismatch",
                    "reference_shape": tuple(reference.shape),
                    "candidate_shape": tuple(candidate.shape),
                },
            )
        if reference.dtype != candidate.dtype:
            return OracleResult(
                True,
                "compile:dtype_mismatch",
                metadata={
                    "kind": "dtype_mismatch",
                    "reference_dtype": str(reference.dtype),
                    "candidate_dtype": str(candidate.dtype),
                },
            )

        reference_float = reference.detach().to(torch.float64)
        candidate_float = candidate.detach().to(torch.float64)
        reference_nan = torch.isnan(reference_float)
        candidate_nan = torch.isnan(candidate_float)
        reference_inf = torch.isinf(reference_float)
        candidate_inf = torch.isinf(candidate_float)
        nan_mismatch = reference_nan != candidate_nan
        inf_mismatch = (reference_inf != candidate_inf) | (
            reference_inf & candidate_inf & (torch.signbit(reference_float) != torch.signbit(candidate_float))
        )
        finite = ~(reference_nan | candidate_nan | reference_inf | candidate_inf)
        absolute = torch.abs(reference_float - candidate_float)
        tolerance = self.atol + self.rtol * torch.abs(reference_float)
        finite_mismatch = finite & (absolute > tolerance)
        mismatch = nan_mismatch | inf_mismatch | finite_mismatch
        mismatch_count = int(mismatch.sum().item())
        finite_absolute = absolute[finite]
        max_absolute = float(finite_absolute.max().item()) if finite_absolute.numel() else 0.0
        relative = absolute / torch.clamp(torch.abs(reference_float), min=1e-12)
        finite_relative = relative[finite]
        max_relative = float(finite_relative.max().item()) if finite_relative.numel() else 0.0
        metadata = {
            "kind": "tensor_mismatch" if mismatch_count else "match",
            "max_abs_error": max_absolute,
            "max_rel_error": max_relative,
            "mismatching_elements": mismatch_count,
            "total_elements": int(reference.numel()),
            "nan_mismatch": bool(nan_mismatch.any().item()),
            "inf_mismatch": bool(inf_mismatch.any().item()),
        }
        return OracleResult(
            interesting=mismatch_count > 0,
            fingerprint="compile:tensor_mismatch" if mismatch_count else "compile:match",
            score=max_absolute,
            metadata=metadata,
        )

    def _compare_scalars(self, reference: float | int, candidate: float | int) -> OracleResult:
        reference_float = float(reference)
        candidate_float = float(candidate)
        if math.isnan(reference_float) != math.isnan(candidate_float):
            return self._simple_result("nan_mismatch", reference, candidate)
        if math.isinf(reference_float) or math.isinf(candidate_float):
            same = reference_float == candidate_float
            return OracleResult(
                not same,
                "compile:match" if same else "compile:inf_mismatch",
                metadata={"kind": "match" if same else "inf_mismatch"},
            )
        absolute = abs(reference_float - candidate_float)
        relative = absolute / max(abs(reference_float), 1e-12)
        interesting = not math.isclose(reference_float, candidate_float, abs_tol=self.atol, rel_tol=self.rtol)
        return OracleResult(
            interesting,
            "compile:value_mismatch" if interesting else "compile:match",
            score=absolute,
            metadata={
                "kind": "value_mismatch" if interesting else "match",
                "max_abs_error": absolute,
                "max_rel_error": relative,
                "mismatching_elements": int(interesting),
                "total_elements": 1,
            },
        )

    @staticmethod
    def _simple_result(kind: str, reference: Any, candidate: Any) -> OracleResult:
        return OracleResult(
            True,
            f"compile:{kind}",
            metadata={"kind": kind, "reference_type": type(reference).__name__, "candidate_type": type(candidate).__name__},
        )
