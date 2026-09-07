from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.run import RunResult
from .base import OracleResult
from .numerical import CompileDifferenceOracle, ExecutionOutcome


@dataclass(frozen=True)
class GradientDifferenceOracle:
    """Compare eager and compiled input gradients for a scalarized output."""

    atol: float = 1e-5
    rtol: float = 1e-5
    compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None
    loss_fn: Callable[[Any], Any] | None = None
    source_evaluator: Callable[[str], tuple[tuple[Any, ...], tuple[Any, ...]]] | None = None

    def evaluate(self, run: Any) -> OracleResult:
        raise TypeError(
            "GradientDifferenceOracle evaluates function pairs; use evaluate_function"
        )

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        return baseline.interesting and candidate.interesting and baseline.fingerprint == candidate.fingerprint

    def compare(self, reference: tuple[Any, ...], candidate: tuple[Any, ...]) -> OracleResult:
        """Compare already-computed gradients by input position."""
        return self._compare_gradients(reference, candidate)

    def evaluate_source(self, source: str, *, cwd: Path, timeout: float) -> tuple[RunResult, OracleResult]:
        """Evaluate a source candidate through a controlled gradient adapter."""
        if self.source_evaluator is None:
            raise TypeError("source_evaluator is required for source reduction")
        started = time.perf_counter()
        try:
            reference, candidate = self.source_evaluator(source)
            result = self.compare(reference, candidate)
            returncode = 1 if result.interesting else 0
            stderr = ""
        except BaseException as error:
            result = OracleResult(
                False,
                None,
                metadata={"reason": "source_adapter_exception", "exception": type(error).__name__},
            )
            returncode = 1
            stderr = str(error)
        run = RunResult(
            command=("gradient-difference-adapter",),
            returncode=returncode,
            stdout="",
            stderr=stderr,
            duration_seconds=time.perf_counter() - started,
        )
        return run, result

    def evaluate_function(self, function: Callable[..., Any], *inputs: Any, **kwargs: Any) -> OracleResult:
        reference_inputs = self._clone_inputs(inputs)
        candidate_inputs = self._clone_inputs(inputs)
        reference = self._capture_gradients(function, reference_inputs, kwargs)
        try:
            compiler = self.compiler or self._torch_compile
            compiled = compiler(function)
        except BaseException as error:
            candidate = ExecutionOutcome(exception=error)
        else:
            candidate = self._capture_gradients(compiled, candidate_inputs, kwargs)

        if reference.exception is not None or candidate.exception is not None:
            return self._compare_exceptions(reference, candidate)
        return self._compare_gradients(reference.value, candidate.value)

    def _capture_gradients(
        self,
        function: Callable[..., Any],
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ExecutionOutcome:
        try:
            output = function(*inputs, **kwargs)
            loss = self._scalarize(output)
            torch = self._torch()
            tracked = [value for value in inputs if torch.is_tensor(value) and value.requires_grad]
            if not tracked or not loss.requires_grad:
                gradients = tuple(None for _ in inputs)
            else:
                computed = torch.autograd.grad(loss, tracked, allow_unused=True)
                computed_iter = iter(computed)
                gradients = tuple(
                    next(computed_iter) if torch.is_tensor(value) and value.requires_grad else None
                    for value in inputs
                )
            return ExecutionOutcome(value=gradients)
        except BaseException as error:
            return ExecutionOutcome(exception=error)

    def _scalarize(self, output: Any) -> Any:
        torch = self._torch()
        if self.loss_fn is not None:
            loss = self.loss_fn(output)
            if not torch.is_tensor(loss) or loss.numel() != 1:
                raise ValueError("loss_fn must return a scalar tensor")
            return loss
        if not torch.is_tensor(output):
            raise TypeError("default gradient scalarization requires a tensor output")
        return output.sum()

    def _compare_gradients(
        self,
        reference: tuple[Any, ...],
        candidate: tuple[Any, ...],
    ) -> OracleResult:
        for index, (reference_gradient, candidate_gradient) in enumerate(zip(reference, candidate)):
            if reference_gradient is None and candidate_gradient is None:
                continue
            if reference_gradient is None or candidate_gradient is None:
                return OracleResult(
                    True,
                    "gradient:none_mismatch",
                    metadata={
                        "kind": "none_mismatch",
                        "input": f"input[{index}]",
                        "scalarization": self._scalarization_name(),
                    },
                )
            comparison = CompileDifferenceOracle(atol=self.atol, rtol=self.rtol).compare(
                reference_gradient, candidate_gradient
            )
            if comparison.interesting:
                metadata = dict(comparison.metadata)
                metadata.update(
                    {
                        "kind": f"gradient_{metadata.get('kind', 'mismatch')}",
                        "input": f"input[{index}]",
                        "scalarization": self._scalarization_name(),
                    }
                )
                return OracleResult(
                    True,
                    f"gradient:{metadata['kind']}",
                    score=comparison.score,
                    metadata=metadata,
                )
        return OracleResult(
            False,
            "gradient:match",
            metadata={"kind": "match", "scalarization": self._scalarization_name()},
        )

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
            return OracleResult(False, "gradient:matching_exception", metadata={"kind": "matching_exception"})
        return OracleResult(
            True,
            "gradient:exception_mismatch",
            metadata={
                "kind": "exception_mismatch",
                "reference_exception": type(reference.exception).__name__,
                "candidate_exception": type(candidate.exception).__name__,
            },
        )

    @staticmethod
    def _exception_result(kind: str, error: BaseException | None) -> OracleResult:
        return OracleResult(
            True,
            f"gradient:{kind}",
            metadata={
                "kind": kind,
                "exception_type": type(error).__name__ if error is not None else None,
                "exception_message": str(error) if error is not None else None,
            },
        )

    @staticmethod
    def _clone_inputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        torch = GradientDifferenceOracle._torch()
        clones = []
        for value in inputs:
            if torch.is_tensor(value):
                clone = value.detach().clone()
                clone.requires_grad_(value.requires_grad)
                clones.append(clone)
            else:
                clones.append(value)
        return tuple(clones)

    def _scalarization_name(self) -> str:
        return "custom" if self.loss_fn is not None else "tensor.sum"

    @staticmethod
    def _torch_compile(function: Callable[..., Any]) -> Callable[..., Any]:
        torch = GradientDifferenceOracle._torch()
        return torch.compile(function)

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch
