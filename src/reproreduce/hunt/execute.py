from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ..oracle import CompileDifferenceOracle, GradientDifferenceOracle, OracleResult
from .config import TensorConfig
from .program import Program


class OutcomeClass(str, Enum):
    PASS = "PASS"
    EAGER_ERROR = "EAGER_ERROR"
    COMPILED_ERROR = "COMPILED_ERROR"
    FORWARD_MISMATCH = "FORWARD_MISMATCH"
    GRADIENT_MISMATCH = "GRADIENT_MISMATCH"
    NONDETERMINISTIC = "NONDETERMINISTIC"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class ExecutionResult:
    classification: OutcomeClass
    oracle_result: OracleResult | None
    program_id: str
    backend: str
    mode: str


class ProgramExecutor:
    def __init__(
        self,
        *,
        backend: str = "aot_eager",
        compiler: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None,
    ):
        self.backend = backend
        self.compiler = compiler

    def run(
        self,
        program: Program,
        configs: tuple[TensorConfig, ...] | None = None,
        *,
        mode: str = "forward",
    ) -> ExecutionResult:
        namespace: dict[str, Any] = {}
        exec(compile(program.to_source(), "generated_program.py", "exec"), namespace)
        function = namespace["generated_program"]
        selected_configs = configs or tuple(
            TensorConfig(
                shape=spec.shape,
                dtype=spec.dtype,
                requires_grad=spec.requires_grad,
                layout=spec.layout,
            )
            for spec in program.inputs
        )
        inputs = tuple(config.materialize() for config in selected_configs)
        compiler = self.compiler or (lambda fn: self._compile(fn))
        if mode == "forward":
            result = CompileDifferenceOracle(compiler=compiler).evaluate_function(function, *inputs)
        elif mode == "gradient":
            result = GradientDifferenceOracle(compiler=compiler).evaluate_function(function, *inputs)
        else:
            raise ValueError(f"unsupported hunt mode: {mode}")
        return ExecutionResult(
            classification=self._classification(result, mode),
            oracle_result=result,
            program_id=program.identity,
            backend=self.backend,
            mode=mode,
        )

    def _compile(self, function: Callable[..., Any]) -> Callable[..., Any]:
        import torch

        return torch.compile(function, backend=self.backend)

    @staticmethod
    def _classification(result: OracleResult, mode: str) -> OutcomeClass:
        if not result.interesting:
            return OutcomeClass.PASS
        kind = str(result.metadata.get("kind", ""))
        if kind == "candidate_only_exception":
            return OutcomeClass.COMPILED_ERROR
        if kind == "reference_only_exception":
            return OutcomeClass.EAGER_ERROR
        if kind.endswith("mismatch") or "mismatch" in kind:
            return OutcomeClass.GRADIENT_MISMATCH if mode == "gradient" else OutcomeClass.FORWARD_MISMATCH
        return OutcomeClass.COMPILED_ERROR
