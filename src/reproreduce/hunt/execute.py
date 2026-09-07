from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ..oracle import CompileDifferenceOracle, GradientDifferenceOracle, OracleResult
from ..oracle.numerical import ExecutionOutcome
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

    def source_oracle(
        self,
        configs: tuple[TensorConfig, ...],
        *,
        mode: str,
    ) -> CompileDifferenceOracle | GradientDifferenceOracle:
        """Build the adapter used when reducing a confirmed generated source."""
        compiler = self.compiler or (lambda fn: self._compile(fn))

        if mode == "forward":
            def evaluate(source: str):
                try:
                    namespace: dict[str, Any] = {}
                    exec(compile(source, "reduced_generated.py", "exec"), namespace)
                    function = namespace["generated_program"]
                    inputs = tuple(config.materialize() for config in configs)
                    oracle = CompileDifferenceOracle(compiler=compiler)
                    reference = oracle._capture(function, *GradientDifferenceOracle._clone_inputs(inputs))
                    try:
                        compiled = compiler(function)
                    except BaseException as error:
                        candidate = ExecutionOutcome(exception=error)
                    else:
                        candidate = oracle._capture(compiled, *GradientDifferenceOracle._clone_inputs(inputs))
                    return reference, candidate
                except BaseException as error:
                    return ExecutionOutcome(exception=error), ExecutionOutcome(exception=error)

            return CompileDifferenceOracle(compiler=compiler, source_evaluator=evaluate)

        if mode == "gradient":
            gradient_oracle = GradientDifferenceOracle(compiler=compiler)

            def evaluate(source: str):
                try:
                    namespace: dict[str, Any] = {}
                    exec(compile(source, "reduced_generated.py", "exec"), namespace)
                    function = namespace["generated_program"]
                    inputs = tuple(config.materialize() for config in configs)
                    reference_inputs = gradient_oracle._clone_inputs(inputs)
                    candidate_inputs = gradient_oracle._clone_inputs(inputs)
                    reference = gradient_oracle._capture_gradients(function, reference_inputs, {})
                    try:
                        compiled = compiler(function)
                    except BaseException as error:
                        candidate = ExecutionOutcome(exception=error)
                    else:
                        candidate = gradient_oracle._capture_gradients(compiled, candidate_inputs, {})
                    return (
                        reference.value if reference.exception is None else tuple(),
                        candidate.value if candidate.exception is None else tuple(),
                    )
                except BaseException:
                    return tuple(), tuple()

            return GradientDifferenceOracle(compiler=compiler, source_evaluator=evaluate)
        raise ValueError(f"unsupported hunt mode: {mode}")

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
