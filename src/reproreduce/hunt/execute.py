from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ..oracle import CompileDifferenceOracle, GradientDifferenceOracle, OracleResult
from ..oracle.numerical import ExecutionOutcome
from .config import TensorConfig
from .program import Program


class HuntTimeout(TimeoutError):
    """Raised by a campaign guard when one generated case exceeds its budget."""


class OutcomeClass(str, Enum):
    PASS = "PASS"
    EAGER_ERROR = "EAGER_ERROR"
    COMPILE_FAILURE = "COMPILE_FAILURE"
    COMPILED_RUNTIME_FAILURE = "COMPILED_RUNTIME_FAILURE"
    FORWARD_MISMATCH = "FORWARD_MISMATCH"
    GRADIENT_MISMATCH = "GRADIENT_MISMATCH"
    NONFINITE_COMPARISON = "NONFINITE_COMPARISON"
    NONDETERMINISTIC = "NONDETERMINISTIC"
    TIMEOUT = "TIMEOUT"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    # Backward-compatible name for callers using the original coarse class.
    COMPILED_ERROR = "COMPILE_FAILURE"


@dataclass(frozen=True)
class ExecutionResult:
    classification: OutcomeClass
    oracle_result: OracleResult | None
    program_id: str
    backend: str
    mode: str
    failure_phase: str | None = None


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
        input_seed: int | None = None,
    ) -> ExecutionResult:
        phase: dict[str, str | None] = {"value": None}
        backend_called = {"value": False}
        uses_default_compiler = self.compiler is None
        try:
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
            inputs = self._materialize_inputs(selected_configs, input_seed)
            compiler = self.compiler or (lambda fn: self._compile(fn, backend_called))

            def observed_compiler(fn: Callable[..., Any]) -> Callable[..., Any]:
                try:
                    compiled = compiler(fn)
                except BaseException:
                    phase["value"] = "compile"
                    raise

                def observed_compiled(*args: Any, **kwargs: Any) -> Any:
                    phase["value"] = "runtime"
                    return compiled(*args, **kwargs)

                return observed_compiled

            atol, rtol = self._oracle_tolerances(selected_configs)
            if mode == "forward":
                result = CompileDifferenceOracle(
                    atol=atol,
                    rtol=rtol,
                    compiler=observed_compiler,
                ).evaluate_function(function, *inputs)
            elif mode == "gradient":
                result = GradientDifferenceOracle(
                    atol=atol,
                    rtol=rtol,
                    compiler=observed_compiler,
                ).evaluate_function(function, *inputs)
            else:
                raise ValueError(f"unsupported hunt mode: {mode}")
            classification = self._classification(result, mode, phase["value"])
            if uses_default_compiler and phase["value"] == "runtime" and not backend_called["value"]:
                metadata = dict(result.metadata)
                metadata.update({"kind": "backend_not_invoked", "backend": self.backend})
                result = OracleResult(False, None, metadata=metadata)
                classification = OutcomeClass.INFRASTRUCTURE_ERROR
        except HuntTimeout:
            raise
        except BaseException as error:
            result = OracleResult(
                interesting=False,
                fingerprint=None,
                metadata={
                    "kind": "infrastructure_error",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            )
            classification = OutcomeClass.INFRASTRUCTURE_ERROR
        return ExecutionResult(
            classification=classification,
            oracle_result=result,
            program_id=program.identity,
            backend=self.backend,
            mode=mode,
            failure_phase=phase["value"],
        )

    def source_oracle(
        self,
        configs: tuple[TensorConfig, ...],
        *,
        mode: str,
        input_seed: int | None = None,
    ) -> CompileDifferenceOracle | GradientDifferenceOracle:
        """Build the adapter used when reducing a confirmed generated source."""
        compiler = self.compiler or (lambda fn: self._compile(fn))
        atol, rtol = self._oracle_tolerances(configs)

        if mode == "forward":
            def evaluate(source: str):
                try:
                    namespace: dict[str, Any] = {}
                    exec(compile(source, "reduced_generated.py", "exec"), namespace)
                    function = namespace["generated_program"]
                    inputs = self._materialize_inputs(configs, input_seed)
                    oracle = CompileDifferenceOracle(atol=atol, rtol=rtol, compiler=compiler)
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

            return CompileDifferenceOracle(
                atol=atol,
                rtol=rtol,
                compiler=compiler,
                source_evaluator=evaluate,
            )

        if mode == "gradient":
            gradient_oracle = GradientDifferenceOracle(atol=atol, rtol=rtol, compiler=compiler)

            def evaluate(source: str):
                try:
                    namespace: dict[str, Any] = {}
                    exec(compile(source, "reduced_generated.py", "exec"), namespace)
                    function = namespace["generated_program"]
                    inputs = self._materialize_inputs(configs, input_seed)
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

            return GradientDifferenceOracle(
                atol=atol,
                rtol=rtol,
                compiler=compiler,
                source_evaluator=evaluate,
            )
        raise ValueError(f"unsupported hunt mode: {mode}")

    @staticmethod
    def _oracle_tolerances(configs: tuple[TensorConfig, ...]) -> tuple[float, float]:
        """Use PyTorch's dtype-aware comparison defaults for generated cases."""
        tolerances = {
            "bfloat16": (1e-5, 1.6e-2),
            "float16": (1e-5, 1e-3),
            "float32": (1e-5, 1.3e-6),
            "float64": (1e-7, 1e-7),
        }
        selected = [tolerances.get(config.dtype, (1e-5, 1e-5)) for config in configs]
        return max(item[0] for item in selected), max(item[1] for item in selected)

    @staticmethod
    def _materialize_inputs(
        configs: tuple[TensorConfig, ...],
        input_seed: int | None,
    ) -> tuple[Any, ...]:
        return tuple(
            config.materialize(seed=input_seed + index if input_seed is not None else None)
            for index, config in enumerate(configs)
        )

    def _compile(
        self,
        function: Callable[..., Any],
        backend_called: dict[str, bool] | None = None,
    ) -> Callable[..., Any]:
        import torch

        torch._dynamo.reset()
        actual_backend = self.backend
        if backend_called is not None:
            from torch._dynamo.backends.registry import lookup_backend

            implementation = lookup_backend(self.backend)

            def tracked_backend(graph_module: Any, example_inputs: list[Any]) -> Any:
                backend_called["value"] = True
                return implementation(graph_module, example_inputs)

            actual_backend = tracked_backend
        return torch.compile(function, backend=actual_backend)

    @staticmethod
    def _classification(
        result: OracleResult,
        mode: str,
        failure_phase: str | None,
    ) -> OutcomeClass:
        if not result.interesting:
            return OutcomeClass.PASS
        kind = str(result.metadata.get("kind", ""))
        if kind == "candidate_only_exception":
            if failure_phase == "compile":
                return OutcomeClass.COMPILE_FAILURE
            return OutcomeClass.COMPILED_RUNTIME_FAILURE
        if kind == "reference_only_exception":
            return OutcomeClass.EAGER_ERROR
        if result.metadata.get("nan_mismatch") or result.metadata.get("inf_mismatch"):
            return OutcomeClass.NONFINITE_COMPARISON
        if kind.endswith("mismatch") or "mismatch" in kind:
            return OutcomeClass.GRADIENT_MISMATCH if mode == "gradient" else OutcomeClass.FORWARD_MISMATCH
        return OutcomeClass.COMPILED_RUNTIME_FAILURE
