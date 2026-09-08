from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import traceback
from typing import Any, Callable, Sequence

from ..oracle import CompileDifferenceOracle, GradientDifferenceOracle, OracleResult
from ..oracle.numerical import ExecutionOutcome
from .config import TensorConfig
from .failures import failure_metadata
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
    INVALID_CASE = "INVALID_CASE"


@dataclass(frozen=True)
class ExecutionResult:
    classification: OutcomeClass
    oracle_result: OracleResult | None
    program_id: str
    backend: str
    mode: str
    failure_phase: str | None = None
    dynamic: bool = False
    shape_index: int | None = None
    shape_trace: tuple[tuple[tuple[int, ...], ...], ...] = ()
    shape_executions: int = 1
    compiled_graphs: int = 0
    recompilation_observed: bool | None = None
    shape_results: tuple[dict[str, Any], ...] = ()
    alias_metadata: dict[str, Any] = field(default_factory=dict)


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
        compile_dynamic: bool | None = None,
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
            compiler = self.compiler or (
                lambda fn: self._compile(fn, backend_called, dynamic=compile_dynamic)
            )

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
            result = self._annotate_failure(result, phase["value"])
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

    def run_alias(
        self,
        program: Program,
        configs: tuple[TensorConfig, ...],
        *,
        return_names: tuple[str, ...],
        alias_pairs: tuple[tuple[str, str, bool], ...],
        mode: str = "forward",
        input_seed: int | None = None,
    ) -> ExecutionResult:
        """Compare return values and post-mutation state for an aliasing case."""
        if mode != "forward":
            raise ValueError("alias_mutation v1 supports forward mode only")
        phase: dict[str, str | None] = {"value": None}
        backend_called = {"value": False}
        compiled_graphs = {"value": 0}
        uses_default_compiler = self.compiler is None
        try:
            namespace: dict[str, Any] = {}
            exec(compile(program.to_source(), "generated_alias_program.py", "exec"), namespace)
            function = namespace["generated_program"]
            eager_inputs = self._materialize_inputs(configs, input_seed)
            compiled_inputs = self._materialize_inputs(configs, input_seed)
            compiler = self.compiler or (
                lambda fn: self._compile(fn, backend_called, compiled_graphs=compiled_graphs)
            )
            try:
                compiled = compiler(function)
            except HuntTimeout:
                raise
            except BaseException as error:
                phase["value"] = "compile"
                metadata = {
                    "kind": "candidate_only_exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
                metadata.update(
                    failure_metadata(
                        type(error).__name__,
                        str(error),
                        phase=phase["value"],
                        traceback_text="".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    )
                )
                result = OracleResult(
                    True,
                    metadata["failure_fingerprint"],
                    metadata=metadata,
                )
                return ExecutionResult(
                    classification=OutcomeClass.COMPILE_FAILURE,
                    oracle_result=result,
                    program_id=program.identity,
                    backend=self.backend,
                    mode=mode,
                    failure_phase=phase["value"],
                    compiled_graphs=compiled_graphs["value"],
                    recompilation_observed=compiled_graphs["value"] > 1,
                )
            eager = self._capture_alias(function, eager_inputs)
            if eager.exception is not None:
                return self._alias_invalid_result(program, mode, "eager_error", eager.exception)
            eager_aliases = self._alias_checks(eager.value["return"], return_names, alias_pairs)
            if not all(check["valid"] for check in eager_aliases):
                return self._alias_invalid_result(
                    program,
                    mode,
                    "alias_relationship_invalid",
                    metadata={"alias_checks": eager_aliases},
                )
            compiled = self._capture_alias(compiled, compiled_inputs)
            phase["value"] = "runtime"
            if compiled.exception is not None:
                comparison = CompileDifferenceOracle().compare(eager, compiled)
                comparison = self._annotate_failure(
                    comparison,
                    phase["value"],
                    error=compiled.exception,
                )
            else:
                compiled_aliases = self._alias_checks(compiled.value["return"], return_names, alias_pairs)
                if not all(check["valid"] for check in compiled_aliases):
                    comparison = OracleResult(
                        True,
                        "alias:relationship_mismatch",
                        metadata={
                            "kind": "alias_relationship_mismatch",
                            "reason": "compiled_alias_relationship_invalid",
                            "alias_checks": compiled_aliases,
                        },
                    )
                else:
                    comparison = CompileDifferenceOracle(
                        atol=self._oracle_tolerances(configs)[0],
                        rtol=self._oracle_tolerances(configs)[1],
                    ).compare(eager.value, compiled.value)
                    if comparison.interesting:
                        metadata = dict(comparison.metadata)
                        metadata.setdefault("reason", "observable_state_mismatch")
                        comparison = OracleResult(
                            True,
                            comparison.fingerprint,
                            comparison.score,
                            metadata,
                        )
            if uses_default_compiler and not backend_called["value"]:
                metadata = dict(comparison.metadata)
                metadata.update({"kind": "backend_not_invoked", "backend": self.backend})
                comparison = OracleResult(False, None, comparison.score, metadata)
                classification = OutcomeClass.INFRASTRUCTURE_ERROR
            else:
                classification = self._classification(comparison, mode, phase["value"])
            alias_metadata = {
                "return_names": return_names,
                "compiled_graphs": compiled_graphs["value"],
                "return_observables_compared": True,
                "eager_aliases": eager_aliases,
                "compiled_aliases": (
                    compiled_aliases if compiled.exception is None else []
                ),
            }
            return ExecutionResult(
                classification=classification,
                oracle_result=comparison,
                program_id=program.identity,
                backend=self.backend,
                mode=mode,
                failure_phase=phase["value"],
                compiled_graphs=compiled_graphs["value"],
                recompilation_observed=compiled_graphs["value"] > 1,
                alias_metadata=alias_metadata,
            )
        except HuntTimeout:
            raise
        except BaseException as error:
            result = OracleResult(
                False,
                None,
                metadata={
                    "kind": "infrastructure_error",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            )
            return ExecutionResult(
                classification=OutcomeClass.INFRASTRUCTURE_ERROR,
                oracle_result=result,
                program_id=program.identity,
                backend=self.backend,
                mode=mode,
                failure_phase=phase["value"],
                compiled_graphs=compiled_graphs["value"],
                recompilation_observed=compiled_graphs["value"] > 1,
            )

    @staticmethod
    def _capture_alias(function: Callable[..., Any], inputs: tuple[Any, ...]) -> ExecutionOutcome:
        try:
            output = function(*inputs)
            snapshots = tuple(
                value.detach().clone() if hasattr(value, "detach") else value
                for value in inputs
            )
            return ExecutionOutcome(value={"return": output, "inputs": snapshots})
        except HuntTimeout:
            raise
        except BaseException as error:
            return ExecutionOutcome(exception=error)

    @staticmethod
    def _alias_checks(
        output: Any,
        return_names: tuple[str, ...],
        alias_pairs: tuple[tuple[str, str, bool], ...],
    ) -> list[dict[str, Any]]:
        import torch

        values = output if isinstance(output, tuple) else (output,)
        mapping = dict(zip(return_names, values))
        checks = []
        for left, right, expected in alias_pairs:
            left_value = mapping.get(left)
            right_value = mapping.get(right)
            observed = False
            if torch.is_tensor(left_value) and torch.is_tensor(right_value):
                try:
                    observed = bool(torch._C._is_alias_of(left_value, right_value))
                except AttributeError:
                    observed = left_value.untyped_storage().data_ptr() == right_value.untyped_storage().data_ptr()
            checks.append(
                {
                    "left": left,
                    "right": right,
                    "expected": expected,
                    "observed": observed,
                    "valid": observed == expected,
                    "left_shape": tuple(left_value.shape) if torch.is_tensor(left_value) else None,
                    "right_shape": tuple(right_value.shape) if torch.is_tensor(right_value) else None,
                    "left_stride": tuple(left_value.stride()) if torch.is_tensor(left_value) else None,
                    "right_stride": tuple(right_value.stride()) if torch.is_tensor(right_value) else None,
                }
            )
        return checks

    def _alias_invalid_result(
        self,
        program: Program,
        mode: str,
        reason: str,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        details = dict(metadata or {})
        details.update({"kind": "invalid_case", "reason": reason})
        if error is not None:
            details.update({"exception_type": type(error).__name__, "exception_message": str(error)})
        return ExecutionResult(
            classification=OutcomeClass.INVALID_CASE,
            oracle_result=OracleResult(False, None, metadata=details),
            program_id=program.identity,
            backend=self.backend,
            mode=mode,
            failure_phase="eager",
            alias_metadata=details,
        )

    def run_dynamic(
        self,
        program: Program,
        config_trace: Sequence[tuple[TensorConfig, ...]],
        *,
        mode: str = "forward",
        input_seed: int | None = None,
        trace_seeds: Sequence[int | None] | None = None,
    ) -> ExecutionResult:
        """Compile one callable dynamically and reuse it across a shape trace."""
        if mode not in {"forward", "gradient"}:
            raise ValueError(f"unsupported hunt mode: {mode}")
        trace = tuple(tuple(configs) for configs in config_trace)
        if len(trace) < 2:
            raise ValueError("dynamic execution requires at least two shape instances")
        if trace_seeds is not None and len(trace_seeds) != len(trace):
            raise ValueError("trace_seeds must match the shape trace length")
        shape_trace = tuple(tuple(config.shape for config in configs) for configs in trace)
        backend_called = {"value": False}
        compiled_graphs = {"value": 0}
        uses_default_compiler = self.compiler is None
        shape_results: list[dict[str, Any]] = []
        first_failure: tuple[OutcomeClass, OracleResult, str | None, int] | None = None

        try:
            namespace: dict[str, Any] = {}
            exec(compile(program.to_source(), "generated_dynamic_program.py", "exec"), namespace)
            function = namespace["generated_program"]
            compiler = self.compiler or (
                lambda fn: self._compile(
                    fn,
                    backend_called,
                    dynamic=True,
                    compiled_graphs=compiled_graphs,
                )
            )
            try:
                compiled = compiler(function)
            except BaseException as error:
                result = OracleResult(
                    True,
                    "dynamic:compile_failure",
                    metadata={
                        "kind": "candidate_only_exception",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                )
                return self._dynamic_result(
                    result,
                    program,
                    mode=mode,
                    shape_trace=shape_trace,
                    shape_index=0,
                    shape_executions=0,
                    compiled_graphs=compiled_graphs["value"],
                    failure_phase="compile",
                    shape_results=(),
                )

            atol, rtol = self._oracle_tolerances(trace[0])
            for index, configs in enumerate(trace):
                step_seed = (
                    trace_seeds[index]
                    if trace_seeds is not None
                    else self._trace_seed(input_seed, index)
                )
                eager_inputs = self._materialize_inputs(configs, step_seed)
                compiled_inputs = self._materialize_inputs(configs, step_seed)
                if mode == "forward":
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
                metadata = dict(comparison.metadata)
                metadata.update(
                    {
                        "dynamic": True,
                        "shape_index": index,
                        "shape": shape_trace[index],
                        "shape_trace": shape_trace,
                        "compiled_graphs": compiled_graphs["value"],
                    }
                )
                step_oracle = OracleResult(
                    comparison.interesting,
                    comparison.fingerprint,
                    comparison.score,
                    metadata,
                )
                step_classification = self._classification(step_oracle, mode, "runtime")
                shape_results.append(
                    {
                        "shape_index": index,
                        "shape": shape_trace[index],
                        "classification": step_classification.value,
                        "fingerprint": step_oracle.fingerprint,
                        "metadata": metadata,
                    }
                )
                if step_classification != OutcomeClass.PASS and first_failure is None:
                    failure_phase = "compile" if (
                        step_classification in {OutcomeClass.COMPILE_FAILURE, OutcomeClass.COMPILED_RUNTIME_FAILURE}
                        and compiled_graphs["value"] == 0
                    ) else "runtime"
                    first_failure = (step_classification, step_oracle, failure_phase, index)

            if first_failure is None:
                result = OracleResult(
                    False,
                    "dynamic:match",
                    metadata={
                        "kind": "match",
                        "dynamic": True,
                        "shape_trace": shape_trace,
                        "compiled_graphs": compiled_graphs["value"],
                    },
                )
                classification = OutcomeClass.PASS
                failure_phase = None
                shape_index = None
            else:
                classification, result, failure_phase, shape_index = first_failure
            if uses_default_compiler and not backend_called["value"]:
                metadata = dict(result.metadata)
                metadata.update({"kind": "backend_not_invoked", "backend": self.backend})
                result = OracleResult(False, None, result.score, metadata)
                classification = OutcomeClass.INFRASTRUCTURE_ERROR
            return ExecutionResult(
                classification=classification,
                oracle_result=result,
                program_id=program.identity,
                backend=self.backend,
                mode=mode,
                failure_phase=failure_phase,
                dynamic=True,
                shape_index=shape_index,
                shape_trace=shape_trace,
                shape_executions=len(trace),
                compiled_graphs=compiled_graphs["value"],
                recompilation_observed=compiled_graphs["value"] > 1,
                shape_results=tuple(shape_results),
            )
        except BaseException as error:
            result = OracleResult(
                False,
                None,
                metadata={
                    "kind": "infrastructure_error",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "dynamic": True,
                },
            )
            return ExecutionResult(
                classification=OutcomeClass.INFRASTRUCTURE_ERROR,
                oracle_result=result,
                program_id=program.identity,
                backend=self.backend,
                mode=mode,
                failure_phase="runtime",
                dynamic=True,
                shape_trace=shape_trace,
                shape_executions=len(shape_results),
                compiled_graphs=compiled_graphs["value"],
                recompilation_observed=compiled_graphs["value"] > 1,
                shape_results=tuple(shape_results),
            )

    @staticmethod
    def _trace_seed(input_seed: int | None, index: int) -> int | None:
        if input_seed is None:
            return None
        return input_seed + index * 1009

    def _dynamic_result(
        self,
        result: OracleResult,
        program: Program,
        *,
        mode: str,
        shape_trace: tuple[tuple[tuple[int, ...], ...], ...],
        shape_index: int | None,
        shape_executions: int,
        compiled_graphs: int,
        failure_phase: str | None,
        shape_results: tuple[dict[str, Any], ...],
    ) -> ExecutionResult:
        return ExecutionResult(
            classification=OutcomeClass.COMPILE_FAILURE,
            oracle_result=result,
            program_id=program.identity,
            backend=self.backend,
            mode=mode,
            failure_phase=failure_phase,
            dynamic=True,
            shape_index=shape_index,
            shape_trace=shape_trace,
            shape_executions=shape_executions,
            compiled_graphs=compiled_graphs,
            recompilation_observed=compiled_graphs > 1,
            shape_results=shape_results,
        )

    def run_static_trace(
        self,
        program: Program,
        config_trace: Sequence[tuple[TensorConfig, ...]],
        *,
        mode: str = "forward",
        input_seed: int | None = None,
    ) -> tuple[ExecutionResult, ...]:
        """Compile a fresh static callable for each trace shape as a diagnostic control."""
        return tuple(
            self.run(
                program,
                tuple(configs),
                mode=mode,
                input_seed=self._trace_seed(input_seed, index),
                compile_dynamic=False,
            )
            for index, configs in enumerate(config_trace)
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
        *,
        dynamic: bool | None = None,
        compiled_graphs: dict[str, int] | None = None,
    ) -> Callable[..., Any]:
        import torch

        torch._dynamo.reset()
        actual_backend = self.backend
        if backend_called is not None:
            from torch._dynamo.backends.registry import lookup_backend

            implementation = lookup_backend(self.backend)

            def tracked_backend(graph_module: Any, example_inputs: list[Any]) -> Any:
                backend_called["value"] = True
                if compiled_graphs is not None:
                    compiled_graphs["value"] += 1
                return implementation(graph_module, example_inputs)

            actual_backend = tracked_backend
        compile_kwargs: dict[str, Any] = {"backend": actual_backend}
        if dynamic is not None:
            compile_kwargs["dynamic"] = dynamic
        return torch.compile(function, **compile_kwargs)

    @staticmethod
    def _annotate_failure(
        result: OracleResult,
        phase: str | None,
        *,
        error: BaseException | None = None,
    ) -> OracleResult:
        if result.metadata.get("kind") not in {"candidate_only_exception", "reference_only_exception"}:
            return result
        metadata = dict(result.metadata)
        metadata.update(
            failure_metadata(
                metadata.get("exception_type"),
                metadata.get("exception_message"),
                phase=phase,
                traceback_text=(
                    "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    if error is not None
                    else None
                ),
            )
        )
        fingerprint = metadata.get("failure_fingerprint") or result.fingerprint
        return OracleResult(result.interesting, fingerprint, result.score, metadata)

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
            stage = result.metadata.get("failure_stage")
            if stage in {
                "compile_wrapper",
                "dynamo_capture",
                "aot_autograd",
                "functionalization",
                "inductor_lowering",
                "code_generation",
                "triton_compile",
            } or failure_phase == "compile":
                return OutcomeClass.COMPILE_FAILURE
            if stage == "harness_timeout":
                return OutcomeClass.TIMEOUT
            return OutcomeClass.COMPILED_RUNTIME_FAILURE
        if kind == "reference_only_exception":
            return OutcomeClass.EAGER_ERROR
        if result.metadata.get("nan_mismatch") or result.metadata.get("inf_mismatch"):
            return OutcomeClass.NONFINITE_COMPARISON
        if kind.endswith("mismatch") or "mismatch" in kind:
            return OutcomeClass.GRADIENT_MISMATCH if mode == "gradient" else OutcomeClass.FORWARD_MISMATCH
        return OutcomeClass.COMPILED_RUNTIME_FAILURE
