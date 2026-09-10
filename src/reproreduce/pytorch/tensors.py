from __future__ import annotations

import ast
from collections.abc import Callable, Iterator

from ..reduce.ast import render_module
from ..reduce.scheduler import invoke_test

_CONSTRUCTORS = {"randn", "zeros", "ones", "empty"}
_DTYPE_CANDIDATES = (
    "float32",
    "float64",
    "bfloat16",
    "float16",
    "int64",
    "int32",
    "bool",
)


def _torch_constructor(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.value.id == "torch" and call.func.attr in _CONSTRUCTORS | {"tensor"}:
            return call.func.attr
    return None


def _smaller_sizes(value: int) -> list[int]:
    return [candidate for candidate in (1, 2, 3, 4, 7, 8, 16) if candidate < value]


def _walk_references(node: ast.AST) -> Iterator[tuple[ast.AST, str, int | None, ast.AST]]:
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield node, field, None, value
            yield from _walk_references(value)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, ast.AST):
                    yield node, field, index, child
                    yield from _walk_references(child)


def _replace_reference(parent: ast.AST, field: str, index: int | None, value: ast.AST) -> None:
    if index is None:
        setattr(parent, field, value)
    else:
        getattr(parent, field)[index] = value


def _record(history: list[dict[str, object]], transform: str, **details: object) -> None:
    history.append({"transform": transform, **details})


def reduce_tensor_constructors(
    source: str,
    test: Callable[[str], bool],
) -> tuple[str, list[dict[str, object]]]:
    """Try conservative shape, value, dtype, and layout changes on explicit torch calls."""
    tree = ast.parse(source)
    history: list[dict[str, object]] = []

    def candidate_source() -> str:
        return render_module(tree.body)

    def run_candidate(transform: str) -> bool:
        return invoke_test(
            test,
            candidate_source(),
            metadata={"transform": transform},
        )

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    for call in calls:
        constructor = _torch_constructor(call)
        if constructor is None:
            continue

        if constructor in _CONSTRUCTORS:
            for argument in call.args:
                if not isinstance(argument, ast.Constant) or not isinstance(argument.value, int):
                    continue
                original = argument.value
                for value in _smaller_sizes(original):
                    if value >= original:
                        continue
                    argument.value = value
                    if run_candidate("TensorShapeChange"):
                        _record(
                            history,
                            "TensorShapeChange",
                            original=original,
                            reduced=value,
                        )
                        original = value
                    else:
                        argument.value = original

        dtype_keyword = next((keyword for keyword in call.keywords if keyword.arg == "dtype"), None)
        if dtype_keyword and isinstance(dtype_keyword.value, ast.Attribute):
            dtype_value = dtype_keyword.value
            if isinstance(dtype_value.value, ast.Name) and dtype_value.value.id == "torch":
                original_dtype = dtype_value.attr
                dtype_attempts = 0
                for candidate_dtype in _DTYPE_CANDIDATES:
                    if dtype_attempts >= 2:
                        break
                    if candidate_dtype == original_dtype:
                        continue
                    dtype_attempts += 1
                    dtype_value.attr = candidate_dtype
                    if run_candidate("DtypeChange"):
                        _record(
                            history,
                            "DtypeChange",
                            original=original_dtype,
                            reduced=candidate_dtype,
                        )
                        original_dtype = candidate_dtype
                    else:
                        _record(
                            history,
                            "DtypeAttempt",
                            original=original_dtype,
                            attempted=candidate_dtype,
                            oracle="FAIL",
                        )
                        dtype_value.attr = original_dtype

        if constructor == "tensor" and call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
            values = call.args[0].elts
            if len(values) > 1:
                original_values = list(values)
                for length in range(len(original_values) - 1, 0, -1):
                    call.args[0].elts = original_values[:length]
                    if run_candidate("TensorValueListShrink"):
                        _record(history, "TensorValueListShrink", remaining=length)
                        original_values = list(call.args[0].elts)
                    else:
                        call.args[0].elts = list(original_values)
            for element in call.args[0].elts:
                if not isinstance(element, ast.Constant):
                    continue
                original_value = element.value
                for replacement in (0, 1, -1):
                    element.value = replacement
                    if run_candidate("TensorValueChange"):
                        _record(
                            history,
                            "TensorValueChange",
                            original=original_value,
                            reduced=replacement,
                        )
                        original_value = replacement
                    else:
                        element.value = original_value

        if constructor in {"randn", "empty"}:
            function = call.func
            assert isinstance(function, ast.Attribute)
            original_constructor = function.attr
            for replacement in ("zeros", "ones"):
                function.attr = replacement
                if run_candidate("TensorConstructorChange"):
                    _record(
                        history,
                        "TensorConstructorChange",
                        original=original_constructor,
                        reduced=replacement,
                    )
                    original_constructor = replacement
                    break
                else:
                    function.attr = original_constructor

    # A contiguous conversion is intentionally attempted only as a diagnostic.
    # If non-contiguity is required, the oracle rejects it and the transpose remains.
    for parent, field, index, node in list(_walk_references(tree)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "transpose"):
            continue
        candidate = ast.Call(
            func=ast.Attribute(value=node, attr="contiguous"),
            args=[],
            keywords=[],
        )
        _replace_reference(parent, field, index, candidate)
        if run_candidate("LayoutChange"):
            _record(history, "LayoutChange", original="noncontiguous", reduced="contiguous")
        else:
            _record(
                history,
                "LayoutAttempt",
                original="noncontiguous",
                attempted="contiguous",
                oracle="FAIL",
            )
            _replace_reference(parent, field, index, node)

    return candidate_source(), history
