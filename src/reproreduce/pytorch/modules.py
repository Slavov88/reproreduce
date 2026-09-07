from __future__ import annotations

import ast
from collections.abc import Callable

from ..reduce.ast import render_module
from ..reduce.ddmin import ddmin


def _is_sequential(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "Sequential":
        return False
    value = call.func.value
    return (
        isinstance(value, ast.Name) and value.id == "nn"
    ) or (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "torch"
        and value.attr == "nn"
    )


def _scope_name(call: ast.Call, index: int) -> str:
    return f"nn.Sequential[{index}]"


def reduce_sequential_modules(
    source: str,
    test: Callable[[str], bool],
) -> tuple[str, list[dict[str, object]]]:
    """Reduce positional ``nn.Sequential`` elements using delta debugging."""
    tree = ast.parse(source)
    history: list[dict[str, object]] = []
    sequential_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _is_sequential(node)]

    for index, call in enumerate(sequential_calls):
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            continue
        current = list(call.args)
        scope = _scope_name(call, index)

        def candidate_test(candidate: list[ast.expr]) -> bool:
            nonlocal current
            call.args = list(candidate)
            candidate_source = render_module(tree.body)
            ast.parse(candidate_source)
            if test(candidate_source):
                removed_count = len(current) - len(candidate)
                current = list(candidate)
                history.append(
                    {
                        "transform": "RemoveModules",
                        "scope": scope,
                        "removed_count": removed_count,
                        "remaining_modules": len(candidate),
                    }
                )
                return True
            call.args = list(current)
            return False

        reduced = ddmin(current, candidate_test)
        if reduced and candidate_test([]):
            reduced = []
        call.args = list(reduced)

    return render_module(tree.body), history
