from __future__ import annotations

import ast
from collections.abc import Callable

from ..reduce.ast import render_module
from ..reduce.ddmin import ddmin


def _container_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return None
    if call.func.value.id != "nn" or call.func.attr not in {"Sequential", "ModuleList"}:
        return None
    return call.func.attr


def _target_key(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return ast.unparse(target)
    return None


def _assigned_target(tree: ast.Module, call: ast.Call) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if node.value is call and len(node.targets) == 1:
                return _target_key(node.targets[0])
        elif isinstance(node, ast.AnnAssign) and node.value is call:
            return _target_key(node.target)
    return None


def _matches_expression(node: ast.AST, key: str) -> bool:
    try:
        return ast.unparse(node) == key
    except (AttributeError, ValueError):
        return False


def _has_static_index(tree: ast.Module, key: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _matches_expression(node.value, key):
            return True
    return False


def _is_iterated(tree: ast.Module, key: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)) and _matches_expression(node.iter, key):
            return True
    return False


def _reduce_module_collection(
    tree: ast.Module,
    call: ast.Call,
    scope: str,
    test: Callable[[str], bool],
    history: list[dict[str, object]],
    *,
    list_literal: ast.List | ast.Tuple | None = None,
) -> None:
    current = list(list_literal.elts if list_literal is not None else call.args)

    def set_elements(elements: list[ast.expr]) -> None:
        if list_literal is not None:
            list_literal.elts = list(elements)
        else:
            call.args = list(elements)

    def candidate_test(candidate: list[ast.expr]) -> bool:
        nonlocal current
        set_elements(candidate)
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
        set_elements(current)
        return False

    reduced = ddmin(current, candidate_test)
    if reduced and candidate_test([]):
        reduced = []
    set_elements(reduced)


def reduce_sequential_modules(
    source: str,
    test: Callable[[str], bool],
) -> tuple[str, list[dict[str, object]]]:
    """Reduce positional ``nn.Sequential`` elements using delta debugging."""
    tree = ast.parse(source)
    history: list[dict[str, object]] = []
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _container_name(node) == "Sequential"]

    for index, call in enumerate(calls):
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            continue
        _reduce_module_collection(tree, call, f"nn.Sequential[{index}]", test, history)

    return render_module(tree.body), history


def reduce_module_lists(
    source: str,
    test: Callable[[str], bool],
) -> tuple[str, list[dict[str, object]]]:
    """Reduce only iteration-based ``nn.ModuleList`` containers.

    Statically indexed lists are left unchanged because removing an element
    would change the meaning of every later index.
    """
    tree = ast.parse(source)
    history: list[dict[str, object]] = []
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _container_name(node) == "ModuleList"]

    for index, call in enumerate(calls):
        if any(isinstance(argument, ast.Starred) for argument in call.args):
            continue
        key = _assigned_target(tree, call)
        if key is None or _has_static_index(tree, key) or not _is_iterated(tree, key):
            continue
        if len(call.args) != 1 or not isinstance(call.args[0], (ast.List, ast.Tuple)):
            continue
        _reduce_module_collection(
            tree,
            call,
            f"nn.ModuleList[{index}]",
            test,
            history,
            list_literal=call.args[0],
        )

    return render_module(tree.body), history
