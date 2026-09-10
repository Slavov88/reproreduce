from __future__ import annotations

import ast
import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Iterable

from .dependency import analyze_source, reduce_dependency
from .scheduler import invoke_test
from .ast import render_module


NodeKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class V2Candidate:
    phase: str
    label: str
    action: str
    target: NodeKey
    related: tuple[NodeKey, ...] = ()
    index: int | None = None
    name: str | None = None
    mode: str | None = None


def _key(node: ast.AST) -> NodeKey:
    return (
        int(getattr(node, "lineno", -1)),
        int(getattr(node, "col_offset", -1)),
        int(getattr(node, "end_lineno", getattr(node, "lineno", -1))),
        int(getattr(node, "end_col_offset", getattr(node, "col_offset", -1))),
    )


def _nonblank(source: str) -> int:
    return sum(bool(line.strip()) for line in source.splitlines())


def _load_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def _statement_lists(tree: ast.AST) -> Iterable[list[ast.stmt]]:
    for owner in ast.walk(tree):
        for _, value in ast.iter_fields(owner):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                yield value


def _find_by_key(tree: ast.AST, key: NodeKey) -> ast.AST | None:
    return next((node for node in ast.walk(tree) if _key(node) == key), None)


def _function_candidates(source: str) -> list[V2Candidate]:
    tree = ast.parse(source)
    output: list[V2Candidate] = []
    parent: dict[ast.AST, ast.AST] = {}
    for owner in ast.walk(tree):
        for child in ast.iter_child_nodes(owner):
            parent[child] = owner

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        if function.decorator_list or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
            continue
        if any(isinstance(parent.get(function), ast.ClassDef) for _ in [0]):
            continue
        positional = list(function.args.posonlyargs) + list(function.args.args)
        if not positional or function.args.defaults:
            continue
        params = [argument.arg for argument in positional]
        body_loads = _load_names(ast.Module(body=function.body, type_ignores=[]))
        unused = [index for index, name in enumerate(params) if name not in body_loads]
        if not unused:
            continue

        calls: list[tuple[ast.Call, str]] = []
        escaped = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function.name:
                if len(node.args) == len(params) and not node.keywords:
                    calls.append((node, "positional"))
                elif not node.args and len(node.keywords) == len(params) and all(keyword.arg in params for keyword in node.keywords):
                    calls.append((node, "keyword"))
                else:
                    escaped = True
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == function.name:
                parent_node = parent.get(node)
                if not (isinstance(parent_node, ast.Call) and parent_node.func is node):
                    escaped = True
        if escaped or not calls:
            continue
        if len({mode for _, mode in calls}) != 1:
            continue
        mode = calls[0][1]
        for index in unused:
            output.append(
                V2Candidate(
                    phase="parameter_call",
                    label=f"remove_parameter:{function.name}:{params[index]}",
                    action="remove_parameter",
                    target=_key(function),
                    related=tuple(_key(call) for call, _ in calls),
                    index=index,
                    name=params[index],
                    mode=mode,
                )
            )
    return sorted(output, key=lambda candidate: candidate.label)


def _replace_parameter_call(source: str, candidate: V2Candidate) -> str:
    tree = ast.parse(source)
    call_keys = set(candidate.related)
    function_key = candidate.target
    index = int(candidate.index or 0)

    class Transformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
            if _key(node) == function_key:
                positional = list(node.args.posonlyargs) + list(node.args.args)
                target = positional[index]
                if target in node.args.posonlyargs:
                    node.args.posonlyargs = [item for item in node.args.posonlyargs if item is not target]
                else:
                    node.args.args = [item for item in node.args.args if item is not target]
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # type: ignore[override]
            if _key(node) == function_key:
                positional = list(node.args.posonlyargs) + list(node.args.args)
                target = positional[index]
                if target in node.args.posonlyargs:
                    node.args.posonlyargs = [item for item in node.args.posonlyargs if item is not target]
                else:
                    node.args.args = [item for item in node.args.args if item is not target]
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call):  # type: ignore[override]
            if _key(node) in call_keys:
                if candidate.mode == "positional":
                    node.args = [argument for position, argument in enumerate(node.args) if position != index]
                else:
                    node.keywords = [keyword for keyword in node.keywords if keyword.arg != candidate.name]
            return self.generic_visit(node)

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _control_flow_candidates(source: str) -> list[V2Candidate]:
    tree = ast.parse(source)
    output: list[V2Candidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if node.body:
                output.append(V2Candidate("control_flow", f"if_body:{_key(node)}", "branch_body", _key(node)))
            if node.orelse:
                output.append(V2Candidate("control_flow", f"if_orelse:{_key(node)}", "branch_orelse", _key(node)))
        elif isinstance(node, ast.Try) and not node.finalbody:
            if node.body:
                output.append(V2Candidate("control_flow", f"try_body:{_key(node)}", "try_body", _key(node)))
            for index, handler in enumerate(node.handlers):
                if handler.body:
                    output.append(V2Candidate("control_flow", f"try_handler:{_key(node)}:{index}", "try_handler", _key(node), index=index))
        elif isinstance(node, (ast.For, ast.AsyncFor)) and node.body:
            target_names = {child.id for child in ast.walk(node.target) if isinstance(child, ast.Name)}
            if not target_names.intersection(_load_names(ast.Module(body=node.body, type_ignores=[]))) and not any(
                isinstance(child, (ast.Break, ast.Continue, ast.Yield, ast.YieldFrom)) for child in ast.walk(node)
            ):
                output.append(V2Candidate("control_flow", f"for_body:{_key(node)}", "for_body", _key(node)))
    return sorted(output, key=lambda candidate: candidate.label)


def _replace_statement(source: str, candidate: V2Candidate) -> str:
    tree = ast.parse(source)

    class Transformer(ast.NodeTransformer):
        def visit(self, node: ast.AST):  # type: ignore[override]
            if _key(node) == candidate.target:
                if candidate.action == "branch_body" and isinstance(node, ast.If):
                    return [copy.deepcopy(item) for item in node.body]
                if candidate.action == "branch_orelse" and isinstance(node, ast.If):
                    return [copy.deepcopy(item) for item in node.orelse]
                if candidate.action == "try_body" and isinstance(node, ast.Try):
                    return [copy.deepcopy(item) for item in node.body]
                if candidate.action == "try_handler" and isinstance(node, ast.Try):
                    return [copy.deepcopy(item) for item in node.handlers[int(candidate.index or 0)].body]
                if candidate.action == "for_body" and isinstance(node, (ast.For, ast.AsyncFor)):
                    return [copy.deepcopy(item) for item in node.body]
            return super().visit(node)

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _call_result_candidates(source: str) -> list[V2Candidate]:
    tree = ast.parse(source)
    output: list[V2Candidate] = []
    for statements in _statement_lists(tree):
        for first, second in zip(statements, statements[1:]):
            if not isinstance(first, ast.Assign) or len(first.targets) != 1 or not isinstance(first.targets[0], ast.Name):
                continue
            name = first.targets[0].id
            if isinstance(second, ast.Return) and isinstance(second.value, ast.Name) and second.value.id == name:
                output.append(V2Candidate("call_result", f"inline_return:{_key(first)}:{_key(second)}", "inline_return", _key(first), related=(_key(second),), name=name))
            elif isinstance(second, ast.Expr) and isinstance(second.value, ast.Call):
                if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name for node in ast.walk(second.value)):
                    output.append(V2Candidate("call_result", f"inline_call:{_key(first)}:{_key(second)}", "inline_call", _key(first), related=(_key(second),), name=name))
            elif isinstance(second, ast.Assign) and any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name for node in ast.walk(second.value)):
                output.append(V2Candidate("call_result", f"inline_assignment:{_key(first)}:{_key(second)}", "inline_assignment", _key(first), related=(_key(second),), name=name))
    return sorted(output, key=lambda candidate: candidate.label)


def _replace_load_names(node: ast.AST, name: str, replacement: ast.AST) -> None:
    class Replacer(ast.NodeTransformer):
        def visit_Name(self, current: ast.Name):  # type: ignore[override]
            if current.id == name and isinstance(current.ctx, ast.Load):
                return ast.copy_location(copy.deepcopy(replacement), current)
            return current

    Replacer().visit(node)


def _inline_call_result(source: str, candidate: V2Candidate) -> str:
    tree = ast.parse(source)
    first_key = candidate.target
    second_key = candidate.related[0]
    first: ast.Assign | None = None
    second: ast.stmt | None = None
    for node in ast.walk(tree):
        if _key(node) == first_key and isinstance(node, ast.Assign):
            first = node
        if _key(node) == second_key and isinstance(node, ast.stmt):
            second = node
    if first is None or second is None or not isinstance(first.targets[0], ast.Name):
        return source
    replacement = first.value
    name = first.targets[0].id
    for statements in _statement_lists(tree):
        if first in statements and second in statements:
            statements.remove(first)
            if candidate.action == "inline_return" and isinstance(second, ast.Return):
                second.value = copy.deepcopy(replacement)
            else:
                _replace_load_names(second, name, replacement)
            break
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _expression_candidates(source: str) -> list[V2Candidate]:
    tree = ast.parse(source)
    output: list[V2Candidate] = []
    for node in ast.walk(tree):
        key = _key(node)
        if isinstance(node, ast.BinOp):
            output.extend([
                V2Candidate("expression", f"binop_left:{key}", "expr_left", key),
                V2Candidate("expression", f"binop_right:{key}", "expr_right", key),
            ])
        elif isinstance(node, ast.BoolOp) and len(node.values) > 1:
            for index in range(len(node.values)):
                output.append(V2Candidate("expression", f"bool_value:{key}:{index}", "bool_value", key, index=index))
        elif isinstance(node, ast.IfExp):
            output.extend([
                V2Candidate("expression", f"ifexp_body:{key}", "expr_body", key),
                V2Candidate("expression", f"ifexp_else:{key}", "expr_orelse", key),
            ])
        elif isinstance(node, ast.Compare):
            output.extend([
                V2Candidate("expression", f"compare_left:{key}", "compare_left", key),
                V2Candidate("expression", f"compare_right:{key}", "compare_right", key, index=0),
            ])
        elif isinstance(node, ast.Subscript):
            output.append(V2Candidate("expression", f"subscript_value:{key}", "subscript_value", key))
        elif isinstance(node, ast.UnaryOp):
            output.append(V2Candidate("expression", f"unary_operand:{key}", "unary_operand", key))
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts:
            for index in range(len(node.elts)):
                output.append(V2Candidate("expression", f"{type(node).__name__.lower()}_remove:{key}:{index}", "sequence_remove", key, index=index))
        elif isinstance(node, ast.Dict) and node.values:
            for index in range(len(node.values)):
                output.append(V2Candidate("expression", f"dict_remove:{key}:{index}", "dict_remove", key, index=index))
        elif isinstance(node, ast.Return) and isinstance(node.value, (ast.List, ast.Tuple)) and node.value.elts:
            for index in range(len(node.value.elts)):
                output.append(V2Candidate("expression", f"return_element:{key}:{index}", "return_element", key, index=index))
    return sorted(output, key=lambda candidate: candidate.label)


def _replace_expression(source: str, candidate: V2Candidate) -> str:
    tree = ast.parse(source)
    index = int(candidate.index or 0)

    class Transformer(ast.NodeTransformer):
        def visit(self, node: ast.AST):  # type: ignore[override]
            if _key(node) == candidate.target:
                if candidate.action in {"expr_left", "expr_right"} and isinstance(node, ast.BinOp):
                    return ast.copy_location(copy.deepcopy(node.left if candidate.action == "expr_left" else node.right), node)
                if candidate.action == "bool_value" and isinstance(node, ast.BoolOp):
                    return ast.copy_location(copy.deepcopy(node.values[index]), node)
                if candidate.action == "expr_body" and isinstance(node, ast.IfExp):
                    return ast.copy_location(copy.deepcopy(node.body), node)
                if candidate.action == "expr_orelse" and isinstance(node, ast.IfExp):
                    return ast.copy_location(copy.deepcopy(node.orelse), node)
                if candidate.action == "compare_left" and isinstance(node, ast.Compare):
                    return ast.copy_location(copy.deepcopy(node.left), node)
                if candidate.action == "compare_right" and isinstance(node, ast.Compare):
                    return ast.copy_location(copy.deepcopy(node.comparators[index]), node)
                if candidate.action == "subscript_value" and isinstance(node, ast.Subscript):
                    return ast.copy_location(copy.deepcopy(node.value), node)
                if candidate.action == "unary_operand" and isinstance(node, ast.UnaryOp):
                    return ast.copy_location(copy.deepcopy(node.operand), node)
                if candidate.action == "sequence_remove" and isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    node.elts = [item for position, item in enumerate(node.elts) if position != index]
                    return node
                if candidate.action == "dict_remove" and isinstance(node, ast.Dict):
                    node.keys = [item for position, item in enumerate(node.keys) if position != index]
                    node.values = [item for position, item in enumerate(node.values) if position != index]
                    return node
                if candidate.action == "return_element" and isinstance(node, ast.Return) and isinstance(node.value, (ast.List, ast.Tuple)):
                    return ast.copy_location(ast.Return(value=copy.deepcopy(node.value.elts[index])), node)
            return super().visit(node)

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def apply_candidate(source: str, candidate: V2Candidate) -> str:
    if candidate.phase == "parameter_call":
        return _replace_parameter_call(source, candidate)
    if candidate.phase == "control_flow":
        return _replace_statement(source, candidate)
    if candidate.phase == "call_result":
        return _inline_call_result(source, candidate)
    return _replace_expression(source, candidate)


def _run_phase(current: str, phase: str, generator, test, history, record_candidate, max_accepts: int = 24) -> tuple[str, bool]:
    changed = False
    for _ in range(max_accepts):
        candidates = generator(current)
        seen: set[str] = set()
        accepted = False
        for candidate in candidates:
            candidate_source = apply_candidate(current, candidate)
            digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
            if candidate_source == current or digest in seen:
                continue
            seen.add(digest)
            try:
                ast.parse(candidate_source)
            except SyntaxError:
                record_candidate(phase, False, current, candidate_source, 0)
                continue
            session = getattr(test, "session", None)
            before_runs = int(getattr(session, "_candidate_runs", 0))
            outcome = invoke_test(
                test,
                candidate_source,
                metadata={"transform": "DependencyV2", "dependency_phase": phase, "label": candidate.label},
            )
            oracle_runs = int(getattr(session, "_candidate_runs", 0)) - before_runs
            record_candidate(phase, outcome, current, candidate_source, oracle_runs)
            if outcome:
                history.append({"transform": "DependencyV2", "phase": phase, "label": candidate.label})
                current = candidate_source
                changed = True
                accepted = True
                break
        if not accepted:
            break
    return current, changed


def reduce_dependency_v2(
    source: str,
    test,
    history: list[dict[str, object]],
    record_candidate,
    record_v2_candidate,
    record_analysis=None,
    record_phase_elapsed=None,
    *,
    max_rounds: int = 4,
) -> tuple[str, list[dict[str, object]]]:
    current = source
    for _ in range(max_rounds):
        round_changed = False
        cleaned, clean_history = reduce_dependency(
            current,
            test,
            [],
            record_candidate,
            record_analysis=record_analysis,
        )
        history.extend(clean_history)
        if cleaned != current:
            current = cleaned
            round_changed = True
        for phase, generator in (
            ("parameter_call", _function_candidates),
            ("call_result", _call_result_candidates),
            ("control_flow", _control_flow_candidates),
            ("expression", _expression_candidates),
        ):
            started = time.perf_counter()
            current, changed = _run_phase(
                current,
                phase,
                generator,
                test,
                history,
                record_v2_candidate,
            )
            if callable(record_phase_elapsed):
                record_phase_elapsed(phase, time.perf_counter() - started)
            round_changed = round_changed or changed
        if not round_changed:
            break
    return current, history
