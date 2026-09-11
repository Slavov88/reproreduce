from __future__ import annotations

import ast
import copy
import hashlib
import statistics
import time
from dataclasses import dataclass

from .ast import render_module
from .dependency import reduce_dependency
from .dependency_v2 import (
    V2Candidate,
    _call_result_candidates,
    _control_flow_candidates,
    _function_candidates,
    _expression_candidates,
    apply_candidate as apply_v2_candidate,
)
from .scheduler import invoke_test


NodeKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class V3Candidate:
    phase: str
    label: str
    action: str
    target: NodeKey
    related: tuple[NodeKey, ...] = ()
    name: str | None = None
    replacement: ast.expr | None = None
    parameters: tuple[str, ...] = ()


def _key(node: ast.AST) -> NodeKey:
    return (
        int(getattr(node, "lineno", -1)),
        int(getattr(node, "col_offset", -1)),
        int(getattr(node, "end_lineno", getattr(node, "lineno", -1))),
        int(getattr(node, "end_col_offset", getattr(node, "col_offset", -1))),
    )


def _load_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def _statement_lists(tree: ast.AST):
    for owner in ast.walk(tree):
        for _, value in ast.iter_fields(owner):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                yield value


def _literal(node: ast.AST) -> ast.expr | None:
    if isinstance(node, ast.Constant):
        return node
    if isinstance(node, (ast.Tuple, ast.List)) and all(_literal(item) is not None for item in node.elts):
        return node
    return None


def _nearest_function(parents: dict[ast.AST, ast.AST], node: ast.AST) -> ast.AST | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}
    for owner in ast.walk(tree):
        for child in ast.iter_child_nodes(owner):
            result[child] = owner
    return result


def _config_models(tree: ast.Module) -> dict[str, dict[str, ast.expr]]:
    models: dict[str, dict[str, ast.expr]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.bases or any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for item in node.body):
            continue
        if not any(isinstance(decorator, ast.Name) and decorator.id == "dataclass" for decorator in node.decorator_list):
            continue
        fields: dict[str, ast.expr] = {}
        valid = True
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name) or item.value is None:
                valid = False
                break
            value = _literal(item.value)
            if value is None:
                valid = False
                break
            fields[item.target.id] = copy.deepcopy(value)
        if valid and fields:
            models[node.name] = fields
    return models


def _function_model_bindings(tree: ast.Module, models: dict[str, dict[str, ast.expr]]) -> dict[NodeKey, dict[str, dict[str, ast.expr]]]:
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    by_name = {node.name: node for node in functions if not node.decorator_list and not node.args.vararg and not node.args.kwarg}
    bindings: dict[NodeKey, dict[str, dict[str, ast.expr]]] = { _key(node): {} for node in functions }
    for function in functions:
        for item in function.body:
            if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                value = item.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in models and not value.args and not value.keywords:
                    bindings[_key(function)][item.targets[0].id] = models[value.func.id]
    changed = True
    while changed:
        changed = False
        for function in functions:
            params = [arg.arg for arg in function.args.posonlyargs + function.args.args]
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != function.name:
                    continue
                if node.keywords or len(node.args) != len(params):
                    continue
                for param, argument in zip(params, node.args):
                    model = None
                    if isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name) and argument.func.id in models and not argument.args and not argument.keywords:
                        model = models[argument.func.id]
                    elif isinstance(argument, ast.Name):
                        caller = next((candidate for candidate in functions if any(child is node for child in ast.walk(candidate))), None)
                        if caller is not None:
                            model = bindings.get(_key(caller), {}).get(argument.id)
                    if model is not None and bindings[_key(function)].get(param) != model:
                        bindings[_key(function)][param] = model
                        changed = True
    return bindings


def _config_candidates(source: str) -> list[V3Candidate]:
    tree = ast.parse(source)
    if not isinstance(tree, ast.Module):
        return []
    models = _config_models(tree)
    if not models:
        return []
    parents = _parents(tree)
    bindings = _function_model_bindings(tree, models)
    local_function_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    blocked: set[tuple[NodeKey, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id not in {name for values in bindings.values() for name in values}:
            continue
        function = _nearest_function(parents, node)
        field_map = bindings.get(_key(function), {}).get(node.id) if function is not None else None
        if field_map is None:
            continue
        parent = parents.get(node)
        if isinstance(node.ctx, ast.Store) and not (isinstance(parent, ast.Assign) and parent.value is not node):
            blocked.add((_key(function), node.id))
        if isinstance(parent, ast.Attribute) and parent.value is node and isinstance(node.ctx, ast.Load):
            if isinstance(parent.ctx, ast.Store):
                blocked.add((_key(function), node.id))
            continue
        if isinstance(parent, ast.Call) and node in parent.args and isinstance(parent.func, ast.Name) and parent.func.id in local_function_names:
            continue
        if isinstance(parent, ast.Call) or isinstance(parent, ast.Return):
            blocked.add((_key(function), node.id))
    output: list[V3Candidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        function = _nearest_function(parents, node)
        field_map = bindings.get(_key(function), {}).get(node.value.id) if function is not None else None
        if field_map is None or (_key(function), node.value.id) in blocked:
            continue
        replacement = field_map.get(node.attr)
        if replacement is not None:
            output.append(V3Candidate("config", f"config_field:{node.value.id}.{node.attr}:{_key(node)}", "replace_expression", _key(node), replacement=copy.deepcopy(replacement)))
    return sorted(output, key=lambda item: item.label)


def _assignment_candidates(source: str) -> list[V3Candidate]:
    tree = ast.parse(source)
    parents = _parents(tree)
    output: list[V3Candidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        replacement: ast.expr | None = None
        action = "alias"
        if isinstance(node.value, ast.Name) and node.value.id != name:
            replacement = ast.Name(id=node.value.id, ctx=ast.Load())
        else:
            replacement = _literal(node.value)
            action = "constant"
        if replacement is None:
            continue
        function = _nearest_function(parents, node)
        scope_key = _key(function) if function is not None else None
        uses = []
        rebinding = False
        for candidate in ast.walk(tree):
            if not isinstance(candidate, ast.Name) or candidate.id != name:
                continue
            if isinstance(candidate.ctx, ast.Store) and getattr(candidate, "lineno", -1) > node.lineno:
                rebinding = True
            if isinstance(candidate.ctx, ast.Load) and candidate.lineno > node.lineno and _key(_nearest_function(parents, candidate) or tree) == (scope_key or _key(tree)):
                uses.append(candidate)
        if rebinding or len(uses) != 1:
            continue
        output.append(V3Candidate("alias" if action == "alias" else "constant", f"{action}:{name}:{_key(node)}", "replace_assignment", _key(node), name=name, replacement=copy.deepcopy(replacement)))
    return sorted(output, key=lambda item: item.label)


def _literal_lookup_candidates(source: str) -> list[V3Candidate]:
    tree = ast.parse(source)
    output: list[V3Candidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        index = node.slice
        replacement: ast.expr | None = None
        if isinstance(value, (ast.Tuple, ast.List)) and isinstance(index, ast.Constant) and isinstance(index.value, int):
            if 0 <= index.value < len(value.elts):
                replacement = copy.deepcopy(value.elts[index.value])
        elif isinstance(value, ast.Dict) and isinstance(index, ast.Constant):
            for key, item in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and key.value == index.value:
                    replacement = copy.deepcopy(item)
                    break
        if replacement is not None:
            output.append(V3Candidate("literal", f"literal_lookup:{_key(node)}", "replace_expression", _key(node), replacement=replacement))
    return sorted(output, key=lambda item: item.label)


def _wrapper_candidates(source: str) -> list[V3Candidate]:
    tree = ast.parse(source)
    parents = _parents(tree)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    output: list[V3Candidate] = []
    for function in functions:
        if function.decorator_list or function.args.args or function.args.posonlyargs or function.args.kwonlyargs or function.args.vararg or function.args.kwarg:
            continue
        if any(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith, ast.FunctionDef, ast.ClassDef)) for node in function.body):
            continue
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function.name and not node.args and not node.keywords]
        if len(calls) != 1:
            continue
        call = calls[0]
        call_statement = parents.get(call)
        if (
            isinstance(call_statement, ast.Expr)
            and len(function.body) > 0
            and all(isinstance(node, ast.stmt) for node in function.body)
            and not any(isinstance(node, (ast.Return, ast.Break, ast.Continue)) for node in function.body)
        ):
            output.append(V3Candidate("wrapper", f"flatten_wrapper:{function.name}:{_key(call)}", "flatten_wrapper", _key(function), related=(_key(call_statement),)))
        if len(function.body) == 1 and isinstance(function.body[0], ast.Return) and function.body[0].value is not None:
            output.append(V3Candidate("wrapper", f"inline_wrapper:{function.name}:{_key(call)}", "inline_wrapper", _key(function), related=(_key(call),), replacement=copy.deepcopy(function.body[0].value)))
    return sorted(output, key=lambda item: item.label)


def _guard_candidates(source: str) -> list[V3Candidate]:
    tree = ast.parse(source)
    output: list[V3Candidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or len(node.body) != 1 or node.orelse:
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq) or not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__":
            output.append(V3Candidate("main_guard", f"main_guard:{_key(node)}", "flatten_guard", _key(node)))
    return sorted(output, key=lambda item: item.label)


def _replace_expression(source: str, candidate: V3Candidate) -> str:
    tree = ast.parse(source)

    class Transformer(ast.NodeTransformer):
        def visit(self, node: ast.AST):  # type: ignore[override]
            if _key(node) == candidate.target and candidate.action == "replace_expression" and candidate.replacement is not None:
                return ast.copy_location(copy.deepcopy(candidate.replacement), node)
            return super().visit(node)

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _replace_assignment(source: str, candidate: V3Candidate) -> str:
    tree = ast.parse(source)
    assignment = next((node for node in ast.walk(tree) if isinstance(node, ast.Assign) and _key(node) == candidate.target), None)
    if not isinstance(assignment, ast.Assign) or not isinstance(assignment.targets[0], ast.Name) or candidate.replacement is None:
        return source
    target_name = assignment.targets[0].id
    parents = _parents(tree)
    target_function = _nearest_function(parents, assignment)

    class Replacer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):  # type: ignore[override]
            if node.id == target_name and isinstance(node.ctx, ast.Load) and node.lineno > assignment.lineno:
                if _nearest_function(parents, node) is target_function:
                    return ast.copy_location(copy.deepcopy(candidate.replacement), node)
            return node

    Replacer().visit(tree)
    for statements in _statement_lists(tree):
        for index, statement in enumerate(statements):
            if statement is assignment:
                statements.pop(index)
                break
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _flatten_statement(source: str, candidate: V3Candidate, body: list[ast.stmt]) -> str:
    tree = ast.parse(source)
    class Transformer(ast.NodeTransformer):
        def visit(self, node: ast.AST):  # type: ignore[override]
            if _key(node) == candidate.target:
                return [copy.deepcopy(item) for item in body]
            return super().visit(node)
    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _apply_wrapper(source: str, candidate: V3Candidate) -> str:
    tree = ast.parse(source)
    function = next((node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _key(node) == candidate.target), None)
    call_key = candidate.related[0] if candidate.related else None
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or call_key is None:
        return source
    if candidate.action == "flatten_wrapper":
        return _flatten_statement(source, V3Candidate(candidate.phase, candidate.label, candidate.action, call_key), function.body)
    if candidate.action == "inline_wrapper" and candidate.replacement is not None:
        class Transformer(ast.NodeTransformer):
            def visit(self, node: ast.AST):  # type: ignore[override]
                if _key(node) == call_key:
                    replacement = copy.deepcopy(candidate.replacement)
                    if isinstance(node, ast.Expr):
                        return ast.copy_location(ast.Expr(value=replacement), node)
                    return ast.copy_location(replacement, node)
                return super().visit(node)
        tree = Transformer().visit(tree)
        ast.fix_missing_locations(tree)
        return render_module(tree.body)
    return source


def _apply_guard(source: str, candidate: V3Candidate) -> str:
    tree = ast.parse(source)
    node = next((item for item in ast.walk(tree) if isinstance(item, ast.If) and _key(item) == candidate.target), None)
    if not isinstance(node, ast.If):
        return source
    return _flatten_statement(source, V3Candidate(candidate.phase, candidate.label, candidate.action, candidate.target), node.body)


def _apply_candidate(source: str, candidate) -> str:
    if isinstance(candidate, V2Candidate):
        return apply_v2_candidate(source, candidate)
    if candidate.action == "replace_expression":
        return _replace_expression(source, candidate)
    if candidate.action == "replace_assignment":
        return _replace_assignment(source, candidate)
    if candidate.action == "flatten_wrapper":
        return _apply_wrapper(source, candidate)
    if candidate.action == "inline_wrapper":
        return _apply_wrapper(source, candidate)
    if candidate.action == "flatten_guard":
        return _apply_guard(source, candidate)
    return source


def _run_phase(current: str, phase: str, generator, test, history, record_candidate, max_candidates: int | None = None) -> tuple[str, bool]:
    changed = False
    candidates_seen = 0
    while True:
        candidates = generator(current)
        accepted = False
        seen: set[str] = set()
        for candidate in candidates:
            session = getattr(test, "session", None)
            if phase == "expression" and session is not None:
                budget = int(getattr(session, "_v3_expression_budget", 0) or 0)
                if budget and int(getattr(session, "_v3_expression_candidates_seen", 0)) >= budget:
                    return current, changed
                session._v3_expression_candidates_seen = int(getattr(session, "_v3_expression_candidates_seen", 0)) + 1
            if max_candidates is not None and candidates_seen >= max_candidates:
                return current, changed
            candidate_source = _apply_candidate(current, candidate)
            digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
            if candidate_source == current or digest in seen:
                continue
            seen.add(digest)
            candidates_seen += 1
            try:
                ast.parse(candidate_source)
            except SyntaxError:
                record_candidate(phase, False, current, candidate_source, 0)
                continue
            before_runs = int(getattr(session, "_candidate_runs", 0))
            outcome = invoke_test(test, candidate_source, metadata={"transform": "DependencyV3", "dependency_phase": phase, "label": candidate.label})
            oracle_runs = int(getattr(session, "_candidate_runs", 0)) - before_runs
            record_candidate(phase, outcome, current, candidate_source, oracle_runs)
            if outcome:
                history.append({"transform": "DependencyV3", "phase": phase, "label": candidate.label})
                current = candidate_source
                changed = True
                accepted = True
                break
        if not accepted:
            return current, changed


def reduce_plumbing_v3(source: str, test, history: list[dict[str, object]], record_dependency_candidate, record_v3_candidate, record_analysis=None, record_phase_elapsed=None, *, max_rounds: int = 4) -> tuple[str, list[dict[str, object]]]:
    current = source
    for _ in range(max_rounds):
        round_changed = False
        cleaned, clean_history = reduce_dependency(current, test, [], record_dependency_candidate, record_analysis=record_analysis)
        history.extend(clean_history)
        if cleaned != current:
            current = cleaned
            round_changed = True
        phases = (
            ("parameter_call", _function_candidates, None),
            ("config", _config_candidates, None),
            ("alias", _assignment_candidates, None),
            ("literal", _literal_lookup_candidates, None),
            ("call_result", _call_result_candidates, None),
            ("wrapper", _wrapper_candidates, None),
            ("main_guard", _guard_candidates, None),
            ("control_flow", _control_flow_candidates, None),
        )
        for phase, generator, _ in phases:
            started = time.perf_counter()
            current, changed = _run_phase(current, phase, generator, test, history, record_v3_candidate)
            if callable(record_phase_elapsed):
                record_phase_elapsed(phase, time.perf_counter() - started)
            round_changed = round_changed or changed
        # Expression search is capped when recent oracle calls are expensive.
        session = getattr(test, "session", None)
        durations = list(getattr(session, "_candidate_durations", []))
        median = statistics.median(durations[-8:]) if durations else 0.0
        expression_budget = 16 if median >= 2.0 else 128
        if session is not None:
            session._v3_expression_budget = expression_budget
        started = time.perf_counter()
        current, changed = _run_phase(current, "expression", _expression_candidates, test, history, record_v3_candidate, max_candidates=expression_budget)
        if callable(record_phase_elapsed):
            record_phase_elapsed("expression", time.perf_counter() - started)
        round_changed = round_changed or changed
        if not round_changed:
            break
    return current, history
