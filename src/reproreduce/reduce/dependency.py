from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

from .ast import render_module
from .scheduler import invoke_test


NodeKey = tuple[int, int, int, int]


@dataclass(frozen=True)
class StatementInfo:
    key: NodeKey
    scope: str
    kind: str
    defines: frozenset[str]
    uses: frozenset[str]
    call_names: frozenset[str]
    side_effect_risk: str


@dataclass
class DependencyAnalysis:
    statements: list[StatementInfo] = field(default_factory=list)
    dependencies: dict[NodeKey, set[NodeKey]] = field(default_factory=dict)
    direct_calls: dict[str, set[str]] = field(default_factory=dict)
    dynamic_features: bool = False
    component_count: int = 0


@dataclass(frozen=True)
class DependencyCandidate:
    kind: str
    label: str
    node_keys: tuple[NodeKey, ...] = ()
    alias: str | None = None
    expression_action: str | None = None


def node_key(node: ast.AST) -> NodeKey:
    return (
        int(getattr(node, "lineno", -1)),
        int(getattr(node, "col_offset", -1)),
        int(getattr(node, "end_lineno", getattr(node, "lineno", -1))),
        int(getattr(node, "end_col_offset", getattr(node, "col_offset", -1))),
    )


def _names_in(node: ast.AST, context: ast.expr_context | None = None) -> tuple[set[str], set[str], set[str]]:
    defines: set[str] = set()
    uses: set[str] = set()
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                uses.add(child.id)
            elif isinstance(child.ctx, (ast.Store, ast.Del)):
                defines.add(child.id)
        elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            calls.add(child.func.id)
    return defines, uses, calls


def _statement_info(statement: ast.stmt, scope: str) -> StatementInfo:
    key = node_key(statement)
    kind = type(statement).__name__
    defines: set[str] = set()
    uses: set[str] = set()
    calls: set[str] = set()
    risk = "low"

    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defines.add(statement.name)
        decorator_and_signature: list[ast.AST] = list(getattr(statement, "decorator_list", ()))
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorator_and_signature.extend(statement.args.defaults)
            decorator_and_signature.extend(statement.args.kw_defaults)
            if statement.returns is not None:
                decorator_and_signature.append(statement.returns)
        else:
            decorator_and_signature.extend(statement.bases)
            decorator_and_signature.extend(keyword.value for keyword in statement.keywords)
        for part in decorator_and_signature:
            _, part_uses, part_calls = _names_in(part)
            uses.update(part_uses)
            calls.update(part_calls)
        risk = "medium" if decorator_and_signature else "low"
    elif isinstance(statement, ast.Import):
        for alias in statement.names:
            defines.add(alias.asname or alias.name.split(".", 1)[0])
    elif isinstance(statement, ast.ImportFrom):
        for alias in statement.names:
            if alias.name != "*":
                defines.add(alias.asname or alias.name)
            else:
                risk = "high"
    else:
        defines, uses, calls = _names_in(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            risk = "high" if any(isinstance(child, ast.Call) for child in ast.walk(statement)) else "medium"
        elif any(isinstance(child, ast.Call) for child in ast.walk(statement)):
            risk = "high"

    return StatementInfo(
        key=key,
        scope=scope,
        kind=kind,
        defines=frozenset(defines),
        uses=frozenset(uses),
        call_names=frozenset(calls),
        side_effect_risk=risk,
    )


def _walk_statement_lists(body: Iterable[ast.stmt], scope: str, output: list[StatementInfo]) -> None:
    for statement in body:
        output.append(_statement_info(statement, scope))
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_statement_lists(statement.body, f"{scope}/function:{statement.name}", output)
            continue
        if isinstance(statement, ast.ClassDef):
            _walk_statement_lists(statement.body, f"{scope}/class:{statement.name}", output)
            continue
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, ast.stmt):
                _walk_statement_lists([child], scope, output)
            elif isinstance(child, ast.AST):
                nested = [node for node in ast.walk(child) if isinstance(node, ast.stmt)]
                # Only descend through statement lists; nested function/class
                # scopes are handled when their defining statement is visited.
                for nested_statement in nested:
                    if isinstance(nested_statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        _walk_statement_lists([nested_statement], scope, output)
                    else:
                        output.append(_statement_info(nested_statement, scope))


def analyze_source(source: str) -> DependencyAnalysis:
    tree = ast.parse(source)
    statements: list[StatementInfo] = []
    _walk_statement_lists(tree.body, "module", statements)
    analysis = DependencyAnalysis(statements=statements)

    by_scope: dict[str, list[StatementInfo]] = {}
    definitions: dict[tuple[str, str], NodeKey] = {}
    for info in statements:
        by_scope.setdefault(info.scope, []).append(info)
        for name in info.defines:
            definitions.setdefault((info.scope, name), info.key)
        if info.kind in {"FunctionDef", "AsyncFunctionDef"}:
            analysis.direct_calls[info.key.__repr__()] = set(info.call_names)

    for info in statements:
        deps: set[NodeKey] = set()
        for name in info.uses | info.call_names:
            defining = definitions.get((info.scope, name))
            if defining is not None and defining != info.key:
                deps.add(defining)
        analysis.dependencies[info.key] = deps

    analysis.dynamic_features = any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "globals", "locals", "getattr", "setattr"}
        )
        for node in ast.walk(tree)
    ) or any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree))
    analysis.component_count = _component_count(analysis.dependencies)
    return analysis


def _component_count(graph: dict[NodeKey, set[NodeKey]]) -> int:
    if not graph:
        return 0
    index = 0
    indices: dict[NodeKey, int] = {}
    low: dict[NodeKey, int] = {}
    stack: list[NodeKey] = []
    on_stack: set[NodeKey] = set()
    count = 0

    def visit(node: NodeKey) -> None:
        nonlocal index, count
        indices[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in graph.get(node, ()):
            if child not in indices:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in on_stack:
                low[node] = min(low[node], indices[child])
        if low[node] == indices[node]:
            count += 1
            while True:
                item = stack.pop()
                on_stack.remove(item)
                if item == node:
                    break

    for node in graph:
        if node not in indices:
            visit(node)
    return count


def dependency_candidates(analysis: DependencyAnalysis) -> list[DependencyCandidate]:
    candidates: list[DependencyCandidate] = []
    unused_by_scope: dict[str, list[StatementInfo]] = {}
    for info in analysis.statements:
        if info.scope == "module":
            same_scope = [other for other in analysis.statements if other.key != info.key]
        else:
            same_scope = [other for other in analysis.statements if other.scope == info.scope and other.key != info.key]
        used_elsewhere = set().union(*(other.uses | other.call_names for other in same_scope)) if same_scope else set()
        unused = info.defines and not (set(info.defines) & used_elsewhere)
        if not unused:
            continue
        if info.kind in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}:
            kind = "unused_definition"
        elif info.kind in {"Import", "ImportFrom"}:
            kind = "unused_import"
        elif info.kind in {"Assign", "AnnAssign", "AugAssign"}:
            kind = "unused_assignment"
        else:
            continue
        candidate = DependencyCandidate(
            kind=kind,
            label=f"{kind}:{info.key}",
            node_keys=(info.key,),
        )
        candidates.append(candidate)
        if kind == "unused_definition":
            unused_by_scope.setdefault(info.scope, []).append(info)

    for scope, infos in sorted(unused_by_scope.items()):
        if len(infos) >= 2:
            keys = tuple(info.key for info in infos)
            candidates.insert(
                0,
                DependencyCandidate(
                    kind="dependency_component",
                    label=f"dependency_component:{scope}:{len(keys)}",
                    node_keys=keys,
                ),
            )

    candidates.sort(key=lambda item: (
        0 if item.kind == "unused_import" else 1 if item.kind == "dependency_component" else 2 if item.kind == "unused_definition" else 3,
        item.label,
    ))
    return candidates


def import_alias_candidates(source: str) -> list[DependencyCandidate]:
    tree = ast.parse(source)
    analysis = analyze_source(source)
    output: list[DependencyCandidate] = []
    for info in analysis.statements:
        node = next((candidate for candidate in ast.walk(tree) if isinstance(candidate, ast.stmt) and node_key(candidate) == info.key), None)
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or len(node.names) < 2:
            continue
        if info.scope == "module":
            same_scope = [other for other in analysis.statements if other.key != info.key]
        else:
            same_scope = [other for other in analysis.statements if other.scope == info.scope and other.key != info.key]
        used = set().union(*(other.uses | other.call_names for other in same_scope)) if same_scope else set()
        for alias in node.names:
            binding = alias.asname or alias.name.split(".", 1)[0]
            if binding not in used:
                output.append(DependencyCandidate("import_alias", f"import_alias:{info.key}:{binding}", (info.key,), alias=binding))
    return sorted(output, key=lambda item: item.label)


def _required_block(owner: ast.AST, field: str) -> bool:
    required = (ast.AsyncFor, ast.AsyncFunctionDef, ast.AsyncWith, ast.ExceptHandler, ast.For, ast.FunctionDef, ast.If, ast.Try, ast.While, ast.With)
    return field == "body" and isinstance(owner, required)


def _remove_nodes(source: str, keys: set[NodeKey]) -> str:
    tree = ast.parse(source)
    for owner in ast.walk(tree):
        for field, value in list(ast.iter_fields(owner)):
            if not isinstance(value, list) or not value or not all(isinstance(item, ast.stmt) for item in value):
                continue
            retained = [item for item in value if node_key(item) not in keys]
            if not retained and _required_block(owner, field):
                retained = [ast.Pass()]
            setattr(owner, field, retained)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def _remove_alias(source: str, key: NodeKey, alias_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or node_key(node) != key:
            continue
        node.names = [alias for alias in node.names if (alias.asname or alias.name.split(".", 1)[0]) != alias_name]
        if not node.names:
            return _remove_nodes(source, {key})
        break
    return render_module(tree.body)


def expression_candidates(source: str) -> list[DependencyCandidate]:
    tree = ast.parse(source)
    output: list[DependencyCandidate] = []
    for node in ast.walk(tree):
        key = node_key(node)
        if isinstance(node, ast.BinOp):
            output.extend([
                DependencyCandidate("expression", f"expression:{key}:left", (key,), expression_action="left"),
                DependencyCandidate("expression", f"expression:{key}:right", (key,), expression_action="right"),
            ])
        elif isinstance(node, ast.BoolOp) and len(node.values) > 1:
            for index in range(len(node.values)):
                output.append(DependencyCandidate("expression", f"expression:{key}:value{index}", (key,), expression_action=f"value:{index}"))
        elif isinstance(node, ast.IfExp):
            output.extend([
                DependencyCandidate("expression", f"expression:{key}:body", (key,), expression_action="body"),
                DependencyCandidate("expression", f"expression:{key}:orelse", (key,), expression_action="orelse"),
            ])
    return sorted(output, key=lambda item: item.label)


def _replace_expression(source: str, key: NodeKey, action: str) -> str:
    tree = ast.parse(source)

    class Replacer(ast.NodeTransformer):
        def visit(self, node: ast.AST):  # type: ignore[override]
            if node_key(node) == key:
                if isinstance(node, ast.BinOp) and action in {"left", "right"}:
                    return ast.copy_location(getattr(node, action), node)
                if isinstance(node, ast.BoolOp) and action.startswith("value:"):
                    return ast.copy_location(node.values[int(action.split(":", 1)[1])], node)
                if isinstance(node, ast.IfExp) and action in {"body", "orelse"}:
                    return ast.copy_location(getattr(node, action), node)
            return super().visit(node)

    tree = Replacer().visit(tree)
    ast.fix_missing_locations(tree)
    return render_module(tree.body)


def apply_candidate(source: str, candidate: DependencyCandidate) -> str:
    if candidate.kind == "import_alias":
        assert candidate.alias is not None
        return _remove_alias(source, candidate.node_keys[0], candidate.alias)
    if candidate.kind == "expression":
        assert candidate.expression_action is not None
        return _replace_expression(source, candidate.node_keys[0], candidate.expression_action)
    return _remove_nodes(source, set(candidate.node_keys))


def reduce_dependency(
    source: str,
    test,
    history: list[dict[str, object]],
    record_candidate,
    *,
    record_analysis=None,
    max_rounds: int = 4,
) -> tuple[str, list[dict[str, object]]]:
    current = source
    for _ in range(max_rounds):
        analysis = analyze_source(current)
        if callable(record_analysis):
            record_analysis(analysis)
        candidates = dependency_candidates(analysis)
        candidates.extend(import_alias_candidates(current))
        candidates.extend(expression_candidates(current))
        if not candidates:
            break
        accepted = False
        for candidate in candidates:
            candidate_source = apply_candidate(current, candidate)
            if candidate_source == current:
                continue
            try:
                ast.parse(candidate_source)
            except SyntaxError:
                record_candidate(candidate.kind, False)
                continue
            outcome = invoke_test(
                test,
                candidate_source,
                metadata={
                    "transform": "DependencyCandidate",
                    "dependency_category": candidate.kind,
                    "label": candidate.label,
                    "dynamic_features": analysis.dynamic_features,
                },
            )
            record_candidate(candidate.kind, outcome)
            if outcome:
                history.append(
                    {
                        "transform": "DependencyCandidate",
                        "category": candidate.kind,
                        "label": candidate.label,
                    }
                )
                current = candidate_source
                accepted = True
                break
        if not accepted:
            break
    return current, history
