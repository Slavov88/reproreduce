from __future__ import annotations

import ast
from collections.abc import Callable

from .ddmin import ddmin


_REQUIRED_BLOCK_OWNERS = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ExceptHandler,
    ast.For,
    ast.FunctionDef,
    ast.If,
    ast.Try,
    ast.While,
    ast.With,
)


def render_module(statements: list[ast.stmt]) -> str:
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _scope_name(owner: ast.AST, field: str) -> str:
    if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"{type(owner).__name__}:{owner.name}.{field}"
    if isinstance(owner, ast.ClassDef):
        return f"ClassDef:{owner.name}.{field}"
    return f"{type(owner).__name__}.{field}"


def _requires_statement(owner: ast.AST, field: str) -> bool:
    if field == "body" and isinstance(owner, _REQUIRED_BLOCK_OWNERS):
        return True
    # A Try node with no except handlers requires a non-empty finally block.
    if field == "finalbody" and isinstance(owner, ast.Try) and not owner.handlers:
        return True
    return False


def _repair_block(owner: ast.AST, field: str, statements: list[ast.stmt]) -> list[ast.stmt]:
    if not statements and _requires_statement(owner, field):
        return [ast.Pass()]
    return statements


def _statement_fields(node: ast.AST) -> list[tuple[str, list[ast.stmt]]]:
    fields: list[tuple[str, list[ast.stmt]]] = []
    for field, value in ast.iter_fields(node):
        if isinstance(value, list) and all(isinstance(item, ast.stmt) for item in value):
            fields.append((field, value))
    return fields


def _reduce_statement_list(
    tree: ast.Module,
    owner: ast.AST,
    field: str,
    test: Callable[[str], bool],
    history: list[dict[str, object]],
) -> None:
    current = list(getattr(owner, field))
    scope = _scope_name(owner, field)

    def candidate_test(candidate: list[ast.stmt]) -> bool:
        nonlocal current
        repaired = _repair_block(owner, field, list(candidate))
        setattr(owner, field, repaired)
        try:
            candidate_source = render_module(tree.body)
            ast.parse(candidate_source)
            accepted = test(candidate_source)
        except (SyntaxError, ValueError):
            accepted = False
        if accepted:
            removed_count = len(current) - len(candidate)
            current = list(candidate)
            history.append(
                {
                    "transform": "RemoveStatements",
                    "scope": scope,
                    "removed_count": removed_count,
                    "remaining_statements": len(candidate),
                }
            )
            return True
        setattr(owner, field, _repair_block(owner, field, current))
        return False

    reduced = ddmin(current, candidate_test)
    # Classic ddmin stops when one item remains. Try the empty set explicitly so
    # required blocks can be repaired with ``pass`` when their contents are
    # irrelevant to the preserved failure.
    if reduced and candidate_test([]):
        reduced = []
    setattr(owner, field, _repair_block(owner, field, reduced))


def _reduce_node(
    tree: ast.Module,
    node: ast.AST,
    test: Callable[[str], bool],
    history: list[dict[str, object]],
) -> None:
    # Snapshot fields so accepted removals do not invalidate traversal.
    for field, _ in list(ast.iter_fields(node)):
        value = getattr(node, field, None)
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            _reduce_statement_list(tree, node, field, test, history)
            for child in list(getattr(node, field)):
                _reduce_node(tree, child, test, history)
        elif isinstance(value, ast.AST):
            _reduce_node(tree, value, test, history)
        elif isinstance(value, list):
            for child in list(value):
                if isinstance(child, ast.AST):
                    _reduce_node(tree, child, test, history)


def reduce_statement_lists(source: str, test) -> tuple[str, list[dict[str, object]]]:
    """Remove statements from top-level and supported nested statement lists."""
    tree = ast.parse(source)
    history: list[dict[str, object]] = []
    _reduce_node(tree, tree, test, history)
    return render_module(tree.body), history


def reduce_top_level_statements(source: str, test) -> tuple[str, list[dict[str, object]]]:
    """Backward-compatible name for the recursive statement reducer."""
    return reduce_statement_lists(source, test)
