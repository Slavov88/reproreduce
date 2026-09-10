from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable

from .ddmin import ddmin
from .scheduler import invoke_test


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


def _statement_id(node: ast.stmt) -> int:
    """Return a stable identity for a statement within one parsed tree."""
    value = getattr(node, "_reproreduce_statement_id", None)
    if value is None:
        # Synthetic repair nodes are not part of the structural state; this
        # fallback keeps the helper total for callers outside the reducer.
        return id(node)
    return value


def _assign_statement_ids(tree: ast.AST) -> None:
    next_id = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            setattr(node, "_reproreduce_statement_id", next_id)
            next_id += 1


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
    state_outcomes: dict[str, bool],
) -> None:
    current = list(getattr(owner, field))
    scope = _scope_name(owner, field)
    attempt_context: dict[str, object] = {}
    trace_enabled = bool(getattr(getattr(test, "session", None), "trace", False))

    def metadata_for(
        parent: list[ast.stmt], candidate: list[ast.stmt], granularity: int
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "transform": "RemoveStatements",
            "scope": scope,
            "collection_size": len(parent),
            "candidate_size": len(candidate),
            "granularity": granularity,
            "removed_count": len(parent) - len(candidate),
        }
        if trace_enabled:
            metadata["parent_source_sha256"] = hashlib.sha256(
                render_module(tree.body).encode("utf-8")
            ).hexdigest()
        return metadata

    def state_key(candidate: list[ast.stmt]) -> str:
        original = getattr(owner, field)
        repaired = _repair_block(owner, field, list(candidate))
        setattr(owner, field, repaired)
        try:
            # AST dumps are canonical for the subsequent ast.unparse call but
            # avoid source generation and subprocess evaluation. Include the
            # complete tree so an identical local subtree under a changed
            # parent is not treated as the same search state.
            return ast.dump(tree, annotate_fields=True, include_attributes=False)
        finally:
            setattr(owner, field, original)

    def record_state_skip(
        parent: list[ast.stmt], candidate: list[ast.stmt], granularity: int, accepted: bool
    ) -> None:
        metadata = metadata_for(parent, candidate, granularity)
        metadata["candidate_state_sha256"] = hashlib.sha256(
            repr(state_key(candidate)).encode("utf-8")
        ).hexdigest()
        recorder = getattr(test, "_record_skipped_state", None)
        if callable(recorder):
            recorder(metadata, accepted)

    def reuse_attempt(
        parent: list[ast.stmt], candidate: list[ast.stmt], granularity: int
    ) -> bool | None:
        nonlocal current
        key = state_key(candidate)
        accepted = state_outcomes.get(key)
        external = False
        if accepted is None:
            reuse = getattr(test, "reuse_structural_state", None)
            if callable(reuse):
                accepted = reuse(
                    key,
                    {
                        **metadata_for(parent, candidate, granularity),
                        "candidate_state_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    },
                )
                external = accepted is not None
        if accepted is None:
            return None
        repaired = _repair_block(owner, field, list(candidate))
        setattr(owner, field, repaired if accepted else _repair_block(owner, field, list(parent)))
        if accepted:
            current = list(candidate)
        if not external:
            record_state_skip(parent, candidate, granularity, accepted)
        return accepted

    def on_attempt(
        parent: list[ast.stmt], candidate: list[ast.stmt], granularity: int
    ) -> None:
        attempt_context.clear()
        attempt_context.update(metadata_for(parent, candidate, granularity))

    def candidate_test(candidate: list[ast.stmt]) -> bool:
        nonlocal current
        key = state_key(candidate)
        if key in state_outcomes:
            # This path is normally handled by ``reuse_attempt``. Keep the
            # guard for direct callers and record it as a structural revisit.
            accepted = state_outcomes[key]
            record_state_skip(current, candidate, attempt_context.get("granularity", 2), accepted)
            setattr(owner, field, _repair_block(owner, field, list(candidate if accepted else current)))
            return accepted
        if tuple(_statement_id(item) for item in candidate) == tuple(_statement_id(item) for item in current) and len(_repair_block(owner, field, list(candidate))) == len(candidate):
            state_outcomes[key] = True
            remember = getattr(test, "remember_structural_state", None)
            if callable(remember):
                remember(key, True)
            recorder = getattr(test, "_record_no_op", None)
            if callable(recorder):
                recorder({**attempt_context, "candidate_state_sha256": hashlib.sha256(repr(key).encode("utf-8")).hexdigest()})
            return True
        repaired = _repair_block(owner, field, list(candidate))
        setattr(owner, field, repaired)
        try:
            candidate_source = render_module(tree.body)
            compile(candidate_source, str(getattr(tree, "filename", "<reproreduce>")), "exec")
            metadata = dict(attempt_context)
            metadata["candidate_source_sha256"] = hashlib.sha256(
                candidate_source.encode("utf-8")
            ).hexdigest()
            accepted = invoke_test(test, candidate_source, metadata=metadata)
        except (SyntaxError, ValueError):
            recorder = getattr(test, "_record_syntax_skip", None)
            if callable(recorder):
                recorder(dict(attempt_context))
            accepted = False
        state_outcomes[key] = accepted
        remember = getattr(test, "remember_structural_state", None)
        if callable(remember):
            remember(key, accepted)
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

    reduced = ddmin(
        current,
        candidate_test,
        on_attempt=on_attempt,
        reuse_attempt=reuse_attempt,
    )
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
    state_outcomes: dict[tuple[str, tuple[int, ...], bool], bool],
) -> None:
    # Snapshot fields so accepted removals do not invalidate traversal.
    for field, _ in list(ast.iter_fields(node)):
        value = getattr(node, field, None)
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            _reduce_statement_list(tree, node, field, test, history, state_outcomes)
            for child in list(getattr(node, field)):
                _reduce_node(tree, child, test, history, state_outcomes)
        elif isinstance(value, ast.AST):
            _reduce_node(tree, value, test, history, state_outcomes)
        elif isinstance(value, list):
            for child in list(value):
                if isinstance(child, ast.AST):
                    _reduce_node(tree, child, test, history, state_outcomes)


def reduce_statement_lists(source: str, test) -> tuple[str, list[dict[str, object]]]:
    """Remove statements from top-level and supported nested statement lists."""
    tree = ast.parse(source)
    history: list[dict[str, object]] = []
    _assign_statement_ids(tree)
    state_outcomes: dict[str, bool] = {}
    _reduce_node(tree, tree, test, history, state_outcomes)
    return render_module(tree.body), history


def reduce_top_level_statements(source: str, test) -> tuple[str, list[dict[str, object]]]:
    """Backward-compatible name for the recursive statement reducer."""
    return reduce_statement_lists(source, test)
