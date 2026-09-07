from __future__ import annotations

import ast

from .ddmin import ddmin


def render_module(statements: list[ast.stmt]) -> str:
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def reduce_top_level_statements(source: str, test) -> tuple[str, list[dict[str, object]]]:
    """Remove top-level statements; every accepted candidate is oracle-tested."""
    tree = ast.parse(source)
    original = list(tree.body)
    accepted: list[dict[str, object]] = []

    def candidate_test(statements: list[ast.stmt]) -> bool:
        candidate_source = render_module(statements)
        if not test(candidate_source):
            return False
        accepted.append({
            "transform": "remove_top_level_statements",
            "remaining_statements": len(statements),
        })
        return True

    reduced = ddmin(original, candidate_test)
    return render_module(reduced), accepted
