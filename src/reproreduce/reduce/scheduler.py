from __future__ import annotations

from collections.abc import Callable
from typing import Any


def invoke_test(
    test: Callable[[str], bool],
    source: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Call a candidate test while optionally attaching search metadata."""
    set_context = getattr(test, "set_context", None)
    if callable(set_context):
        set_context(metadata or {})
    return test(source)
