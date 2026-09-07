from __future__ import annotations

from collections.abc import Callable, Sequence


def _chunks(items: Sequence, count: int) -> list[list]:
    size = max(1, (len(items) + count - 1) // count)
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def ddmin(items: Sequence, test: Callable[[list], bool]) -> list:
    """Reduce a sequence using classic delta debugging."""
    current = list(items)
    granularity = 2

    while len(current) >= 2:
        chunks = _chunks(current, granularity)
        reduced = False
        for chunk in chunks:
            candidate = list(current)
            for item in chunk:
                candidate.remove(item)
            if test(candidate):
                current = candidate
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return current
