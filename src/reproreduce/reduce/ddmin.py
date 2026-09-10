from __future__ import annotations

from collections.abc import Callable, Sequence


AttemptReuse = Callable[[list, list, int], bool | None]
BatchTest = Callable[[list[list]], list[bool]]


def _chunks(items: Sequence, count: int) -> list[list]:
    size = max(1, (len(items) + count - 1) // count)
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def ddmin(
    items: Sequence,
    test: Callable[[list], bool],
    *,
    on_attempt: Callable[[list, list, int], None] | None = None,
    reuse_attempt: AttemptReuse | None = None,
    test_batch: BatchTest | None = None,
) -> list:
    """Reduce a sequence using classic delta debugging.

    ``on_attempt`` is diagnostic-only and runs immediately before each
    complement candidate is tested.
    """
    current = list(items)
    granularity = 2

    while len(current) >= 2:
        chunks = _chunks(current, granularity)
        reduced = False
        if test_batch is None:
            for chunk in chunks:
                candidate = list(current)
                for item in chunk:
                    candidate.remove(item)
                if reuse_attempt is not None:
                    reused = reuse_attempt(list(current), list(candidate), granularity)
                    if reused is not None:
                        if reused:
                            current = candidate
                            granularity = max(granularity - 1, 2)
                            reduced = True
                            break
                        continue
                if on_attempt is not None:
                    on_attempt(list(current), list(candidate), granularity)
                if test(candidate):
                    current = candidate
                    granularity = max(granularity - 1, 2)
                    reduced = True
                    break
        else:
            batch: list[list] = []
            for chunk in chunks:
                candidate = list(current)
                for item in chunk:
                    candidate.remove(item)
                if on_attempt is not None:
                    on_attempt(list(current), list(candidate), granularity)
                batch.append(candidate)
            if batch:
                outcomes = test_batch(batch)
                if len(outcomes) != len(batch):
                    raise ValueError("test_batch returned the wrong number of outcomes")
                for candidate, accepted in zip(batch, outcomes):
                    if accepted:
                        current = candidate
                        granularity = max(granularity - 1, 2)
                        reduced = True
                        break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return current
