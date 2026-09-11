from pathlib import Path

from .core.session import ReductionSession
from .core.result import ReductionResult
from .oracle.base import FailureOracle
from .oracle.exception import ExceptionOracle


def reduce(
    program: str | Path,
    *,
    oracle: FailureOracle | None = None,
    timeout: float = 30.0,
    cache: str | Path | None = None,
    trace: bool = False,
    deduplicate: bool = True,
    structural_deduplicate: bool = True,
    jobs: int = 1,
    strategy: str = "standard",
) -> ReductionResult:
    """Reduce a Python program while preserving its baseline failure.

    ``strategy="dependency"`` enables V1 conservative static candidate proposals;
    ``strategy="dependency_v2"`` additionally explores coordinated local-function,
    control-flow, call-result, and expression candidates; ``dependency_v3`` adds
    bounded plumbing/configuration reduction with cost-aware expression search.
    every accepted proposal is still checked by the configured oracle.
    """
    selected_oracle = oracle or ExceptionOracle()
    session = ReductionSession(
        program=Path(program),
        oracle=selected_oracle,
        timeout=timeout,
        cache_path=Path(cache) if cache is not None else None,
        trace=trace,
        deduplicate=deduplicate,
        structural_deduplicate=structural_deduplicate,
        jobs=jobs,
        strategy=strategy,
    )
    return session.reduce()
