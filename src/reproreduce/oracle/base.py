from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.run import RunResult


@dataclass(frozen=True)
class OracleResult:
    interesting: bool
    fingerprint: str | None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FailureOracle(Protocol):
    def evaluate(self, run: RunResult) -> OracleResult:
        """Classify one candidate run."""

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        """Return whether two interesting results represent the same failure."""
        ...
