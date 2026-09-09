from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import TensorConfig
from .execute import ExecutionResult, OutcomeClass
from .program import Program


@dataclass(frozen=True)
class ConfirmationPolicy:
    attempts: int = 5
    min_successes: int = 5


@dataclass(frozen=True)
class ConfirmationResult:
    attempts: int
    successes: int
    required_successes: int
    classifications: tuple[OutcomeClass, ...]

    @property
    def reproducible(self) -> bool:
        return self.successes >= self.required_successes


@dataclass(frozen=True)
class FindingFingerprint:
    backend: str
    classification: str
    operations: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    discrepancy_kind: str

    @classmethod
    def from_case(
        cls,
        result: ExecutionResult,
        program: Program,
        configs: Iterable[TensorConfig],
    ) -> "FindingFingerprint":
        metadata = result.oracle_result.metadata if result.oracle_result else {}
        return cls(
            backend=result.backend,
            classification=result.classification.value,
            operations=tuple(operation.name for operation in program.operations),
            shapes=tuple(config.shape for config in configs),
            dtypes=tuple(config.dtype for config in configs),
            layouts=tuple(config.layout for config in configs),
            discrepancy_kind=str(metadata.get("kind", "unknown")),
        )

    def serialize(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()


class FindingDeduplicator:
    def __init__(self):
        self._findings: dict[str, FindingFingerprint] = {}

    def add(self, finding: FindingFingerprint) -> bool:
        """Add a finding and return whether it was not already present."""
        if finding.identity in self._findings:
            return False
        self._findings[finding.identity] = finding
        return True

    @property
    def findings(self) -> tuple[FindingFingerprint, ...]:
        return tuple(self._findings.values())


def confirm(
    run: Callable[[], ExecutionResult],
    *,
    expected: OutcomeClass,
    policy: ConfirmationPolicy = ConfirmationPolicy(),
    expected_fingerprint: str | None = None,
) -> ConfirmationResult:
    classifications: list[OutcomeClass] = []
    successes = 0
    for _ in range(policy.attempts):
        result = run()
        classifications.append(result.classification)
        fingerprint_matches = (
            expected_fingerprint is None
            or (result.oracle_result is not None and result.oracle_result.fingerprint == expected_fingerprint)
        )
        if result.classification == expected and fingerprint_matches:
            successes += 1
    return ConfirmationResult(
        attempts=policy.attempts,
        successes=successes,
        required_successes=policy.min_successes,
        classifications=tuple(classifications),
    )
