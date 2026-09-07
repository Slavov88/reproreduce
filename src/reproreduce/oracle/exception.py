import re
from dataclasses import dataclass
from typing import Pattern

from ..core.run import RunResult
from .base import OracleResult

_EXCEPTION_LINE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit|BaseException)?)"
    r"(?:\s*:\s*(?P<message>.*))?\s*$"
)


def exception_details(run: RunResult) -> tuple[str, str] | None:
    """Extract the final Python exception type and message from stderr."""
    for line in reversed(run.stderr.splitlines()):
        match = _EXCEPTION_LINE.match(line)
        if match and (match.group("type").endswith(("Error", "Exception", "Interrupt", "Exit"))):
            return match.group("type"), match.group("message") or ""
    return None


@dataclass(frozen=True)
class ExceptionOracle:
    """Preserve a configured Python exception and optional message pattern."""

    exception_type: str | None = None
    message_regex: str | None = None

    def evaluate(self, run: RunResult) -> OracleResult:
        details = exception_details(run)
        if details is None:
            return OracleResult(False, None, metadata={"reason": "no_python_exception"})

        exception_type, message = details
        type_matches = self.exception_type is None or exception_type == self.exception_type
        message_matches = self.message_regex is None or re.search(self.message_regex, message) is not None
        interesting = type_matches and message_matches
        fingerprint = f"{exception_type}: {message}" if interesting else f"{exception_type}: {message}"
        return OracleResult(
            interesting=interesting,
            fingerprint=fingerprint,
            metadata={"exception_type": exception_type, "message": message},
        )

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        if not baseline.interesting or not candidate.interesting:
            return False
        baseline_type = baseline.metadata.get("exception_type")
        candidate_type = candidate.metadata.get("exception_type")
        return baseline_type == candidate_type
