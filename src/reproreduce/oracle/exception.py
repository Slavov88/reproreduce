from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from ..core.run import RunResult
from .base import OracleResult

_EXCEPTION_LINE = re.compile(
    r"^\s*(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit|BaseException)?)"
    r"(?:\s*:\s*(?P<message>.*))?\s*$"
)
_FRAME_LINE = re.compile(r'^\s*File "(?P<path>.+?)", line \d+, in (?P<function>.+?)\s*$')
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s:'\"]+")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_LINE_NUMBER = re.compile(r"\bline\s+\d+\b")


@dataclass(frozen=True)
class FailureFingerprint:
    """Stable, deliberately small identity for a Python exception failure."""

    exception_type: str | None
    message_signature: str | None
    relevant_frames: tuple[str, ...] = ()
    signal: int | None = None

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _normalize_message(message: str) -> str:
    normalized = _PATH.sub("<PATH>", message)
    normalized = _ADDRESS.sub("<ADDRESS>", normalized)
    normalized = _LINE_NUMBER.sub("line <NUMBER>", normalized)
    return normalized


def _relevant_frames(stderr: str) -> tuple[str, ...]:
    frames: list[str] = []
    for line in stderr.splitlines():
        match = _FRAME_LINE.match(line)
        if match:
            # Function names survive line movement and temporary candidate paths.
            frames.append(f"in {match.group('function')}")
    return tuple(frames[-3:])


def exception_details(run: RunResult) -> tuple[str, str] | None:
    """Extract the final Python exception type and message from stderr."""
    for line in reversed(run.stderr.splitlines()):
        match = _EXCEPTION_LINE.match(line)
        if match and match.group("type").endswith(
            ("Error", "Exception", "Interrupt", "Exit", "BaseException")
        ):
            return match.group("type"), match.group("message") or ""
    return None


def exception_fingerprint(run: RunResult) -> FailureFingerprint | None:
    if run.timed_out:
        return None
    details = exception_details(run)
    if details is None:
        return None
    exception_type, message = details
    signal = -run.returncode if run.returncode is not None and run.returncode < 0 else None
    return FailureFingerprint(
        exception_type=exception_type,
        message_signature=_normalize_message(message),
        relevant_frames=_relevant_frames(run.stderr),
        signal=signal,
    )


@dataclass(frozen=True)
class ExceptionOracle:
    """Preserve a configured Python exception and optional message pattern."""

    exception_type: str | None = None
    message_regex: str | None = None

    def evaluate(self, run: RunResult) -> OracleResult:
        if run.timed_out:
            return OracleResult(False, None, metadata={"reason": "timeout"})

        fingerprint = exception_fingerprint(run)
        if fingerprint is None:
            return OracleResult(False, None, metadata={"reason": "no_python_exception"})

        message = fingerprint.message_signature or ""
        type_matches = self.exception_type is None or fingerprint.exception_type == self.exception_type
        message_matches = self.message_regex is None or re.search(self.message_regex, message) is not None
        interesting = type_matches and message_matches
        metadata = {
            "exception_type": fingerprint.exception_type,
            "message": message,
            "message_signature": fingerprint.message_signature,
            "relevant_frames": fingerprint.relevant_frames,
            "signal": fingerprint.signal,
        }
        return OracleResult(
            interesting=interesting,
            fingerprint=fingerprint.canonical(),
            metadata=metadata,
        )

    def same_failure(self, baseline: OracleResult, candidate: OracleResult) -> bool:
        if not baseline.interesting or not candidate.interesting:
            return False
        return (
            baseline.metadata.get("exception_type") == candidate.metadata.get("exception_type")
            and baseline.metadata.get("message_signature") == candidate.metadata.get("message_signature")
            and baseline.metadata.get("signal") == candidate.metadata.get("signal")
        )
