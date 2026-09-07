from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..execute.subprocess import execute_candidate
from ..oracle.base import FailureOracle, OracleResult
from .cache import CandidateCache
from .result import ReductionResult
from .run import RunResult
from ..reduce.ast import reduce_top_level_statements


class ReductionSession:
    def __init__(
        self,
        *,
        program: Path,
        oracle: FailureOracle,
        timeout: float,
        cache_path: Path | None,
    ):
        self.program = program.resolve()
        self.oracle = oracle
        self.timeout = timeout
        self.source = self.program.read_text(encoding="utf-8")
        self.cache_path = cache_path
        self._baseline: OracleResult | None = None
        self._evaluations = 0
        self._history: list[dict[str, object]] = []

    def _evaluate(self, source: str) -> tuple[RunResult, OracleResult]:
        cached = self.cache.get(source) if self.cache is not None else None
        if cached is not None:
            return cached
        run = execute_candidate(source, cwd=self.program.parent, timeout=self.timeout)
        result = self.oracle.evaluate(run)
        if self.cache is not None:
            self.cache.put(source, run, result)
        self._evaluations += 1
        return run, result

    def _preserves_failure(self, source: str) -> bool:
        run, result = self._evaluate(source)
        assert self._baseline is not None
        accepted = result.interesting and self.oracle.same_failure(self._baseline, result)
        if accepted:
            self._history.append(
                {
                    "evaluation": self._evaluations,
                    "loc": len(source.splitlines()),
                    "duration_seconds": run.duration_seconds,
                    "fingerprint": result.fingerprint,
                }
            )
        return accepted

    def reduce(self) -> ReductionResult:
        temporary_cache = self.cache_path is None
        if temporary_cache:
            descriptor, temporary_path = tempfile.mkstemp(suffix=".sqlite3")
            os.close(descriptor)
            cache_path = Path(temporary_path)
        else:
            cache_path = self.cache_path
        self.cache = CandidateCache(cache_path)
        try:
            original_run, baseline = self._evaluate(self.source)
            if not baseline.interesting:
                raise ValueError(
                    "The original program did not match the failure oracle. "
                    f"stderr:\n{original_run.stderr}"
                )
            self._baseline = baseline
            reduced_source, ast_history = reduce_top_level_statements(
                self.source, self._preserves_failure
            )
            self._history.extend(ast_history)
            reduced_run, reduced_result = self._evaluate(reduced_source)
            if not (reduced_result.interesting and self.oracle.same_failure(baseline, reduced_result)):
                raise RuntimeError("Reducer produced a candidate that does not preserve the baseline failure")
            return ReductionResult(
                original_source=self.source,
                reduced_source=reduced_source,
                original_run=original_run,
                reduced_run=reduced_run,
                history=self._history,
            )
        finally:
            self.cache.close()
            if temporary_cache:
                try:
                    cache_path.unlink()
                except OSError:
                    pass
