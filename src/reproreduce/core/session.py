from __future__ import annotations

import os
import statistics
import tempfile
import time
from pathlib import Path

from ..execute.subprocess import execute_candidate
from ..oracle.base import FailureOracle, OracleResult
from .cache import CandidateCache
from .result import ReductionResult
from .run import RunResult
from ..reduce.ast import reduce_top_level_statements
from ..pytorch.modules import reduce_module_lists, reduce_sequential_modules
from ..pytorch.tensors import reduce_tensor_constructors


MAX_REDUCTION_PASSES = 2


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
        self._cache: CandidateCache | None = None
        self._candidate_runs = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._candidate_durations: list[float] = []
        self._rejected_transformations = 0
        self._started_at = 0.0

    def _evaluate(self, source: str) -> tuple[RunResult, OracleResult]:
        cached = self._cache.get(source) if self._cache is not None else None
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        source_evaluator = getattr(self.oracle, "evaluate_source", None)
        if callable(source_evaluator):
            run, result = source_evaluator(
                source,
                cwd=self.program.parent,
                timeout=self.timeout,
            )
        else:
            run = execute_candidate(source, cwd=self.program.parent, timeout=self.timeout)
            result = self.oracle.evaluate(run)
        if self._cache is not None:
            self._cache.put(source, run, result)
        self._candidate_runs += 1
        self._candidate_durations.append(run.duration_seconds)
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
        else:
            self._rejected_transformations += 1
        return accepted

    def reduce(self) -> ReductionResult:
        self._started_at = time.perf_counter()
        temporary_cache = self.cache_path is None
        if temporary_cache:
            descriptor, temporary_path = tempfile.mkstemp(suffix=".sqlite3")
            os.close(descriptor)
            cache_path = Path(temporary_path)
        else:
            cache_path = self.cache_path
        self._cache = CandidateCache(cache_path)
        try:
            original_run, baseline = self._evaluate(self.source)
            if not baseline.interesting:
                raise ValueError(
                    "The original program did not match the failure oracle. "
                    f"stderr:\n{original_run.stderr}"
                )
            self._baseline = baseline
            reduced_source = self.source
            for _ in range(MAX_REDUCTION_PASSES):
                before = reduced_source

                reduced_source, ast_history = reduce_top_level_statements(
                    reduced_source, self._preserves_failure
                )
                self._history.extend(ast_history)
                reduced_source, module_history = reduce_sequential_modules(
                    reduced_source, self._preserves_failure
                )
                self._history.extend(module_history)
                reduced_source, module_list_history = reduce_module_lists(
                    reduced_source, self._preserves_failure
                )
                self._history.extend(module_list_history)
                reduced_source, tensor_history = reduce_tensor_constructors(
                    reduced_source, self._preserves_failure
                )
                self._history.extend(tensor_history)
                reduced_source, cleanup_history = reduce_top_level_statements(
                    reduced_source, self._preserves_failure
                )
                self._history.extend(cleanup_history)

                if reduced_source == before:
                    break

            reduced_run, reduced_result = self._evaluate(reduced_source)
            if not (reduced_result.interesting and self.oracle.same_failure(baseline, reduced_result)):
                raise RuntimeError("Reducer produced a candidate that does not preserve the baseline failure")
            return ReductionResult(
                original_source=self.source,
                reduced_source=reduced_source,
                original_run=original_run,
                reduced_run=reduced_run,
                history=self._history,
                metrics=self._metrics(),
            )
        finally:
            self._cache.close()
            if temporary_cache:
                try:
                    cache_path.unlink()
                except OSError:
                    pass

    def _metrics(self) -> dict[str, int | float]:
        accepted = sum(1 for entry in self._history if "transform" in entry)
        durations = self._candidate_durations
        return {
            "candidate_runs": self._candidate_runs,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_candidate_execution_time": sum(durations),
            "median_candidate_execution_time": statistics.median(durations) if durations else 0.0,
            "total_reduction_wall_time": time.perf_counter() - self._started_at,
            "accepted_transformations": accepted,
            "rejected_transformations": self._rejected_transformations,
        }
