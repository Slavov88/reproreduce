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
        self._candidate_run_results: list[RunResult] = []
        self._candidate_call_seconds = 0.0
        self._oracle_seconds = 0.0
        self._cache_lookup_seconds = 0.0
        self._cache_write_seconds = 0.0
        self._cache_hit_sources: dict[str, int] = {}
        self._rejected_transformations = 0
        self._started_at = 0.0

    def _evaluate(self, source: str) -> tuple[RunResult, OracleResult]:
        if self._cache is not None:
            lookup_started = time.perf_counter()
            cached = self._cache.get(source)
            self._cache_lookup_seconds += time.perf_counter() - lookup_started
        else:
            cached = None
        if cached is not None:
            self._cache_hits += 1
            key = CandidateCache.key(source)
            self._cache_hit_sources[key] = self._cache_hit_sources.get(key, 0) + 1
            return cached
        self._cache_misses += 1
        source_evaluator = getattr(self.oracle, "evaluate_source", None)
        candidate_started = time.perf_counter()
        if callable(source_evaluator):
            run, result = source_evaluator(
                source,
                cwd=self.program.parent,
                timeout=self.timeout,
            )
        else:
            run = execute_candidate(source, cwd=self.program.parent, timeout=self.timeout)
            oracle_started = time.perf_counter()
            result = self.oracle.evaluate(run)
            self._oracle_seconds += time.perf_counter() - oracle_started
        self._candidate_call_seconds += time.perf_counter() - candidate_started
        if self._cache is not None:
            write_started = time.perf_counter()
            self._cache.put(source, run, result)
            self._cache_write_seconds += time.perf_counter() - write_started
        self._candidate_runs += 1
        self._candidate_durations.append(run.duration_seconds)
        self._candidate_run_results.append(run)
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
        wall_time = time.perf_counter() - self._started_at
        candidate_execution = sum(durations)
        accounted = (
            self._candidate_call_seconds
            + self._oracle_seconds
            + self._cache_lookup_seconds
            + self._cache_write_seconds
        )
        return {
            "candidate_runs": self._candidate_runs,
            "candidate_requests": self._candidate_runs + self._cache_hits,
            "unique_candidate_sources": self._candidate_runs,
            "duplicate_candidate_sources": len(self._cache_hit_sources),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_candidate_execution_time": candidate_execution,
            "candidate_call_seconds": self._candidate_call_seconds,
            "median_candidate_execution_time": statistics.median(durations) if durations else 0.0,
            "subprocess_source_write_seconds": sum(
                run.source_write_seconds for run in self._candidate_run_results
            ),
            "subprocess_startup_seconds": sum(
                run.process_startup_seconds for run in self._candidate_run_results
            ),
            "subprocess_wait_seconds": sum(
                run.process_wait_seconds for run in self._candidate_run_results
            ),
            "oracle_seconds": self._oracle_seconds,
            "cache_lookup_seconds": self._cache_lookup_seconds,
            "cache_write_seconds": self._cache_write_seconds,
            "reducer_bookkeeping_seconds": max(0.0, wall_time - accounted),
            "total_reduction_wall_time": wall_time,
            "accepted_transformations": accepted,
            "rejected_transformations": self._rejected_transformations,
        }
