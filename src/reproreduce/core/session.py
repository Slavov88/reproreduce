from __future__ import annotations

import os
import statistics
import tempfile
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..execute.subprocess import execute_candidate
from ..oracle.base import FailureOracle, OracleResult
from .cache import CandidateCache
from .result import ReductionResult
from .run import RunResult
from ..reduce.ast import reduce_top_level_statements
from ..reduce.dependency import reduce_dependency
from ..pytorch.modules import reduce_module_lists, reduce_sequential_modules
from ..pytorch.tensors import reduce_tensor_constructors


MAX_REDUCTION_PASSES = 2


class _SessionCandidateTest:
    def __init__(
        self,
        session: "ReductionSession",
        *,
        deduplicate: bool,
        structural_deduplicate: bool = True,
        jobs: int = 1,
    ) -> None:
        self.session = session
        self.deduplicate = deduplicate
        self.structural_deduplicate = structural_deduplicate
        self.jobs = jobs
        self._context: dict[str, object] = {}
        self._outcomes: dict[str, bool] = {}
        self._structural_outcomes: dict[str, bool] = {}

    def set_context(self, metadata: dict[str, object]) -> None:
        self._context = dict(metadata)

    def reuse_structural_state(
        self, key: str, context: dict[str, object]
    ) -> bool | None:
        if not self.structural_deduplicate:
            return None
        accepted = self._structural_outcomes.get(key)
        if accepted is None:
            return None
        self.session._record_skipped_state(
            {
                **context,
                "candidate_state_sha256": CandidateCache.key(key),
            },
            accepted,
        )
        return accepted

    def remember_structural_state(self, key: str, accepted: bool) -> None:
        if self.structural_deduplicate:
            self._structural_outcomes[key] = accepted

    def evaluate_batch(
        self,
        sources: list[str],
        contexts: list[dict[str, object]],
    ) -> list[bool]:
        outcomes: list[bool | None] = [None] * len(sources)
        unique_sources: list[str] = []
        unique_contexts: list[dict[str, object]] = []
        unique_positions: list[list[int]] = []
        positions_by_key: dict[str, int] = {}
        for index, (source, context) in enumerate(zip(sources, contexts)):
            self.session._scheduler_requests += 1
            key = CandidateCache.key(source)
            self.session._scheduler_sources.add(key)
            if self.deduplicate and key in self._outcomes:
                accepted = self._outcomes[key]
                self.session._record_skipped_duplicate(source, context, accepted)
                outcomes[index] = accepted
                continue
            unique_index = positions_by_key.get(key)
            if unique_index is None:
                positions_by_key[key] = len(unique_sources)
                unique_sources.append(source)
                unique_contexts.append(context)
                unique_positions.append([index])
            else:
                unique_positions[unique_index].append(index)
        evaluated = self.session._evaluate_source_batch(
            unique_sources,
            unique_contexts,
            jobs=self.jobs,
        )
        for source, positions, accepted in zip(unique_sources, unique_positions, evaluated):
            key = CandidateCache.key(source)
            self._outcomes[key] = accepted
            for index in positions:
                outcomes[index] = accepted
        return [bool(outcome) for outcome in outcomes]

    def __call__(self, source: str) -> bool:
        context = self._context
        self._context = {}
        self.session._scheduler_requests += 1
        key = CandidateCache.key(source)
        self.session._scheduler_sources.add(key)
        if self.deduplicate and key in self._outcomes:
            accepted = self._outcomes[key]
            self.session._record_skipped_duplicate(source, context, accepted)
            return accepted
        accepted = self.session._preserves_failure(source, context=context)
        self._outcomes[key] = accepted
        return accepted


class ReductionSession:
    def __init__(
        self,
        *,
        program: Path,
        oracle: FailureOracle,
        timeout: float,
        cache_path: Path | None,
        trace: bool = False,
        deduplicate: bool = True,
        structural_deduplicate: bool = True,
        jobs: int = 1,
        strategy: str = "standard",
    ):
        if jobs < 1:
            raise ValueError("jobs must be at least 1")
        if strategy not in {"standard", "dependency"}:
            raise ValueError("strategy must be 'standard' or 'dependency'")
        self.program = program.resolve()
        self.oracle = oracle
        self.timeout = timeout
        self.source = self.program.read_text(encoding="utf-8")
        self.cache_path = cache_path
        self.trace = trace
        self.deduplicate = deduplicate
        self.structural_deduplicate = structural_deduplicate
        self.jobs = jobs
        self.strategy = strategy
        self._dependency_metrics: dict[str, int] = {
            "dependency_candidates": 0,
            "dependency_components": 0,
            "dependency_component_candidates": 0,
            "unused_definition_candidates": 0,
            "unused_assignment_candidates": 0,
            "unused_import_candidates": 0,
            "import_alias_candidates": 0,
            "parameter_candidates": 0,
            "expression_candidates": 0,
            "dependency_accepted_dependency_component": 0,
            "dependency_accepted_unused_definition": 0,
            "dependency_accepted_unused_assignment": 0,
            "dependency_accepted_unused_import": 0,
            "dependency_accepted_import_alias": 0,
            "dependency_accepted_expression": 0,
        }
        self._baseline: OracleResult | None = None
        self._evaluations = 0
        self._history: list[dict[str, object]] = []
        self._cache: CandidateCache | None = None
        self._memory_cache: dict[str, tuple[RunResult, OracleResult]] = {}
        self._memory_cache_hits = 0
        self._sqlite_cache_hits = 0
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
        self._search_trace: list[dict[str, object]] = []
        self._scheduler_requests = 0
        self._scheduler_sources: set[str] = set()
        self._skipped_duplicate_candidates = 0
        self._skipped_state_candidates = 0
        self._no_op_skips = 0
        self._syntax_skips = 0
        self._last_evaluation_kind = "unknown"
        self._rejected_transformations = 0
        self._started_at = 0.0
        self._candidates_submitted = 0
        self._candidates_completed = 0
        self._candidates_cancelled = 0
        self._speculative_executions = 0
        self._useful_executions = 0
        self._peak_concurrency = 0
        self._active_concurrency = 0
        self._concurrency_lock = threading.Lock()

    def _lookup_cached(self, source: str) -> tuple[RunResult, OracleResult] | None:
        cached = self._memory_cache.get(source)
        if cached is not None:
            self._memory_cache_hits += 1
            self._last_evaluation_kind = "memory_cache_hit"
        elif self._cache is not None:
            lookup_started = time.perf_counter()
            cached = self._cache.get(source)
            self._cache_lookup_seconds += time.perf_counter() - lookup_started
            if cached is not None:
                self._sqlite_cache_hits += 1
                self._last_evaluation_kind = "sqlite_cache_hit"
                self._memory_cache[source] = cached
        if cached is not None:
            self._cache_hits += 1
            key = CandidateCache.key(source)
            self._cache_hit_sources[key] = self._cache_hit_sources.get(key, 0) + 1
        return cached

    def _execute_uncached(
        self, source: str
    ) -> tuple[RunResult, OracleResult, float, float]:
        candidate_started = time.perf_counter()
        oracle_seconds = 0.0
        source_evaluator = getattr(self.oracle, "evaluate_source", None)
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
            oracle_seconds = time.perf_counter() - oracle_started
        return run, result, time.perf_counter() - candidate_started, oracle_seconds

    def _commit_evaluation(
        self,
        source: str,
        payload: tuple[RunResult, OracleResult, float, float],
    ) -> tuple[RunResult, OracleResult]:
        run, result, candidate_seconds, oracle_seconds = payload
        if self._cache is not None:
            write_started = time.perf_counter()
            self._cache.put(source, run, result)
            self._cache_write_seconds += time.perf_counter() - write_started
        self._candidate_call_seconds += candidate_seconds
        self._oracle_seconds += oracle_seconds
        self._candidate_runs += 1
        self._candidate_durations.append(run.duration_seconds)
        self._candidate_run_results.append(run)
        self._memory_cache[source] = (run, result)
        self._evaluations += 1
        return run, result

    def _evaluate(self, source: str) -> tuple[RunResult, OracleResult]:
        cached = self._lookup_cached(source)
        if cached is not None:
            return cached
        self._cache_misses += 1
        self._last_evaluation_kind = "oracle_execution"
        self._candidates_submitted += 1
        payload = self._execute_uncached(source)
        self._candidates_completed += 1
        self._useful_executions += 1
        return self._commit_evaluation(source, payload)

    def _record_skipped_state(
        self,
        context: dict[str, object],
        accepted: bool,
    ) -> None:
        self._skipped_state_candidates += 1
        if self.trace:
            self._search_trace.append(
                {
                    **context,
                    "request_index": len(self._search_trace) + 1,
                    "evaluation_kind": "scheduler_skip_state",
                    "candidate_previously_tested": True,
                    "accepted": accepted,
                    "reason": "skipped_structural_state",
                }
            )

    def _record_dependency_analysis(self, analysis) -> None:
        self._dependency_metrics["dependency_components"] = max(
            self._dependency_metrics["dependency_components"],
            int(getattr(analysis, "component_count", 0)),
        )

    def _record_dependency_candidate(self, category: str, accepted: bool) -> None:
        self._dependency_metrics["dependency_candidates"] += 1
        category_key = {
            "dependency_component": "dependency_component_candidates",
            "unused_definition": "unused_definition_candidates",
            "unused_assignment": "unused_assignment_candidates",
            "unused_import": "unused_import_candidates",
            "import_alias": "import_alias_candidates",
            "expression": "expression_candidates",
        }.get(category)
        if category_key is not None:
            self._dependency_metrics[category_key] += 1
        if accepted:
            accepted_key = f"dependency_accepted_{category}"
            if accepted_key in self._dependency_metrics:
                self._dependency_metrics[accepted_key] += 1

    def _record_syntax_skip(self, context: dict[str, object]) -> None:
        self._syntax_skips += 1
        if self.trace:
            self._search_trace.append(
                {
                    **context,
                    "request_index": len(self._search_trace) + 1,
                    "evaluation_kind": "compile_skip_syntax",
                    "candidate_previously_tested": False,
                    "accepted": False,
                    "reason": "local_compile_rejected",
                }
            )

    def _record_no_op(self, context: dict[str, object]) -> None:
        self._no_op_skips += 1
        if self.trace:
            self._search_trace.append(
                {
                    **context,
                    "request_index": len(self._search_trace) + 1,
                    "evaluation_kind": "no_op_skip",
                    "candidate_previously_tested": False,
                    "accepted": True,
                    "reason": "no_op_transformation",
                }
            )

    def _record_skipped_duplicate(
        self,
        source: str,
        context: dict[str, object],
        accepted: bool,
    ) -> None:
        self._skipped_duplicate_candidates += 1
        if self.trace:
            self._search_trace.append(
                {
                    **context,
                    "request_index": len(self._search_trace) + 1,
                    "candidate_source_sha256": CandidateCache.key(source),
                    "evaluation_kind": "scheduler_skip_duplicate",
                    "candidate_previously_tested": True,
                    "accepted": accepted,
                    "reason": "skipped_duplicate",
                }
            )

    def _classify_result(
        self,
        source: str,
        run: RunResult,
        result: OracleResult,
        *,
        context: dict[str, object] | None = None,
        evaluation_kind: str,
        record_history: bool,
    ) -> bool:
        assert self._baseline is not None
        accepted = result.interesting and self.oracle.same_failure(self._baseline, result)
        if self.trace:
            if accepted:
                reason = "failure_preserved"
            elif not result.interesting:
                reason = "oracle_not_interesting"
            else:
                reason = "fingerprint_mismatch"
            self._search_trace.append(
                {
                    **(context or {}),
                    "request_index": len(self._search_trace) + 1,
                    "candidate_source_sha256": CandidateCache.key(source),
                    "evaluation_kind": evaluation_kind,
                    "candidate_previously_tested": evaluation_kind != "oracle_execution",
                    "accepted": accepted,
                    "reason": reason,
                }
            )
        if accepted and record_history:
            self._history.append(
                {
                    "evaluation": self._evaluations,
                    "loc": len(source.splitlines()),
                    "duration_seconds": run.duration_seconds,
                    "fingerprint": result.fingerprint,
                }
            )
        elif not accepted:
            self._rejected_transformations += 1
        return accepted

    def _preserves_failure(
        self,
        source: str,
        *,
        context: dict[str, object] | None = None,
    ) -> bool:
        run, result = self._evaluate(source)
        return self._classify_result(
            source,
            run,
            result,
            context=context,
            evaluation_kind=self._last_evaluation_kind,
            record_history=True,
        )

    def _parallel_execute(
        self, source: str
    ) -> tuple[RunResult, OracleResult, float, float]:
        with self._concurrency_lock:
            self._active_concurrency += 1
            self._peak_concurrency = max(self._peak_concurrency, self._active_concurrency)
        try:
            return self._execute_uncached(source)
        finally:
            with self._concurrency_lock:
                self._active_concurrency -= 1

    def _evaluate_source_batch(
        self,
        sources: list[str],
        contexts: list[dict[str, object]],
        *,
        jobs: int,
    ) -> list[bool]:
        if not sources:
            return []
        if jobs <= 1 or callable(getattr(self.oracle, "evaluate_source", None)):
            outcomes: list[bool] = []
            for source, context in zip(sources, contexts):
                run, result = self._evaluate(source)
                outcomes.append(
                    self._classify_result(
                        source,
                        run,
                        result,
                        context=context,
                        evaluation_kind=self._last_evaluation_kind,
                        record_history=False,
                    )
                )
            return outcomes

        outcomes: list[bool | None] = [None] * len(sources)
        source_positions: dict[str, list[int]] = {}
        unique_positions: list[int] = []
        for index, source in enumerate(sources):
            cached = self._lookup_cached(source)
            if cached is not None:
                run, result = cached
                outcomes[index] = self._classify_result(
                    source,
                    run,
                    result,
                    context=contexts[index],
                    evaluation_kind=self._last_evaluation_kind,
                    record_history=False,
                )
                continue
            positions = source_positions.setdefault(source, [])
            positions.append(index)
            if len(positions) == 1:
                unique_positions.append(index)

        known_successes = [index for index, outcome in enumerate(outcomes) if outcome]
        earliest_known_success = min(known_successes) if known_successes else None
        submit_positions = [
            index
            for index in unique_positions
            if earliest_known_success is None or index < earliest_known_success
        ]
        self._cache_misses += len(submit_positions)
        self._candidates_submitted += len(submit_positions)
        payloads: dict[int, tuple[RunResult, OracleResult, float, float]] = {}
        cancelled_positions: set[int] = set()
        earliest_success = earliest_known_success

        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="reproreduce") as executor:
            futures: dict[int, Future] = {
                index: executor.submit(self._parallel_execute, sources[index])
                for index in submit_positions
            }
            for index in submit_positions:
                future = futures[index]
                if future.cancelled():
                    cancelled_positions.add(index)
                    continue
                payload = future.result()
                payloads[index] = payload
                run, result, _, _ = payload
                self._candidates_completed += 1
                accepted = self._classify_result(
                    sources[index],
                    run,
                    result,
                    context=contexts[index],
                    evaluation_kind="oracle_execution",
                    record_history=False,
                )
                outcomes[index] = accepted
                if accepted and (earliest_success is None or index < earliest_success):
                    earliest_success = index
                    for later_index, later_future in futures.items():
                        if later_index > index and later_future.cancel():
                            cancelled_positions.add(later_index)
            for index, future in futures.items():
                if index in payloads or index in cancelled_positions:
                    continue
                if future.cancelled():
                    cancelled_positions.add(index)
                    continue
                payloads[index] = future.result()
                run, result, _, _ = payloads[index]
                self._candidates_completed += 1
                outcomes[index] = self._classify_result(
                    sources[index],
                    run,
                    result,
                    context=contexts[index],
                    evaluation_kind="oracle_execution",
                    record_history=False,
                )

        self._candidates_cancelled += len(cancelled_positions)
        for index, payload in payloads.items():
            self._commit_evaluation(sources[index], payload)
        for source, positions in source_positions.items():
            first = positions[0]
            for duplicate in positions[1:]:
                outcomes[duplicate] = outcomes[first]

        if earliest_success is None:
            self._useful_executions += len(payloads)
        else:
            self._speculative_executions += sum(
                1 for index in payloads if index > earliest_success
            )
            self._useful_executions += sum(
                1 for index in payloads if index <= earliest_success
            )
        return [bool(outcome) for outcome in outcomes]

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
            candidate_test = _SessionCandidateTest(
                self,
                deduplicate=self.deduplicate,
                structural_deduplicate=self.structural_deduplicate,
                jobs=self.jobs,
            )
            reduced_source = self.source
            for _ in range(MAX_REDUCTION_PASSES):
                before = reduced_source

                if self.strategy == "dependency":
                    reduced_source, dependency_history = reduce_dependency(
                        reduced_source,
                        candidate_test,
                        [],
                        self._record_dependency_candidate,
                        record_analysis=self._record_dependency_analysis,
                    )
                    self._history.extend(dependency_history)

                reduced_source, ast_history = reduce_top_level_statements(
                    reduced_source, candidate_test
                )
                self._history.extend(ast_history)
                reduced_source, module_history = reduce_sequential_modules(
                    reduced_source, candidate_test
                )
                self._history.extend(module_history)
                reduced_source, module_list_history = reduce_module_lists(
                    reduced_source, candidate_test
                )
                self._history.extend(module_list_history)
                reduced_source, tensor_history = reduce_tensor_constructors(
                    reduced_source, candidate_test
                )
                self._history.extend(tensor_history)
                reduced_source, cleanup_history = reduce_top_level_statements(
                    reduced_source, candidate_test
                )
                self._history.extend(cleanup_history)

                if self.strategy == "dependency":
                    reduced_source, dependency_history = reduce_dependency(
                        reduced_source,
                        candidate_test,
                        [],
                        self._record_dependency_candidate,
                        record_analysis=self._record_dependency_analysis,
                    )
                    self._history.extend(dependency_history)

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
                search_trace=self._search_trace,
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
            "scheduler_requests": self._scheduler_requests,
            "unique_source_candidates": len(self._scheduler_sources),
            "unique_candidate_sources": self._candidate_runs,
            "oracle_executions": self._candidate_runs,
            "skipped_duplicate_candidates": self._skipped_duplicate_candidates,
            "skipped_structural_states": self._skipped_state_candidates,
            "no_op_skips": self._no_op_skips,
            "syntax_skips": self._syntax_skips,
            "duplicate_candidate_sources": len(self._cache_hit_sources),
            "cache_hits": self._cache_hits,
            "memory_cache_hits": self._memory_cache_hits,
            "sqlite_cache_hits": self._sqlite_cache_hits,
            "memory_cache_entries": len(self._memory_cache),
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
            "jobs": self.jobs,
            "candidates_submitted": self._candidates_submitted,
            "candidates_completed": self._candidates_completed,
            "candidates_cancelled": self._candidates_cancelled,
            "speculative_executions": self._speculative_executions,
            "useful_executions": self._useful_executions,
            "peak_concurrency": max(self._peak_concurrency, 1),
            "strategy": self.strategy,
            **self._dependency_metrics,
        }
