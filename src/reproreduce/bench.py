from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .api import reduce
from .core.run import RunResult, run_python
from .oracle.exception import ExceptionOracle


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    filename: str
    oracle_type: str
    message: str
    classification: str
    description: str
    compiler_failure: bool = False


SPECS = (
    BenchmarkSpec(
        name="large_exception",
        filename="large_exception.py",
        oracle_type="RuntimeError",
        message="REPROREDUCE_LARGE_TARGET",
        classification="SYNTHETIC",
        description="Large deterministic Python program with a buried exception.",
    ),
    BenchmarkSpec(
        name="large_inductor_index_fill",
        filename="large_inductor_index_fill.py",
        oracle_type="AssertionError",
        message="n=copy_",
        classification="HISTORICAL_REAL_BUG",
        description="Large wrapper around the verified PyTorch #178952 failure.",
        compiler_failure=True,
    ),
    BenchmarkSpec(
        name="nested_python",
        filename="nested_python.py",
        oracle_type="RuntimeError",
        message="REPROREDUCE_NESTED_TARGET",
        classification="SYNTHETIC",
        description="Nested function, branch, loop, and exception-handler stress test.",
    ),
    BenchmarkSpec(
        name="generated_pytorch",
        filename="generated_pytorch.py",
        oracle_type="RuntimeError",
        message="REPROREDUCE_GENERATED_TORCH_TARGET",
        classification="GENERATED_REALISTIC",
        description="Deterministic PyTorch-heavy generated program with a target exception.",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nonblank_loc(source: str) -> int:
    return sum(bool(line.strip()) for line in source.splitlines())


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _environment() -> dict[str, object]:
    environment: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch
    except ImportError:
        environment["torch"] = None
    else:
        environment["torch"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
    return environment


class BenchmarkOracle:
    """Use exact exception matching except for stable compiler target signatures."""

    def __init__(self, spec: BenchmarkSpec):
        self.base = ExceptionOracle(exception_type=spec.oracle_type, message_regex=spec.message)
        self.compiler_failure = spec.compiler_failure
        self.message_regex = spec.message

    def evaluate(self, run: RunResult):
        return self.base.evaluate(run)

    def same_failure(self, baseline, candidate) -> bool:
        if not self.base.same_failure(baseline, candidate):
            if not self.compiler_failure or not baseline.interesting or not candidate.interesting:
                return False
            if baseline.metadata.get("exception_type") != candidate.metadata.get("exception_type"):
                return False
            if baseline.metadata.get("relevant_frames") != candidate.metadata.get("relevant_frames"):
                return False
            candidate_message = str(candidate.metadata.get("message_signature") or "")
            if self.message_regex is None or re.search(self.message_regex, candidate_message) is None:
                return False
        return True


def _oracle(spec: BenchmarkSpec) -> BenchmarkOracle:
    return BenchmarkOracle(spec)


def _stable_fingerprint(spec: BenchmarkSpec, result) -> str | None:
    if result.fingerprint is None:
        return None
    metadata = result.metadata
    if spec.compiler_failure:
        frames = tuple(metadata.get("relevant_frames") or ())
        stage = frames[-1] if frames else "unknown-stage"
        return json.dumps(
            {
                "kind": "compiler_exception",
                "exception_type": metadata.get("exception_type"),
                "target": spec.message,
                "stage": stage,
            },
            sort_keys=True,
        )
    return result.fingerprint


def _run_exported(path: Path, timeout: float) -> RunResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=path.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return RunResult(
            command=(sys.executable, "-u", str(path)),
            returncode=None,
            stdout=error.stdout or "",
            stderr=(error.stderr or "") + f"\nReproReduce timeout after {timeout:g}s.\n",
            duration_seconds=time.perf_counter() - started,
            timed_out=True,
        )
    return RunResult(
        command=(sys.executable, "-u", str(path)),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_seconds=time.perf_counter() - started,
    )


def _failure_metadata(oracle: BenchmarkOracle, run: RunResult) -> tuple[Any, dict[str, Any]]:
    result = oracle.evaluate(run)
    return result, result.metadata


def run_one(
    spec: BenchmarkSpec,
    *,
    root: Path,
    workspace: Path,
    repeat: int,
    timeout: float,
) -> dict[str, object]:
    source_path = root / "benchmarks" / "large_reduction_v1" / spec.filename
    source = source_path.read_text(encoding="utf-8")
    oracle = _oracle(spec)
    original_run = run_python(source, cwd=source_path.parent, timeout=timeout)
    baseline = oracle.evaluate(original_run)
    record: dict[str, object] = {
        "name": spec.name,
        "repeat": repeat,
        "classification": spec.classification,
        "description": spec.description,
        "oracle": f"exception:{spec.oracle_type}",
        "source": str(source_path.relative_to(root)).replace("\\", "/"),
        "source_sha256": _sha256(source),
        "original_physical_loc": len(source.splitlines()),
        "original_nonblank_loc": _nonblank_loc(source),
        "baseline_interesting": baseline.interesting,
        "fingerprint_before": _stable_fingerprint(spec, baseline),
        "fingerprint_before_metadata": baseline.metadata,
        "environment": _environment(),
        "timeout_seconds": timeout,
        "termination_normal": False,
        "export_success": False,
        "standalone_repro_success": False,
    }
    if not baseline.interesting:
        record.update(
            {
                "status": "NOT FRESHLY EXECUTED",
                "limitation": "The original fixture did not match its configured oracle in this environment.",
                "original_returncode": original_run.returncode,
                "original_stderr_tail": original_run.stderr[-4000:],
            }
        )
        return record

    run_workspace = workspace / f"{spec.name}-repeat-{repeat}"
    run_workspace.mkdir(parents=True, exist_ok=True)
    cache_path = run_workspace / "candidates.sqlite3"
    export_path = run_workspace / "export"
    started = time.perf_counter()
    try:
        result = reduce(
            source_path,
            oracle=oracle,
            timeout=timeout,
            cache=cache_path,
        )
    except Exception as error:  # benchmark records must preserve failures for later diagnosis
        record.update(
            {
                "status": "REDUCTION_ERROR",
                "limitation": f"{type(error).__name__}: {error}",
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
        return record

    wall_time = time.perf_counter() - started
    reduced_source = result.reduced_source
    reduced_result = oracle.evaluate(result.reduced_run)
    reduced_result_object, reduced_metadata = _failure_metadata(oracle, result.reduced_run)
    fingerprint_after = _stable_fingerprint(spec, reduced_result_object)
    fingerprint_preserved = oracle.same_failure(baseline, reduced_result)
    export_success = False
    standalone_success = False
    standalone_result: RunResult | None = None
    export_error: str | None = None
    try:
        result.export(export_path)
        export_success = (export_path / "repro.py").is_file()
        if export_success:
            standalone_result = _run_exported(export_path / "repro.py", timeout)
            standalone_oracle = oracle.evaluate(standalone_result)
            standalone_success = oracle.same_failure(baseline, standalone_oracle)
    except Exception as error:  # keep the reduction result even if packaging fails
        export_error = f"{type(error).__name__}: {error}"

    candidate_runs = int(result.metrics.get("candidate_runs", 0))
    cache_hits = int(result.metrics.get("cache_hits", 0))
    candidate_evaluations = candidate_runs + cache_hits
    profile_keys = {
        "candidate_call_seconds",
        "subprocess_source_write_seconds",
        "subprocess_startup_seconds",
        "subprocess_wait_seconds",
        "oracle_seconds",
        "cache_lookup_seconds",
        "cache_write_seconds",
        "reducer_bookkeeping_seconds",
        "candidate_requests",
        "unique_candidate_sources",
        "duplicate_candidate_sources",
    }
    record.update(
        {
            "status": "COMPUTATIONALLY VERIFIED" if fingerprint_preserved and standalone_success else "OBSERVED",
            "termination_normal": True,
            "reduced_physical_loc": len(reduced_source.splitlines()),
            "reduced_nonblank_loc": _nonblank_loc(reduced_source),
            "reduction_fraction": 1.0 - _nonblank_loc(reduced_source) / _nonblank_loc(source),
            "reduction_percent": 100.0 * (1.0 - _nonblank_loc(reduced_source) / _nonblank_loc(source)),
            "candidate_runs": candidate_runs,
            "candidate_evaluations": candidate_evaluations,
            "cache_hits": cache_hits,
            "unique_candidate_sources": result.metrics.get("unique_candidate_sources", candidate_runs),
            "duplicate_candidate_sources": result.metrics.get("duplicate_candidate_sources", 0),
            "profiling": {key: result.metrics.get(key, 0.0) for key in sorted(profile_keys)},
            "cache_hit_ratio": cache_hits / candidate_evaluations if candidate_evaluations else 0.0,
            "wall_time_seconds": wall_time,
            "candidate_execution_seconds": result.metrics.get("total_candidate_execution_time", 0.0),
            "median_candidate_execution_seconds": result.metrics.get("median_candidate_execution_time", 0.0),
            "accepted_transformations": result.metrics.get("accepted_transformations", 0),
            "rejected_transformations": result.metrics.get("rejected_transformations", 0),
            "fingerprint_after": fingerprint_after,
            "fingerprint_after_metadata": reduced_metadata,
            "fingerprint_preserved": fingerprint_preserved,
            "export_success": export_success,
            "standalone_repro_success": standalone_success,
            "reduced_source_sha256": _sha256(reduced_source),
            "standalone_returncode": standalone_result.returncode if standalone_result else None,
            "standalone_stderr_tail": standalone_result.stderr[-4000:] if standalone_result else None,
            "export_error": export_error,
            "limitation": (
                "Nested structure was reduced recursively; retained control-flow shape is recorded in the exported source."
                if spec.name == "nested_python"
                else None
            ),
        }
    )
    if not fingerprint_preserved:
        record["limitation"] = "The final reduced candidate did not preserve the baseline failure fingerprint."
    elif not standalone_success:
        record["limitation"] = "The reduced source preserved the in-process oracle, but exported standalone verification failed."
    return record


def _determinism(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["name"]), []).append(record)
    output: dict[str, object] = {}
    for name, items in groups.items():
        output[name] = {
            "repeats": len(items),
            "reduced_nonblank_loc": [item.get("reduced_nonblank_loc") for item in items],
            "reduced_source_sha256": [item.get("reduced_source_sha256") for item in items],
            "candidate_runs": [item.get("candidate_runs") for item in items],
            "fingerprint_preserved": [item.get("fingerprint_preserved") for item in items],
            "exact_reduced_source_match": len({item.get("reduced_source_sha256") for item in items}) <= 1,
            "exact_candidate_count_match": len({item.get("candidate_runs") for item in items}) <= 1,
        }
    return output


def run_benchmarks(
    *,
    root: Path,
    names: list[str] | None = None,
    repeats: int = 1,
    timeout: float = 30.0,
    workspace: Path | None = None,
) -> dict[str, object]:
    selected = [spec for spec in SPECS if names is None or spec.name in names]
    unknown = sorted(set(names or ()) - {spec.name for spec in SPECS})
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    workspace = workspace or root / ".benchmarks" / "large-reduction-v1"
    workspace.mkdir(parents=True, exist_ok=True)
    records = [
        run_one(spec, root=root, workspace=workspace, repeat=repeat, timeout=timeout)
        for spec in selected
        for repeat in range(1, repeats + 1)
    ]
    return {
        "schema_version": 1,
        "suite": "large-reduction-v1",
        "status_labels": ["OBSERVED", "COMPUTATIONALLY VERIFIED", "NOT FRESHLY EXECUTED"],
        "commit": _git_commit(root),
        "command": "python -m reproreduce.bench",
        "environment": _environment(),
        "benchmarks": [asdict(spec) for spec in selected],
        "records": records,
        "determinism": _determinism(records),
    }


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the large-input ReproReduce benchmark suite.")
    parser.add_argument("--benchmark", action="append", dest="names", choices=[spec.name for spec in SPECS])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--root", type=Path, default=_repo_root())
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/large_benchmarks_v1.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    workspace = args.workspace
    if workspace is not None and not workspace.is_absolute():
        workspace = root / workspace
    report = run_benchmarks(
        root=root,
        names=args.names,
        repeats=args.repeats,
        timeout=args.timeout,
        workspace=workspace,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(report["records"]), "commit": report["commit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
