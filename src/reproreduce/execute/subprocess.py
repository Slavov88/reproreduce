from pathlib import Path

from ..core.run import RunResult, run_python


def execute_candidate(source: str, *, cwd: Path, timeout: float) -> RunResult:
    return run_python(source, cwd=cwd, timeout=timeout)
