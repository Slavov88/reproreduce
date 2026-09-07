from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_python(
    source: str,
    *,
    cwd: Path,
    timeout: float,
    environment: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> RunResult:
    """Run candidate source without allowing a crash to kill the reducer."""
    cwd = cwd.resolve()
    executable = python_executable or sys.executable
    env = os.environ.copy()
    if environment:
        env.update(environment)

    candidate_path: Path | None = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py", prefix=".reproreduce-", dir=cwd, delete=False
        ) as handle:
            handle.write(source)
            candidate_path = Path(handle.name)

        command = (executable, "-u", str(candidate_path))
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired as error:
            process.kill()
            stdout, stderr = process.communicate()
            timeout_message = f"\nReproReduce timeout after {timeout:g}s.\n"
            stderr = (stderr or "") + timeout_message
            timed_out = True
        return RunResult(
            command=command,
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=time.perf_counter() - started,
            timed_out=timed_out,
        )
    finally:
        if candidate_path is not None:
            try:
                candidate_path.unlink()
            except OSError:
                pass
