from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

from .run import RunResult


@dataclass
class ReductionResult:
    original_source: str
    reduced_source: str
    original_run: RunResult
    reduced_run: RunResult
    history: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, int | float] = field(default_factory=dict)

    @property
    def original_loc(self) -> int:
        return len(self.original_source.splitlines())

    @property
    def reduced_loc(self) -> int:
        return len(self.reduced_source.splitlines())

    def summary(self) -> str:
        return "\n".join(
            [
                "Original",
                "--------------------------------",
                f"Python LOC: {self.original_loc:>18}",
                "",
                "Reduced",
                "--------------------------------",
                f"Python LOC: {self.reduced_loc:>18}",
                "",
                "Failure preserved: yes",
                f"Candidate runs: {self.metrics.get('candidate_runs', 0):>14}",
                f"Cache hits: {self.metrics.get('cache_hits', 0):>17}",
                f"Reduction wall time: {self.metrics.get('total_reduction_wall_time', 0.0):>8.2f}s",
            ]
        )

    def export(self, output: str | Path) -> Path:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "repro.py").write_text(self.reduced_source, encoding="utf-8")
        (destination / "README.md").write_text(
            "# Reduced reproducer\n\n"
            "Run with:\n\n```bash\npython repro.py\n```\n\n"
            f"Original LOC: {self.original_loc}\n\n"
            f"Reduced LOC: {self.reduced_loc}\n\n"
            "The reduced script preserved the configured failure oracle. "
            "See `reduction.json` for run status, commands, and captured evidence.\n",
            encoding="utf-8",
        )
        (destination / "environment.json").write_text(
            json.dumps(
                {"python": platform.python_version(), "platform": platform.platform()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "reduction.json").write_text(
            json.dumps(
                {
                    "original_loc": self.original_loc,
                    "reduced_loc": self.reduced_loc,
                    "history": self.history,
                    "metrics": self.metrics,
                    "original_run": _run_metadata(self.original_run),
                    "reduced_run": _run_metadata(self.reduced_run),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination


def _run_metadata(run: RunResult) -> dict[str, object]:
    return {
        "command": list(run.command),
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "duration_seconds": run.duration_seconds,
        "stderr_tail": run.stderr[-4000:],
    }
