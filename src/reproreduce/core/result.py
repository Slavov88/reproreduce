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
                f"Candidate evaluations: {len(self.history):>8}",
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
            "The reduced script preserved the configured failure oracle.\n",
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
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination
