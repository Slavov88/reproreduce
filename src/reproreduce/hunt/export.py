from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from ..core.result import ReductionResult


def export_finding(
    result: ReductionResult,
    output: str | Path,
    *,
    finding: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "repro.py").write_text(result.reduced_source, encoding="utf-8")

    metrics = result.metrics
    report = "\n".join(
        [
            "# ReproReduce finding",
            "",
            "## Reproduction",
            "",
            "The reduced program is intended to be run with `python repro.py`.",
            "",
            f"- Original LOC: {result.original_loc}",
            f"- Reduced LOC: {result.reduced_loc}",
            f"- Reduced candidate exit code: {result.reduced_run.returncode}",
            "",
            "## Reduction metrics",
            "",
            f"- Candidate runs: {metrics.get('candidate_runs', 0)}",
            f"- Cache hits: {metrics.get('cache_hits', 0)}",
            f"- Wall time: {metrics.get('total_reduction_wall_time', 0.0):.3f} seconds",
            "",
        ]
    )
    (destination / "report.md").write_text(report, encoding="utf-8")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        environment["pytorch"] = torch.__version__
    except ImportError:
        pass
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    finding_data = {
        "original_loc": result.original_loc,
        "reduced_loc": result.reduced_loc,
        "metrics": result.metrics,
        "finding": finding or {},
    }
    (destination / "finding.json").write_text(json.dumps(finding_data, indent=2, default=str) + "\n", encoding="utf-8")
    return destination
