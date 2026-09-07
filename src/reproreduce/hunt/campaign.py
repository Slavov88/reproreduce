from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import TensorConfig
from .confirm import ConfirmationPolicy, FindingDeduplicator, FindingFingerprint, confirm
from .execute import OutcomeClass, ProgramExecutor
from .generator import TensorProgramGenerator


@dataclass
class HuntStats:
    cases_generated: int = 0
    valid_eager: int = 0
    compiled_cases: int = 0
    candidate_only_errors: int = 0
    forward_mismatches: int = 0
    gradient_mismatches: int = 0
    timeouts: int = 0
    nondeterministic: int = 0
    unique_findings: int = 0
    classifications: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_generated": self.cases_generated,
            "valid_eager": self.valid_eager,
            "compiled_cases": self.compiled_cases,
            "candidate_only_errors": self.candidate_only_errors,
            "forward_mismatches": self.forward_mismatches,
            "gradient_mismatches": self.gradient_mismatches,
            "timeouts": self.timeouts,
            "nondeterministic": self.nondeterministic,
            "unique_findings": self.unique_findings,
            "classifications": dict(self.classifications),
        }


def run_campaign(
    *,
    cases: int,
    seed: int = 0,
    backend: str = "aot_eager",
    mode: str = "forward",
    confirmation: ConfirmationPolicy | None = None,
    executor: ProgramExecutor | None = None,
) -> tuple[HuntStats, FindingDeduplicator]:
    if cases < 0:
        raise ValueError("cases must be non-negative")
    if mode not in {"forward", "gradient"}:
        raise ValueError("mode must be 'forward' or 'gradient'")
    selected_executor = executor or ProgramExecutor(backend=backend)
    stats = HuntStats()
    deduplicator = FindingDeduplicator()

    for case_seed in range(seed, seed + cases):
        program = TensorProgramGenerator(case_seed).generate()
        configs = tuple(
            TensorConfig(
                shape=spec.shape,
                dtype=spec.dtype,
                requires_grad=spec.requires_grad,
                layout="contiguous",
            )
            for spec in program.inputs
        )
        stats.cases_generated += 1
        result = selected_executor.run(program, configs, mode=mode)
        stats.classifications[result.classification.value] += 1
        if result.classification != OutcomeClass.EAGER_ERROR:
            stats.valid_eager += 1
        if result.classification not in {OutcomeClass.EAGER_ERROR, OutcomeClass.COMPILED_ERROR}:
            stats.compiled_cases += 1
        if result.classification == OutcomeClass.COMPILED_ERROR:
            stats.candidate_only_errors += 1
        elif result.classification == OutcomeClass.FORWARD_MISMATCH:
            stats.forward_mismatches += 1
        elif result.classification == OutcomeClass.GRADIENT_MISMATCH:
            stats.gradient_mismatches += 1

        mismatch_classes = {OutcomeClass.FORWARD_MISMATCH, OutcomeClass.GRADIENT_MISMATCH}
        if result.classification in mismatch_classes:
            policy = confirmation or ConfirmationPolicy(attempts=5, min_successes=5)
            confirmed = confirm(
                lambda: selected_executor.run(program, configs, mode=mode),
                expected=result.classification,
                policy=policy,
            )
            if confirmed.reproducible:
                finding = FindingFingerprint.from_case(result, program, configs)
                if deduplicator.add(finding):
                    stats.unique_findings += 1
    return stats, deduplicator


def write_campaign_report(stats: HuntStats, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
