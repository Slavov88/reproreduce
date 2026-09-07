from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import TensorConfigGenerator
from .confirm import ConfirmationPolicy, FindingDeduplicator, FindingFingerprint, confirm
from .execute import OutcomeClass, ProgramExecutor
from .generator import TensorProgramGenerator


@dataclass
class HuntStats:
    cases_generated: int = 0
    valid_eager: int = 0
    compiled_cases: int = 0
    eager_invalid: int = 0
    compile_failures: int = 0
    compiled_runtime_failures: int = 0
    candidate_only_errors: int = 0
    infrastructure_failures: int = 0
    forward_mismatches: int = 0
    gradient_mismatches: int = 0
    nonfinite_comparisons: int = 0
    timeouts: int = 0
    nondeterministic: int = 0
    unique_findings: int = 0
    classifications: Counter[str] = field(default_factory=Counter)
    cases: list[dict[str, Any]] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("cases", None)
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_generated": self.cases_generated,
            "valid_eager": self.valid_eager,
            "compiled_cases": self.compiled_cases,
            "eager_invalid": self.eager_invalid,
            "compile_failures": self.compile_failures,
            "compiled_runtime_failures": self.compiled_runtime_failures,
            "candidate_only_errors": self.candidate_only_errors,
            "infrastructure_failures": self.infrastructure_failures,
            "forward_mismatches": self.forward_mismatches,
            "gradient_mismatches": self.gradient_mismatches,
            "nonfinite_comparisons": self.nonfinite_comparisons,
            "timeouts": self.timeouts,
            "nondeterministic": self.nondeterministic,
            "unique_findings": self.unique_findings,
            "classifications": dict(self.classifications),
            "cases": self.cases,
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
        config = TensorConfigGenerator(case_seed).generate()
        if mode == "gradient" and not config.requires_grad:
            config = replace(config, requires_grad=True)
        program = TensorProgramGenerator(case_seed).generate(config)
        configs = (config, config)
        stats.cases_generated += 1

        def run_case():
            return selected_executor.run(program, configs, mode=mode, input_seed=case_seed)

        result = run_case()
        stats.classifications[result.classification.value] += 1

        if result.classification == OutcomeClass.EAGER_ERROR:
            stats.eager_invalid += 1
        elif result.classification != OutcomeClass.INFRASTRUCTURE_ERROR:
            stats.valid_eager += 1
        if result.classification not in {
            OutcomeClass.EAGER_ERROR,
            OutcomeClass.COMPILE_FAILURE,
            OutcomeClass.INFRASTRUCTURE_ERROR,
        }:
            stats.compiled_cases += 1
        if result.classification in {
            OutcomeClass.COMPILE_FAILURE,
            OutcomeClass.COMPILED_RUNTIME_FAILURE,
        }:
            stats.candidate_only_errors += 1
        if result.classification == OutcomeClass.COMPILE_FAILURE:
            stats.compile_failures += 1
        elif result.classification == OutcomeClass.COMPILED_RUNTIME_FAILURE:
            stats.compiled_runtime_failures += 1
        elif result.classification == OutcomeClass.INFRASTRUCTURE_ERROR:
            stats.infrastructure_failures += 1
        elif result.classification == OutcomeClass.FORWARD_MISMATCH:
            stats.forward_mismatches += 1
        elif result.classification == OutcomeClass.GRADIENT_MISMATCH:
            stats.gradient_mismatches += 1
        elif result.classification == OutcomeClass.NONFINITE_COMPARISON:
            stats.nonfinite_comparisons += 1
        elif result.classification == OutcomeClass.TIMEOUT:
            stats.timeouts += 1
        elif result.classification == OutcomeClass.NONDETERMINISTIC:
            stats.nondeterministic += 1

        confirmation_result = None
        finding_id = None
        mismatch_classes = {OutcomeClass.FORWARD_MISMATCH, OutcomeClass.GRADIENT_MISMATCH}
        if result.classification in mismatch_classes:
            policy = confirmation or ConfirmationPolicy(attempts=5, min_successes=5)
            confirmation_result = confirm(
                run_case,
                expected=result.classification,
                policy=policy,
                expected_fingerprint=(
                    result.oracle_result.fingerprint if result.oracle_result is not None else None
                ),
            )
            if confirmation_result.reproducible:
                finding = FindingFingerprint.from_case(result, program, configs)
                finding_id = finding.identity
                if deduplicator.add(finding):
                    stats.unique_findings += 1

        stats.cases.append(
            {
                "seed": case_seed,
                "program_id": program.identity,
                "program": program.to_dict(),
                "configs": [config.to_dict() for config in configs],
                "input_seed": case_seed,
                "backend": backend,
                "mode": mode,
                "classification": result.classification.value,
                "failure_phase": result.failure_phase,
                "fingerprint": result.oracle_result.fingerprint if result.oracle_result else None,
                "metadata": result.oracle_result.metadata if result.oracle_result else {},
                "confirmation": (
                    {
                        "attempts": confirmation_result.attempts,
                        "successes": confirmation_result.successes,
                        "required_successes": confirmation_result.required_successes,
                        "classifications": [item.value for item in confirmation_result.classifications],
                        "reproducible": confirmation_result.reproducible,
                    }
                    if confirmation_result is not None
                    else None
                ),
                "finding_id": finding_id,
            }
        )
    return stats, deduplicator


def write_campaign_report(stats: HuntStats, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    return path
