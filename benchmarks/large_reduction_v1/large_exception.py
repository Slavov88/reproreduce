"""Deterministic large synthetic exception benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from itertools import pairwise

WINDOW = 5
SCALE = 1.25
DEFAULT_RECORDS = tuple(range(12))

def feature_stage_00(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 2) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_01(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 3) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_02(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 4) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_03(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 5) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_04(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 6) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_05(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 7) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_06(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 8) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_07(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 9) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_08(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 10) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_09(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 11) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_10(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 12) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_11(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 13) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_12(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 14) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_13(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 15) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_14(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 16) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_15(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 17) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_16(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 18) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_17(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 19) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_18(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 20) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_19(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 21) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_20(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 22) * 0.125
        adjusted = (value + offset) * 1.0
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_21(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 23) * 0.125
        adjusted = (value + offset) * 1.1
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_22(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 24) * 0.125
        adjusted = (value + offset) * 1.2
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

def feature_stage_23(values: list[float]) -> list[float]:
    result = []
    for position, value in enumerate(values):
        offset = (position % 25) * 0.125
        adjusted = (value + offset) * 1.3
        if position % 2:
            adjusted -= 0.25
        result.append(round(adjusted, 6))
    return result

@dataclass
class FeatureLedger:
    values: list[float]
    labels: list[str]

    def add(self, value: float, label: str) -> None:
        self.values.append(float(value))
        self.labels.append(label.strip().lower())

    def summary(self) -> dict[str, float | int]:
        total = sum(self.values)
        return {
            "count": len(self.values),
            "mean": statistics.fmean(self.values) if self.values else 0.0,
            "total": total,
        }

def prepare_records(records: tuple[int, ...]) -> list[float]:
    values = [float(item) for item in records]
    for stage in (feature_stage_00, feature_stage_03, feature_stage_07):
        values = stage(values)
    return values

def summarize_records(values: list[float]) -> dict[str, object]:
    ledger = FeatureLedger([], [])
    for index, value in enumerate(values):
        ledger.add(value, f"feature-{index}")
    summary = ledger.summary()
    summary["ready"] = bool(values) and summary["count"] >= 0
    return summary

def compute_diagnostics(values: list[float]) -> dict[str, float]:
    differences = [right - left for left, right in pairwise(values)]
    return {
        "variance": statistics.pvariance(values) if values else 0.0,
        "span": max(values, default=0.0) - min(values, default=0.0),
        "drift": sum(differences),
    }

def load_reference_table() -> dict[str, float]:
    table = {}
    for index in range(16):
        table[f"item-{index}"] = math.sin(index / WINDOW) * SCALE
    return table

def main() -> None:
    records = DEFAULT_RECORDS
    prepared = prepare_records(records)
    report = summarize_records(prepared)
    diagnostics = compute_diagnostics(prepared)
    reference = load_reference_table()
    _ = (report, diagnostics, reference)
    raise RuntimeError("REPROREDUCE_LARGE_TARGET")

if __name__ == "__main__":
    main()
