"""Nested statement stress benchmark for the recursive AST reducer."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Record:
    identifier: int
    value: float
    active: bool


def decode_record(raw: tuple[int, float, bool]) -> Record:
    identifier, value, active = raw
    return Record(identifier=identifier, value=float(value), active=active)


def load_records() -> list[Record]:
    raw = [(0, 1.0, True), (1, 2.0, False), (2, 3.0, True)]
    records = []
    for item in raw:
        records.append(decode_record(item))
    return records


def normalize_record(record: Record) -> Record:
    bounded = max(-10.0, min(10.0, record.value))
    return Record(record.identifier, bounded / 2.0, record.active)


def score_record(record: Record) -> float:
    base = math.sin(record.value) + math.cos(record.value / 2.0)
    adjustment = record.identifier * 0.01
    return base + adjustment


def format_record(record: Record, score: float) -> dict[str, object]:
    return {
        "id": record.identifier,
        "active": record.active,
        "score": round(score, 5),
        "label": "eligible" if record.active else "held",
    }


def audit_record(payload: dict[str, object]) -> bool:
    score = float(payload["score"])
    return bool(payload["active"]) and score > -2.0


def make_context() -> dict[str, object]:
    records = [normalize_record(item) for item in load_records()]
    payloads = []
    for record in records:
        payload = format_record(record, score_record(record))
        payload["audited"] = audit_record(payload)
        payloads.append(payload)
    return {"records": records, "payloads": payloads, "mode": "nested-stress"}


def feature_window(values: list[float], width: int = 3) -> list[float]:
    result = []
    for start in range(max(0, len(values) - width + 1)):
        window = values[start : start + width]
        result.append(sum(window) / max(1, len(window)))
    return result


def summarize_window(values: list[float]) -> dict[str, float]:
    window = feature_window(values)
    return {
        "mean": sum(window) / max(1, len(window)),
        "minimum": min(window, default=0.0),
        "maximum": max(window, default=0.0),
    }


def encode_identifier(identifier: int) -> str:
    prefix = "record"
    return f"{prefix}-{identifier:04d}"


def choose_bucket(value: float) -> str:
    if value < -1.0:
        return "low"
    if value > 1.0:
        return "high"
    return "middle"


def weighted_score(value: float, weight: float = 0.75) -> float:
    baseline = value * weight
    correction = (1.0 - weight) * math.tanh(value)
    return baseline + correction


def build_index(records: list[Record]) -> dict[str, Record]:
    index = {}
    for record in records:
        index[encode_identifier(record.identifier)] = record
    return index


def annotate_records(records: list[Record]) -> list[dict[str, object]]:
    annotations = []
    for record in records:
        annotations.append({
            "key": encode_identifier(record.identifier),
            "bucket": choose_bucket(record.value),
            "weighted": weighted_score(record.value),
        })
    return annotations


def fold_annotations(annotations: list[dict[str, object]]) -> float:
    total = 0.0
    for annotation in annotations:
        total += float(annotation["weighted"])
    return total


def diagnostics_for_context(context: dict[str, object]) -> dict[str, object]:
    records = context.get("records", [])
    values = [record.value for record in records if isinstance(record, Record)]
    annotations = annotate_records(values if values and isinstance(values[0], Record) else [])
    return {"window": summarize_window([float(item) for item in values]), "total": fold_annotations(annotations)}


def partition_records(records: list[Record]) -> tuple[list[Record], list[Record]]:
    active = []
    inactive = []
    for record in records:
        (active if record.active else inactive).append(record)
    return active, inactive


def record_magnitude(record: Record) -> float:
    return abs(record.value) + record.identifier * 0.001


def rank_records(records: list[Record]) -> list[Record]:
    return sorted(records, key=record_magnitude, reverse=True)


def merge_labels(records: list[Record]) -> str:
    labels = [choose_bucket(record.value) for record in records]
    return ",".join(labels)


def stable_checksum(records: list[Record]) -> int:
    checksum = 0
    for record in records:
        checksum = (checksum * 31 + record.identifier) % 100003
    return checksum


def build_feature_vector(record: Record) -> list[float]:
    return [record.value, float(record.identifier), float(record.active)]


def flatten_features(records: list[Record]) -> list[float]:
    values = []
    for record in records:
        values.extend(build_feature_vector(record))
    return values


def compare_records(left: Record, right: Record) -> int:
    left_key = (left.active, left.value, left.identifier)
    right_key = (right.active, right.value, right.identifier)
    return (left_key > right_key) - (left_key < right_key)


def compact_context(context: dict[str, object]) -> dict[str, object]:
    records = context.get("records", [])
    if not isinstance(records, list):
        return {}
    return {"count": len(records), "checksum": stable_checksum(records)}


def choose_representative(records: list[Record]) -> Record | None:
    ranked = rank_records(records)
    return ranked[0] if ranked else None


def serialize_records(records: list[Record]) -> list[str]:
    return [f"{record.identifier}:{record.value:.3f}:{int(record.active)}" for record in records]


def parse_serialized(values: list[str]) -> list[tuple[int, float, bool]]:
    parsed = []
    for value in values:
        identifier, number, active = value.split(":")
        parsed.append((int(identifier), float(number), bool(int(active))))
    return parsed


def accumulate_scores(records: list[Record]) -> float:
    total = 0.0
    for record in records:
        total += score_record(record)
    return total


def describe_records(records: list[Record]) -> dict[str, object]:
    active, inactive = partition_records(records)
    return {"active": len(active), "inactive": len(inactive), "score": accumulate_scores(records)}


def apply_threshold(records: list[Record], threshold: float) -> list[Record]:
    return [record for record in records if record.value >= threshold]


def record_lookup(records: list[Record]) -> dict[int, Record]:
    return {record.identifier: record for record in records}


def grouped_values(records: list[Record]) -> dict[bool, list[float]]:
    groups: dict[bool, list[float]] = {True: [], False: []}
    for record in records:
        groups[record.active].append(record.value)
    return groups


def validate_context(context: dict[str, object]) -> bool:
    return bool(context.get("records")) and "payloads" in context


def render_context(context: dict[str, object]) -> str:
    mode = str(context.get("mode", "unknown"))
    payloads = context.get("payloads", [])
    return f"{mode}:{len(payloads)}"


def context_checksum(context: dict[str, object]) -> int:
    rendered = render_context(context)
    return sum(ord(character) for character in rendered)


def normalized_values(records: list[Record]) -> list[float]:
    values = [record.value for record in records]
    scale = max((abs(value) for value in values), default=1.0)
    return [value / max(scale, 1e-9) for value in values]


def record_signature(record: Record) -> str:
    return f"{record.identifier}|{record.value:.4f}|{record.active}"


def record_family(record: Record) -> str:
    return "active" if record.active else "inactive"


def recoverable_side_path(payload: dict[str, object]) -> float:
    try:
        value = float(payload["score"])
        if value < -100.0:
            raise ValueError("out of range")
        return value
    except (KeyError, ValueError):
        return 0.0


def reproduce() -> None:
    context = make_context()
    payloads = context["payloads"]
    for position, payload in enumerate(payloads):
        if payload["active"]:
            try:
                score = recoverable_side_path(payload)
                if position >= 0 and score > -2.0:
                    raise RuntimeError("REPROREDUCE_NESTED_TARGET")
            except ValueError:
                fallback = recoverable_side_path(payload)
                context["fallback"] = fallback
        else:
            context["skipped"] = position


def main() -> None:
    settings = {"enabled": True, "retries": 3, "label": "research"}
    if settings["enabled"]:
        reproduce()


if __name__ == "__main__":
    main()
