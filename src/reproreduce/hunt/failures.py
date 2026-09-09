from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"]+[\\/])*[^\s'\"]+")
_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
_NODE_SUFFIX_RE = re.compile(r"(?P<node>permute|slice|copy|index_put|expand|clone|empty|view)_\d+")
_SPACE_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"placeholders=\{([^}]*)\}")


def normalize_failure_message(message: str) -> str:
    """Normalize unstable paths, addresses, and generated node suffixes."""
    normalized = _PATH_RE.sub("<PATH>", str(message).replace("\\", "/"))
    normalized = _ADDRESS_RE.sub("<ADDR>", normalized)
    normalized = _NODE_SUFFIX_RE.sub(r"\g<node>", normalized)
    normalized = _PLACEHOLDER_RE.sub(
        lambda match: "placeholders={" + ", ".join(sorted(item.strip() for item in match.group(1).split(","))) + "}",
        normalized,
    )
    normalized = re.sub(r"line \d+", "line <LINE>", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def failure_cause(message: str) -> str:
    """Return the stable innermost error line from a wrapped compiler message."""
    lines = [line.strip() for line in str(message).replace("\r", "").splitlines() if line.strip()]
    preferred = (
        "HuntTimeout:",
        "AssertionError:",
        "NotImplementedError:",
        "LoweringException:",
        "BackendCompilerFailed:",
        "RuntimeError:",
        "Unsupported:",
    )
    for prefix in preferred:
        for line in lines:
            if line.startswith(prefix):
                return normalize_failure_message(line)
    return normalize_failure_message(lines[0] if lines else "unknown failure")


def failure_stage(
    exception_type: str | None,
    message: str | None,
    *,
    phase: str | None = None,
    traceback_text: str | None = None,
) -> str:
    """Classify the deepest observable failure stage without guessing from op names."""
    text = f"{exception_type or ''}\n{message or ''}\n{traceback_text or ''}".lower()
    if "hunttimeout" in text or "case exceeded" in text:
        return "harness_timeout"
    if "triton" in text and ("compile" in text or "compiler" in text):
        return "triton_compile"
    if (
        "aotautograd" in text
        or "aot autograd" in text
        or "assert_functional_graph" in text
        or "functional_utils.py" in text
    ):
        return "aot_autograd"
    if "functionaliz" in text:
        return "functionalization"
    if (
        "torch._dynamo.exc.unsupported" in text
        or "graph break" in text
        or "dynamo unsupported" in text
    ):
        return "dynamo_capture"
    if "loweringexception" in text or "n=copy_" in text or "inductor" in text and "assertionerror" in text:
        return "inductor_lowering" if traceback_text else "backend_wrapped_unknown"
    if "codegen" in text or "code generation" in text or "cppcompile" in text:
        return "code_generation"
    if phase == "compile":
        return "compile_wrapper"
    if phase == "runtime":
        return "compiled_runtime"
    return "unknown"


def relevant_frame(message: str) -> str | None:
    """Extract a stable subsystem frame when a traceback is present."""
    for raw_line in str(message).splitlines():
        line = raw_line.strip()
        if "torch/_inductor/" in line or "torch\\_inductor\\" in line:
            return normalize_failure_message(line)
        if "torch._inductor" in line:
            return normalize_failure_message(line)
    return None


def failure_metadata(
    exception_type: str | None,
    message: str | None,
    *,
    phase: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    combined_message = f"{message or ''}\n{traceback_text or ''}"
    cause = failure_cause(combined_message)
    stage = failure_stage(
        exception_type,
        message,
        phase=phase,
        traceback_text=traceback_text,
    )
    normalized_type = exception_type or "UnknownException"
    fingerprint = f"{stage}|{normalized_type}|{cause}"
    data: dict[str, Any] = {
        "failure_stage": stage,
        "failure_cause": cause,
        "failure_fingerprint": fingerprint,
        "normalized_exception_message": normalize_failure_message(message or ""),
    }
    frame = relevant_frame(combined_message)
    if frame is not None:
        data["relevant_frame"] = frame
    if traceback_text:
        data["failure_traceback"] = traceback_text
    return data


def load_failure_clusters(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("cases", []) if isinstance(payload, dict) else payload
    return cluster_failure_records(
        [record for record in records if record.get("classification") not in {"PASS"}]
    )


def cluster_failure_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cluster campaign records using stored or backward-compatible fingerprints."""
    clusters: dict[str, dict[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        coverage = record.get("coverage") or {}
        exception_type = metadata.get("exception_type")
        message = metadata.get("exception_message", "")
        diagnostic = failure_metadata(
            exception_type,
            message,
            phase=record.get("failure_phase"),
        )
        fingerprint = metadata.get("failure_fingerprint") or diagnostic["failure_fingerprint"]
        cluster = clusters.setdefault(
            fingerprint,
            {
                "fingerprint": fingerprint,
                "count": 0,
                "seeds": [],
                "exception_types": {},
                "failure_stages": {},
                "mutation_ops": {},
                "dtypes": {},
                "layouts": {},
                "examples": [],
            },
        )
        cluster["count"] += 1
        cluster["seeds"].append(record.get("seed"))
        for key, value in (
            ("exception_types", exception_type or "unknown"),
            ("failure_stages", metadata.get("failure_stage") or diagnostic["failure_stage"]),
            ("mutation_ops", (coverage.get("mutation") or {}).get("op", "unknown")),
            ("dtypes", coverage.get("dtype", "unknown")),
            ("layouts", coverage.get("layout", "unknown")),
        ):
            cluster[key][value] = cluster[key].get(value, 0) + 1
        if len(cluster["examples"]) < 3:
            cluster["examples"].append(
                {
                    "seed": record.get("seed"),
                    "cause": diagnostic["failure_cause"],
                    "exception_message": message,
                }
            )
    return clusters
