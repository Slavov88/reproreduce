from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parent


def _write(name: str, lines: list[str]) -> None:
    path = ROOT / name
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"{path}: {sum(bool(line.strip()) for line in lines)} nonblank lines")


def build_large_exception() -> list[str]:
    lines = [
        "\"\"\"Deterministic large synthetic exception benchmark.\"\"\"",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "import math",
        "import statistics",
        "from itertools import pairwise",
        "",
        "WINDOW = 5",
        "SCALE = 1.25",
        "DEFAULT_RECORDS = tuple(range(12))",
        "",
    ]
    for index in range(24):
        lines.extend(
            [
                f"def feature_stage_{index:02d}(values: list[float]) -> list[float]:",
                "    result = []",
                "    for position, value in enumerate(values):",
                f"        offset = (position % {index + 2}) * 0.125",
                f"        adjusted = (value + offset) * {1.0 + (index % 4) / 10:.1f}",
                "        if position % 2:",
                "            adjusted -= 0.25",
                "        result.append(round(adjusted, 6))",
                "    return result",
                "",
            ]
        )
    lines.extend(
        [
            "@dataclass",
            "class FeatureLedger:",
            "    values: list[float]",
            "    labels: list[str]",
            "",
            "    def add(self, value: float, label: str) -> None:",
            "        self.values.append(float(value))",
            "        self.labels.append(label.strip().lower())",
            "",
            "    def summary(self) -> dict[str, float | int]:",
            "        total = sum(self.values)",
            "        return {",
            "            \"count\": len(self.values),",
            "            \"mean\": statistics.fmean(self.values) if self.values else 0.0,",
            "            \"total\": total,",
            "        }",
            "",
            "def prepare_records(records: tuple[int, ...]) -> list[float]:",
            "    values = [float(item) for item in records]",
            "    for stage in (feature_stage_00, feature_stage_03, feature_stage_07):",
            "        values = stage(values)",
            "    return values",
            "",
            "def summarize_records(values: list[float]) -> dict[str, object]:",
            "    ledger = FeatureLedger([], [])",
            "    for index, value in enumerate(values):",
            "        ledger.add(value, f\"feature-{index}\")",
            "    summary = ledger.summary()",
            "    summary[\"ready\"] = bool(values) and summary[\"count\"] >= 0",
            "    return summary",
            "",
            "def compute_diagnostics(values: list[float]) -> dict[str, float]:",
            "    differences = [right - left for left, right in pairwise(values)]",
            "    return {",
            "        \"variance\": statistics.pvariance(values) if values else 0.0,",
            "        \"span\": max(values, default=0.0) - min(values, default=0.0),",
            "        \"drift\": sum(differences),",
            "    }",
            "",
            "def load_reference_table() -> dict[str, float]:",
            "    table = {}",
            "    for index in range(16):",
            "        table[f\"item-{index}\"] = math.sin(index / WINDOW) * SCALE",
            "    return table",
            "",
            "def main() -> None:",
            "    records = DEFAULT_RECORDS",
            "    prepared = prepare_records(records)",
            "    report = summarize_records(prepared)",
            "    diagnostics = compute_diagnostics(prepared)",
            "    reference = load_reference_table()",
            "    _ = (report, diagnostics, reference)",
            "    raise RuntimeError(\"REPROREDUCE_LARGE_TARGET\")",
            "",
            "if __name__ == \"__main__\":",
            "    main()",
        ]
    )
    return lines


def build_generated_pytorch() -> list[str]:
    lines = [
        "\"\"\"Deterministic generated PyTorch-heavy reduction benchmark.\"\"\"",
        "from __future__ import annotations",
        "",
        "import math",
        "from dataclasses import dataclass",
        "",
        "import torch",
        "from torch import nn",
        "",
        "torch.manual_seed(0)",
        "FEATURES = 8",
        "",
    ]
    for index in range(28):
        lines.extend(
            [
                f"def tensor_feature_{index:02d}(value: torch.Tensor) -> torch.Tensor:",
                f"    scale = {1.0 + index / 20:.2f}",
                f"    bias = {index % 3 - 1}.0",
                "    transformed = value * scale + bias",
                "    transformed = torch.tanh(transformed)",
                "    if transformed.ndim > 1:",
                "        transformed = transformed + transformed.mean(dim=-1, keepdim=True)",
                "    return transformed",
                "",
            ]
        )
    for index in range(12):
        lines.extend(
            [
                f"class FeatureBlock{index:02d}(nn.Module):",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                f"        self.layer = nn.Linear(FEATURES, FEATURES, bias={index % 2 == 0})",
                "",
                "    def forward(self, value: torch.Tensor) -> torch.Tensor:",
                "        return torch.relu(self.layer(value))",
                "",
            ]
        )
    lines.extend(
        [
            "@dataclass",
            "class BatchMetadata:",
            "    rows: int",
            "    columns: int",
            "    tag: str",
            "",
            "def build_model() -> nn.Module:",
            "    return nn.Sequential(FeatureBlock00(), FeatureBlock01(), FeatureBlock02())",
            "",
            "def build_batch() -> tuple[torch.Tensor, BatchMetadata]:",
            "    values = torch.arange(32, dtype=torch.float32).reshape(4, FEATURES) / 10.0",
            "    metadata = BatchMetadata(rows=4, columns=FEATURES, tag=\"benchmark\")",
            "    return values, metadata",
            "",
            "def collect_auxiliary_statistics(value: torch.Tensor) -> dict[str, float]:",
            "    return {",
            "        \"mean\": float(value.mean()),",
            "        \"std\": float(value.std()),",
            "        \"energy\": float(torch.square(value).sum()),",
            "    }",
            "",
            "def main() -> None:",
            "    model = build_model()",
            "    batch, metadata = build_batch()",
            "    activated = model(batch)",
            "    statistics = collect_auxiliary_statistics(activated)",
            "    shape_guard = activated.shape[0] == metadata.rows",
            "    _ = (statistics, shape_guard)",
            "    raise RuntimeError(\"REPROREDUCE_GENERATED_TORCH_TARGET\")",
            "",
            "if __name__ == \"__main__\":",
            "    main()",
        ]
    )
    return lines


if __name__ == "__main__":
    _write("large_exception.py", build_large_exception())
    _write("generated_pytorch.py", build_generated_pytorch())
