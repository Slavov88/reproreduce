"""Deterministic generated PyTorch-heavy reduction benchmark."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

torch.manual_seed(0)
FEATURES = 8

def tensor_feature_00(value: torch.Tensor) -> torch.Tensor:
    scale = 1.00
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_01(value: torch.Tensor) -> torch.Tensor:
    scale = 1.05
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_02(value: torch.Tensor) -> torch.Tensor:
    scale = 1.10
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_03(value: torch.Tensor) -> torch.Tensor:
    scale = 1.15
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_04(value: torch.Tensor) -> torch.Tensor:
    scale = 1.20
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_05(value: torch.Tensor) -> torch.Tensor:
    scale = 1.25
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_06(value: torch.Tensor) -> torch.Tensor:
    scale = 1.30
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_07(value: torch.Tensor) -> torch.Tensor:
    scale = 1.35
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_08(value: torch.Tensor) -> torch.Tensor:
    scale = 1.40
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_09(value: torch.Tensor) -> torch.Tensor:
    scale = 1.45
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_10(value: torch.Tensor) -> torch.Tensor:
    scale = 1.50
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_11(value: torch.Tensor) -> torch.Tensor:
    scale = 1.55
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_12(value: torch.Tensor) -> torch.Tensor:
    scale = 1.60
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_13(value: torch.Tensor) -> torch.Tensor:
    scale = 1.65
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_14(value: torch.Tensor) -> torch.Tensor:
    scale = 1.70
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_15(value: torch.Tensor) -> torch.Tensor:
    scale = 1.75
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_16(value: torch.Tensor) -> torch.Tensor:
    scale = 1.80
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_17(value: torch.Tensor) -> torch.Tensor:
    scale = 1.85
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_18(value: torch.Tensor) -> torch.Tensor:
    scale = 1.90
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_19(value: torch.Tensor) -> torch.Tensor:
    scale = 1.95
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_20(value: torch.Tensor) -> torch.Tensor:
    scale = 2.00
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_21(value: torch.Tensor) -> torch.Tensor:
    scale = 2.05
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_22(value: torch.Tensor) -> torch.Tensor:
    scale = 2.10
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_23(value: torch.Tensor) -> torch.Tensor:
    scale = 2.15
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_24(value: torch.Tensor) -> torch.Tensor:
    scale = 2.20
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_25(value: torch.Tensor) -> torch.Tensor:
    scale = 2.25
    bias = 0.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_26(value: torch.Tensor) -> torch.Tensor:
    scale = 2.30
    bias = 1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

def tensor_feature_27(value: torch.Tensor) -> torch.Tensor:
    scale = 2.35
    bias = -1.0
    transformed = value * scale + bias
    transformed = torch.tanh(transformed)
    if transformed.ndim > 1:
        transformed = transformed + transformed.mean(dim=-1, keepdim=True)
    return transformed

class FeatureBlock00(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock01(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock02(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock03(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock04(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock05(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock06(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock07(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock08(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock09(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock10(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

class FeatureBlock11(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, FEATURES, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer(value))

@dataclass
class BatchMetadata:
    rows: int
    columns: int
    tag: str

def build_model() -> nn.Module:
    return nn.Sequential(FeatureBlock00(), FeatureBlock01(), FeatureBlock02())

def build_batch() -> tuple[torch.Tensor, BatchMetadata]:
    values = torch.arange(32, dtype=torch.float32).reshape(4, FEATURES) / 10.0
    metadata = BatchMetadata(rows=4, columns=FEATURES, tag="benchmark")
    return values, metadata

def collect_auxiliary_statistics(value: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(value.mean()),
        "std": float(value.std()),
        "energy": float(torch.square(value).sum()),
    }

def main() -> None:
    model = build_model()
    batch, metadata = build_batch()
    activated = model(batch)
    statistics = collect_auxiliary_statistics(activated)
    shape_guard = activated.shape[0] == metadata.rows
    _ = (statistics, shape_guard)
    raise RuntimeError("REPROREDUCE_GENERATED_TORCH_TARGET")

if __name__ == "__main__":
    main()
