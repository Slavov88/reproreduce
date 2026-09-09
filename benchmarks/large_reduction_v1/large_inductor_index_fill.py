"""Large realistic wrapper around the historical PyTorch #178952 failure."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def normalize_batch(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean()
    scale = value.std().clamp_min(1e-5)
    return centered / scale


def make_feature_table(rows: int, columns: int) -> torch.Tensor:
    base = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return normalize_batch(base / max(columns, 1))


def unrelated_projection(value: torch.Tensor) -> torch.Tensor:
    weight = torch.eye(value.shape[-1], dtype=value.dtype)
    return value @ weight + 0.125


def reshape_for_report(value: torch.Tensor) -> torch.Tensor:
    flattened = value.flatten()
    return flattened[: value.numel()].reshape(value.shape)


def compute_histogram(value: torch.Tensor) -> dict[str, float]:
    return {
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "energy": float(torch.square(value).sum()),
    }


def prepare_metadata(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "numel": value.numel(),
        "contiguous": value.is_contiguous(),
    }


def make_mask(value: torch.Tensor) -> torch.Tensor:
    return value > value.mean()


def apply_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, value, torch.zeros_like(value))


def summarize_rows(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dim=-1, keepdim=True)


def add_residual(value: torch.Tensor) -> torch.Tensor:
    return value + torch.sin(value) * 0.01


def mix_channels(value: torch.Tensor) -> torch.Tensor:
    return value.flip(-1) * 0.5 + value * 0.5


def detach_for_logging(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu()


def covariance_proxy(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered


def normalize_columns(value: torch.Tensor) -> torch.Tensor:
    norms = value.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return value / norms


def add_position_encoding(value: torch.Tensor) -> torch.Tensor:
    position = torch.arange(value.shape[-1], dtype=value.dtype)
    return value + position * 0.001


def residual_mix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.lerp(left, right, 0.25)


def project_for_storage(value: torch.Tensor) -> torch.Tensor:
    flattened = value.reshape(-1)
    return flattened.clone().reshape_as(value)


def estimate_entropy(value: torch.Tensor) -> float:
    probabilities = torch.softmax(value.flatten(), dim=0)
    return float(-(probabilities * probabilities.log()).sum())


def make_attention_mask(rows: int, columns: int) -> torch.Tensor:
    return torch.ones((rows, columns), dtype=torch.bool).triu(diagonal=0)


def clip_extremes(value: torch.Tensor) -> torch.Tensor:
    lower = value.quantile(0.05)
    upper = value.quantile(0.95)
    return value.clamp(lower, upper)


def channel_statistics(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.mean(dim=0).flatten()]


def serialize_shape(value: torch.Tensor) -> str:
    return "x".join(str(item) for item in value.shape)


def quantize_for_preview(value: torch.Tensor) -> torch.Tensor:
    scaled = (value * 16.0).round()
    return scaled.clamp(-128.0, 127.0) / 16.0


def compare_batches(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left - right).abs()
    return {"max": float(difference.max()), "mean": float(difference.mean())}


def format_report(metadata: dict[str, object]) -> str:
    shape = metadata.get("shape", ())
    dtype = metadata.get("dtype", "unknown")
    return f"shape={shape}; dtype={dtype}"


def build_encoder() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, 3),
    )


class ReportHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.projection(value)))


class PreprocessingGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = build_encoder()
        self.head = ReportHead()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


@dataclass
class ExperimentConfig:
    rows: int = 2
    columns: int = 3
    index_value: tuple[int, int] = (0, 1)
    fill_value: float = 0.5


def build_input(config: ExperimentConfig) -> torch.Tensor:
    torch.manual_seed(0)
    value = torch.randn((config.rows, config.columns), dtype=torch.float32)
    return value


def preprocess(value: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
    normalized = normalize_batch(value)
    projected = unrelated_projection(normalized)
    reshaped = reshape_for_report(projected)
    mask = make_mask(reshaped)
    masked = apply_mask(reshaped, mask)
    metadata = prepare_metadata(masked)
    return add_residual(masked), metadata


def build_auxiliary_outputs(value: torch.Tensor) -> dict[str, object]:
    histogram = compute_histogram(value)
    rows = summarize_rows(value)
    mixed = mix_channels(value)
    logged = detach_for_logging(mixed)
    return {"histogram": histogram, "rows": rows, "logged": logged}


def target_model(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The non-contiguous-view index_fill_ operation from PyTorch #178952."""
    unused = value + 1.0
    view = value.transpose(0, 1)
    view.index_fill_(-1, index, 0.5)
    check = view.contiguous()
    return value


def run_experiment(config: ExperimentConfig) -> torch.Tensor:
    value = build_input(config)
    processed, metadata = preprocess(value)
    graph = PreprocessingGraph()
    auxiliary = build_auxiliary_outputs(processed)
    encoded = graph(processed)
    _ = (metadata, auxiliary, encoded)
    index = torch.tensor(config.index_value, dtype=torch.long)
    eager_input = value.clone()
    eager_output = target_model(eager_input, index)
    compiled = torch.compile(target_model, backend="inductor")
    compiled_input = eager_output.clone()
    return compiled(compiled_input, index)


def main() -> None:
    config = ExperimentConfig()
    output = run_experiment(config)
    print(output)


if __name__ == "__main__":
    main()
