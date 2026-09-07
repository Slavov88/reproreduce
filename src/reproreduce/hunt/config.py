from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .program import TensorSpec


@dataclass(frozen=True)
class TensorConfig:
    shape: tuple[int, ...]
    dtype: str = "float32"
    requires_grad: bool = True
    layout: str = "contiguous"

    def to_spec(self, name: str) -> TensorSpec:
        return TensorSpec(
            name=name,
            shape=self.shape,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            layout=self.layout,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "layout": self.layout,
        }

    def materialize(self) -> Any:
        import torch

        value = torch.randn(
            self.shape,
            dtype=getattr(torch, self.dtype),
            requires_grad=self.requires_grad,
        )
        if self.layout == "transpose":
            if value.ndim < 2:
                raise ValueError("transpose layout requires rank >= 2")
            return value.transpose(0, 1)
        if self.layout == "slice":
            if value.ndim < 1:
                raise ValueError("slice layout requires rank >= 1")
            return value[..., ::2]
        if self.layout != "contiguous":
            raise ValueError(f"unknown layout: {self.layout}")
        return value


class TensorConfigGenerator:
    def __init__(self, seed: int):
        self.random = random.Random(seed)

    def generate(self) -> TensorConfig:
        rank = self.random.choice((1, 2, 3))
        shape = tuple(self.random.choice((1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33)) for _ in range(rank))
        dtype = self.random.choice(("float32", "bfloat16", "float64"))
        layout_choices = ["contiguous"]
        if rank >= 2:
            layout_choices.append("transpose")
        if rank >= 1:
            layout_choices.append("slice")
        return TensorConfig(
            shape=shape,
            dtype=dtype,
            requires_grad=self.random.choice((True, False)),
            layout=self.random.choice(layout_choices),
        )
