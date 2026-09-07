from __future__ import annotations

import random
from dataclasses import dataclass

from .config import TensorConfig
from .program import Operation, Program, TensorSpec


_EDGE_DIMS = (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33)


@dataclass(frozen=True)
class ValueInfo:
    name: str
    shape: tuple[int, ...]


class TensorProgramGenerator:
    """Generate short, shape-valid programs from a deterministic operator grammar."""

    def __init__(self, seed: int, *, max_operations: int = 6):
        self.random = random.Random(seed)
        self.max_operations = max_operations

    def generate(self, config: TensorConfig | None = None) -> Program:
        if config is None:
            rank = self.random.choice((1, 2, 3))
            shape = tuple(self.random.choice(_EDGE_DIMS) for _ in range(rank))
            inputs = (TensorSpec("x", shape), TensorSpec("y", shape))
        else:
            shape = config.shape
            inputs = (config.to_spec("x"), config.to_spec("y"))
        values = [ValueInfo("x", shape), ValueInfo("y", shape)]
        operations: list[Operation] = []

        for index in range(self.random.randint(2, self.max_operations)):
            operation, output_shape = self._choose_operation(values)
            operations.append(operation)
            values.append(ValueInfo(f"v{index}", output_shape))

        return Program(
            inputs=inputs,
            operations=tuple(operations),
            output=values[-1].name,
            differentiable=True,
        )

    def _choose_operation(self, values: list[ValueInfo]) -> tuple[Operation, tuple[int, ...]]:
        choices = ["unary", "binary", "sum", "reshape"]
        if any(len(value.shape) >= 2 for value in values):
            choices.append("transpose")
        if any(len(value.shape) == 3 for value in values):
            choices.append("permute")
        if any(value.shape for value in values):
            choices.append("slice")
        if self._has_matching_shapes(values):
            choices.append("cat")
        kind = self.random.choice(choices)

        if kind == "unary":
            value = self.random.choice(values)
            return Operation.create(self.random.choice(("sin", "cos", "exp", "relu")), (value.name,)), value.shape
        if kind == "binary":
            left = self.random.choice(values)
            compatible = [value for value in values if value.shape == left.shape]
            right = self.random.choice(compatible)
            name = self.random.choice(("add", "sub", "mul", "div"))
            return Operation.create(name, (left.name, right.name)), left.shape
        if kind == "sum":
            value = self.random.choice(values)
            return Operation.create(self.random.choice(("sum", "mean")), (value.name,)), ()
        if kind == "reshape":
            value = self.random.choice(values)
            numel = 1
            for dimension in value.shape:
                numel *= dimension
            new_shape = (numel,) if numel > 1 else (1,)
            return Operation.create("reshape", (value.name,), shape=new_shape), new_shape
        if kind == "transpose":
            value = self.random.choice([item for item in values if len(item.shape) >= 2])
            return Operation.create("transpose", (value.name,), dim0=0, dim1=1), (
                value.shape[1],
                value.shape[0],
                *value.shape[2:],
            )
        if kind == "permute":
            value = self.random.choice([item for item in values if len(item.shape) == 3])
            dims = (2, 1, 0)
            return Operation.create("permute", (value.name,), dims=dims), tuple(value.shape[index] for index in dims)
        if kind == "slice":
            value = self.random.choice([item for item in values if item.shape])
            shape = (*value.shape[:-1], max(1, (value.shape[-1] + 1) // 2))
            return Operation.create("slice", (value.name,), step=2), shape
        if kind == "cat":
            candidates = [value for value in values if value.shape]
            left = self.random.choice(candidates)
            compatible = [value for value in candidates if value.shape == left.shape]
            right = self.random.choice(compatible)
            output_shape = (*left.shape[:-1], left.shape[-1] + right.shape[-1])
            return Operation.create("cat", (left.name, right.name), dim=len(left.shape) - 1), output_shape
        raise AssertionError(f"unknown operation kind: {kind}")

    @staticmethod
    def _has_matching_shapes(values: list[ValueInfo]) -> bool:
        return any(value.shape for value in values) and len(values) >= 2
