from __future__ import annotations

import random
from dataclasses import dataclass

from .config import TensorConfig
from .program import Operation, Program, TensorSpec


_EDGE_DIMS = (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33)
_BROADCAST_DIMS = (2, 3, 7, 8, 15, 16, 17)


@dataclass(frozen=True)
class ValueInfo:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BroadcastCase:
    program: Program
    configs: tuple[TensorConfig, ...]


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


class StructuredBroadcastGenerator:
    """Generate deterministic broadcast/layout coverage cells rather than generic chains."""

    _PATTERNS = (
        "scalar_tensor",
        "row_matrix",
        "column_matrix",
        "singleton_middle",
        "multi_axis",
        "chained",
    )
    _DTYPES = ("float32", "float64", "bfloat16")
    _LAYOUT_MODES = ("contiguous", "lhs_noncontiguous", "rhs_noncontiguous", "both_noncontiguous")
    _BINARY_OPS = ("add", "sub", "mul", "div", "maximum", "minimum", "where")
    _POST_OPS = ("unary", "sum", "reshape", "transpose", "permute", "slice")

    def __init__(self, seed: int, *, mode: str = "forward"):
        self.seed = seed
        self.random = random.Random(seed)
        self.mode = mode

    def generate(self) -> BroadcastCase:
        pattern = self._PATTERNS[self.seed % len(self._PATTERNS)]
        dtype = self._DTYPES[(self.seed // len(self._PATTERNS)) % len(self._DTYPES)]
        layout_mode = self._LAYOUT_MODES[
            (self.seed // (len(self._PATTERNS) * len(self._DTYPES))) % len(self._LAYOUT_MODES)
        ]
        post_kind = self._POST_OPS[(self.seed // 2) % len(self._POST_OPS)]
        shapes = self._shapes(pattern)
        layouts = self._layouts(shapes, layout_mode)
        configs = tuple(
            TensorConfig(shape=shape, dtype=dtype, requires_grad=self.mode == "gradient", layout=layout)
            for shape, layout in zip(shapes, layouts)
        )
        specs = tuple(config.to_spec(name) for name, config in zip(self._names(len(shapes)), configs))
        values = [ValueInfo(spec.name, spec.shape) for spec in specs]
        operations: list[Operation] = []
        binary_name = self._BINARY_OPS[(self.seed // 3) % len(self._BINARY_OPS)]
        if pattern == "chained":
            first_shape = self._broadcast_shape(values[0].shape, values[1].shape)
            operations.append(self._broadcast_operation(binary_name, values[0].name, values[1].name))
            values.append(ValueInfo("v0", first_shape))
            second_shape = self._broadcast_shape(values[-1].shape, values[2].shape)
            operations.append(self._broadcast_operation("mul", values[-1].name, values[2].name))
            values.append(ValueInfo("v1", second_shape))
        else:
            output_shape = self._broadcast_shape(values[0].shape, values[1].shape)
            operations.append(self._broadcast_operation(binary_name, values[0].name, values[1].name))
            values.append(ValueInfo("v0", output_shape))
        current = values[-1]
        operations.append(self._post_operation(post_kind, current, len(operations)))
        values.append(ValueInfo(f"v{len(operations) - 1}", self._post_shape(post_kind, current.shape)))
        metadata = (
            ("broadcast_pattern", pattern),
            ("broadcast_shape", current.shape),
            ("output_shape", values[-1].shape),
            ("lhs_shape", shapes[0]),
            ("rhs_shape", shapes[1]),
            ("input_shapes", shapes),
            ("input_layouts", layouts),
            ("dtype", dtype),
            ("post_op", post_kind),
            ("mode", self.mode),
        )
        program = Program(
            inputs=specs,
            operations=tuple(operations),
            output=values[-1].name,
            differentiable=True,
            metadata=metadata,
        )
        return BroadcastCase(program=program, configs=configs)

    def _shapes(self, pattern: str) -> tuple[tuple[int, ...], ...]:
        m = _BROADCAST_DIMS[(self.seed // 7) % len(_BROADCAST_DIMS)]
        n = _BROADCAST_DIMS[(self.seed // 11) % len(_BROADCAST_DIMS)]
        b = _BROADCAST_DIMS[(self.seed // 13) % len(_BROADCAST_DIMS)]
        if pattern == "scalar_tensor":
            return (), (n,)
        if pattern == "row_matrix":
            return (1, n), (m, n)
        if pattern == "column_matrix":
            return (m, 1), (m, n)
        if pattern == "singleton_middle":
            return (b, 1, n), (b, m, n)
        if pattern == "multi_axis":
            return (1, m, 1), (b, m, n)
        if pattern == "chained":
            return (1, m, n), (b, 1, n), (b, m, 1)
        raise AssertionError(pattern)

    def _layouts(self, shapes: tuple[tuple[int, ...], ...], mode: str) -> tuple[str, ...]:
        layouts = ["contiguous"] * len(shapes)
        if mode in {"lhs_noncontiguous", "both_noncontiguous"}:
            layouts[0] = self._noncontiguous_layout(shapes[0])
        if mode in {"rhs_noncontiguous", "both_noncontiguous"} and len(shapes) > 1:
            layouts[1] = self._noncontiguous_layout(shapes[1])
        if mode == "both_noncontiguous" and len(shapes) > 2:
            layouts[2] = self._noncontiguous_layout(shapes[2])
        return tuple(layouts)

    @staticmethod
    def _noncontiguous_layout(shape: tuple[int, ...]) -> str:
        if shape and shape[-1] > 1:
            return "slice"
        if len(shape) >= 2 and shape[0] > 1 and shape[1] > 1:
            return "transpose"
        return "contiguous"

    @staticmethod
    def _names(count: int) -> tuple[str, ...]:
        return tuple(chr(ord("x") + index) for index in range(count))

    @staticmethod
    def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for left_dim, right_dim in zip(reversed(left), reversed(right)):
            if left_dim == 1:
                result.append(right_dim)
            elif right_dim == 1 or left_dim == right_dim:
                result.append(left_dim)
            else:
                raise ValueError(f"incompatible broadcast shapes: {left} and {right}")
        longer = left if len(left) > len(right) else right
        result.extend(reversed(longer[: abs(len(left) - len(right))]))
        return tuple(reversed(result))

    @staticmethod
    def _broadcast_operation(name: str, left: str, right: str) -> Operation:
        if name == "div":
            return Operation.create(name, (left, right), safe_denominator=True)
        return Operation.create(name, (left, right))

    @staticmethod
    def _post_operation(kind: str, value: ValueInfo, index: int) -> Operation:
        if kind == "unary":
            return Operation.create("sin", (value.name,))
        if kind == "sum":
            return Operation.create("sum", (value.name,))
        if kind == "reshape":
            numel = 1
            for dimension in value.shape:
                numel *= dimension
            return Operation.create("reshape", (value.name,), shape=(numel,))
        if kind == "transpose":
            if len(value.shape) < 2:
                return Operation.create("sin", (value.name,))
            return Operation.create("transpose", (value.name,), dim0=0, dim1=1)
        if kind == "permute":
            if len(value.shape) != 3:
                return Operation.create("sin", (value.name,))
            return Operation.create("permute", (value.name,), dims=(2, 1, 0))
        if kind == "slice":
            return Operation.create("slice", (value.name,), step=2)
        raise AssertionError(kind)

    @staticmethod
    def _post_shape(kind: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        if kind in {"unary", "transpose", "permute"}:
            if kind == "transpose" and len(shape) >= 2:
                return (shape[1], shape[0], *shape[2:])
            if kind == "permute" and len(shape) == 3:
                return tuple(shape[index] for index in (2, 1, 0))
            return shape
        if kind == "sum":
            return ()
        if kind == "reshape":
            numel = 1
            for dimension in shape:
                numel *= dimension
            return (numel,)
        if kind == "slice":
            return (*shape[:-1], max(1, (shape[-1] + 1) // 2))
        raise AssertionError(kind)
