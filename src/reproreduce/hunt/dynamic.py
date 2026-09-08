from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .config import TensorConfig
from .program import Operation, Program, TensorSpec

Shape = tuple[int, ...]
StepShapes = tuple[Shape, ...]
ShapeTrace = tuple[StepShapes, ...]
ConfigTrace = tuple[tuple[TensorConfig, ...], ...]

_DYNAMIC_DIMS = (
    (4, 7, 16, 5),
    (3, 8, 13, 6),
    (2, 5, 9, 4),
    (5, 11, 3, 8),
)
_FIXED_DIMS = (4, 8, 3, 5)


@dataclass(frozen=True)
class DynamicCase:
    program: Program
    config_trace: ConfigTrace

    @property
    def shape_trace(self) -> ShapeTrace:
        return tuple(
            tuple(config.shape for config in step)
            for step in self.config_trace
        )

    @property
    def trace_length(self) -> int:
        return len(self.config_trace)

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program.to_dict(),
            "shape_trace": self.shape_trace,
            "config_trace": tuple(
                tuple(config.to_dict() for config in step)
                for step in self.config_trace
            ),
        }


class DynamicShapeGenerator:
    """Generate one program plus a compatible multi-shape input trace."""

    _PATTERNS = (
        "vary_dim0",
        "vary_dim0_broadcast_reduction",
        "vary_dim1_broadcast_reduction",
        "two_dims_broadcast_transform",
        "batch_broadcast_permute",
        "dynamic_reduction",
        "dynamic_slice_reduction",
        "dynamic_concat_reshape",
    )
    _DTYPES = ("float32", "float64", "bfloat16")
    _LAYOUT_MODES = (
        "contiguous",
        "lhs_slice",
        "rhs_slice",
        "both_slice",
        "transpose_pair",
    )

    def __init__(self, seed: int, *, mode: str = "forward", trace_length: int = 4):
        if mode not in {"forward", "gradient"}:
            raise ValueError("mode must be 'forward' or 'gradient'")
        if trace_length < 2:
            raise ValueError("dynamic traces require at least two shapes")
        self.seed = seed
        self.random = random.Random(seed)
        self.mode = mode
        self.trace_length = trace_length

    def generate(self) -> DynamicCase:
        pattern = self._PATTERNS[self.seed % len(self._PATTERNS)]
        dtype = self._DTYPES[(self.seed // len(self._PATTERNS)) % len(self._DTYPES)]
        layout_mode = self._LAYOUT_MODES[
            (self.seed // (len(self._PATTERNS) * len(self._DTYPES))) % len(self._LAYOUT_MODES)
        ]
        shape_trace, varying_dims = self._shape_trace(pattern)
        layouts = self._layouts(shape_trace[0], layout_mode)
        config_trace = tuple(
            tuple(
                TensorConfig(
                    shape=shape,
                    dtype=dtype,
                    requires_grad=self.mode == "gradient",
                    layout=layout,
                )
                for shape, layout in zip(step_shapes, layouts)
            )
            for step_shapes in shape_trace
        )

        names = tuple(chr(ord("x") + index) for index in range(len(layouts)))
        specs = tuple(config.to_spec(name) for name, config in zip(names, config_trace[0]))
        values = {name: list(step_shapes[index] for step_shapes in shape_trace) for index, name in enumerate(names)}
        operations: list[Operation] = []

        def append(
            operation: Operation,
            output_shapes: list[Shape],
        ) -> str:
            index = len(operations)
            operations.append(operation)
            name = f"v{index}"
            values[name] = output_shapes
            return name

        lhs, rhs = names[0], names[1]
        binary_name = ("add", "sub", "mul")[self.seed % 3]
        if pattern == "dynamic_concat_reshape":
            current = append(
                Operation.create("cat", (lhs, rhs), dim=0),
                [_cat_shape(left, right, dim=0) for left, right in zip(values[lhs], values[rhs])],
            )
            current = append(
                Operation.create("reshape", (current,), shape=(-1,)),
                [_reshape_flat(shape) for shape in values[current]],
            )
        else:
            current = append(
                self._binary_operation(binary_name, lhs, rhs),
                [_broadcast_shape(left, right) for left, right in zip(values[lhs], values[rhs])],
            )
            if pattern == "vary_dim0":
                current = append(
                    Operation.create("sin", (current,)),
                    list(values[current]),
                )
            elif pattern == "vary_dim0_broadcast_reduction":
                current = append(
                    Operation.create("sum", (current,), dim=0),
                    [_reduce_shape(shape, dim=0) for shape in values[current]],
                )
            elif pattern == "vary_dim1_broadcast_reduction":
                current = append(
                    Operation.create("mean", (current,), dim=1),
                    [_reduce_shape(shape, dim=1) for shape in values[current]],
                )
            elif pattern == "two_dims_broadcast_transform":
                current = append(
                    Operation.create("transpose", (current,), dim0=0, dim1=1),
                    [_transpose_shape(shape, 0, 1) for shape in values[current]],
                )
                current = append(
                    Operation.create("reshape", (current,), shape=(-1,)),
                    [_reshape_flat(shape) for shape in values[current]],
                )
            elif pattern == "batch_broadcast_permute":
                current = append(
                    Operation.create("permute", (current,), dims=(2, 1, 0)),
                    [_permute_shape(shape, (2, 1, 0)) for shape in values[current]],
                )
                current = append(
                    Operation.create("sum", (current,), dim=1),
                    [_reduce_shape(shape, dim=1) for shape in values[current]],
                )
            elif pattern == "dynamic_reduction":
                current = append(
                    Operation.create("amax", (current,), dim=1),
                    [_reduce_shape(shape, dim=1) for shape in values[current]],
                )
                current = append(
                    Operation.create("relu", (current,)),
                    list(values[current]),
                )
            elif pattern == "dynamic_slice_reduction":
                current = append(
                    Operation.create("slice", (current,), step=2),
                    [_slice_shape(shape) for shape in values[current]],
                )
                current = append(
                    Operation.create("mean", (current,), dim=1),
                    [_reduce_shape(shape, dim=1) for shape in values[current]],
                )
            else:
                raise AssertionError(pattern)

        output_trace = tuple(values[current])
        metadata = (
            ("family", "dynamic"),
            ("dynamic_compile", True),
            ("symbolic_pattern", pattern),
            ("shape_trace", shape_trace),
            ("output_shape_trace", output_trace),
            ("varying_dims", varying_dims),
            ("operation_family", _operation_family(pattern)),
            ("input_shapes", shape_trace[0]),
            ("input_layouts", layouts),
            ("dtype", dtype),
            ("mode", self.mode),
            ("trace_length", self.trace_length),
            ("zero_sized_dimensions", False),
        )
        program = Program(
            inputs=specs,
            operations=tuple(operations),
            output=current,
            differentiable=True,
            metadata=metadata,
        )
        return DynamicCase(program=program, config_trace=config_trace)

    def _shape_trace(self, pattern: str) -> tuple[ShapeTrace, tuple[int, ...]]:
        dims = _DYNAMIC_DIMS[(self.seed // len(self._PATTERNS)) % len(_DYNAMIC_DIMS)]
        fixed = _FIXED_DIMS[(self.seed // 5) % len(_FIXED_DIMS)]
        if pattern == "vary_dim0":
            return tuple((((value,), (value,)) for value in dims[: self.trace_length])), (0,)
        if pattern == "vary_dim0_broadcast_reduction":
            return tuple((((value, fixed), (1, fixed)) for value in dims[: self.trace_length])), (0,)
        if pattern == "vary_dim1_broadcast_reduction":
            return tuple((((fixed, value), (fixed, 1)) for value in dims[: self.trace_length])), (1,)
        if pattern == "two_dims_broadcast_transform":
            other = _DYNAMIC_DIMS[(self.seed // 11) % len(_DYNAMIC_DIMS)]
            return tuple(
                (((left, right), (1, right)) for left, right in zip(dims[: self.trace_length], other[: self.trace_length]))
            ), (0, 1)
        if pattern == "batch_broadcast_permute":
            other = _DYNAMIC_DIMS[(self.seed // 13) % len(_DYNAMIC_DIMS)]
            return tuple(
                (((batch, length, fixed), (batch, 1, fixed)) for batch, length in zip(dims[: self.trace_length], other[: self.trace_length]))
            ), (0, 1)
        if pattern == "dynamic_reduction":
            other = _DYNAMIC_DIMS[(self.seed // 17) % len(_DYNAMIC_DIMS)]
            return tuple(
                (((batch, length, fixed), (batch, length, fixed)) for batch, length in zip(dims[: self.trace_length], other[: self.trace_length]))
            ), (0, 1)
        if pattern == "dynamic_slice_reduction":
            return tuple((((value, fixed), (value, fixed)) for value in dims[: self.trace_length])), (0,)
        if pattern == "dynamic_concat_reshape":
            return tuple((((value, fixed), (value, fixed)) for value in dims[: self.trace_length])), (0,)
        raise AssertionError(pattern)

    @staticmethod
    def _layouts(first_step: StepShapes, mode: str) -> tuple[str, ...]:
        layouts = ["contiguous"] * len(first_step)
        if mode in {"lhs_slice", "both_slice"}:
            layouts[0] = "slice" if first_step[0][-1] > 1 else "contiguous"
        if mode in {"rhs_slice", "both_slice"} and len(first_step) > 1:
            layouts[1] = "slice" if first_step[1][-1] > 1 else "contiguous"
        if mode == "transpose_pair":
            for index, shape in enumerate(first_step):
                layouts[index] = "transpose" if len(shape) >= 2 else "slice"
        return tuple(layouts)

    @staticmethod
    def _binary_operation(name: str, left: str, right: str) -> Operation:
        if name == "div":
            return Operation.create(name, (left, right), safe_denominator=True)
        return Operation.create(name, (left, right))


def _operation_family(pattern: str) -> str:
    return {
        "vary_dim0": "dynamic_elementwise",
        "vary_dim0_broadcast_reduction": "broadcast_reduction",
        "vary_dim1_broadcast_reduction": "broadcast_reduction",
        "two_dims_broadcast_transform": "broadcast_transform_reshape",
        "batch_broadcast_permute": "broadcast_permute_reduction",
        "dynamic_reduction": "dynamic_reduction",
        "dynamic_slice_reduction": "slice_reduction",
        "dynamic_concat_reshape": "concat_reshape",
    }[pattern]


def _broadcast_shape(left: Shape, right: Shape) -> Shape:
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


def _reduce_shape(shape: Shape, *, dim: int) -> Shape:
    normalized = dim if dim >= 0 else len(shape) + dim
    return tuple(value for index, value in enumerate(shape) if index != normalized)


def _transpose_shape(shape: Shape, dim0: int, dim1: int) -> Shape:
    result = list(shape)
    result[dim0], result[dim1] = result[dim1], result[dim0]
    return tuple(result)


def _permute_shape(shape: Shape, dims: tuple[int, ...]) -> Shape:
    return tuple(shape[index] for index in dims)


def _slice_shape(shape: Shape) -> Shape:
    return (*shape[:-1], max(1, (shape[-1] + 1) // 2))


def _reshape_flat(shape: Shape) -> Shape:
    total = 1
    for value in shape:
        total *= value
    return (total,)


def _cat_shape(left: Shape, right: Shape, *, dim: int) -> Shape:
    if len(left) != len(right):
        raise ValueError("concatenation requires equal ranks")
    result = list(left)
    for index, (left_dim, right_dim) in enumerate(zip(left, right)):
        if index != dim and left_dim != right_dim:
            raise ValueError("concatenation requires matching non-concat dimensions")
    result[dim] += right[dim]
    return tuple(result)
