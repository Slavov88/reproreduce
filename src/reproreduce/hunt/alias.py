from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .config import TensorConfig
from .program import Operation, Program


@dataclass(frozen=True)
class AliasMutationCase:
    program: Program
    configs: tuple[TensorConfig, ...]
    return_names: tuple[str, ...]
    alias_pairs: tuple[tuple[str, str, bool], ...]


class AliasMutationGenerator:
    """Generate small, structured storage-aliasing and mutation programs."""

    _PATTERNS = (
        "mutate_view_observe_base",
        "mutate_base_observe_view",
        "sibling_views",
        "overlapping_views",
        "chained_views",
        "mutation_reduction",
        "mutation_view_transform",
        "sequential_mutations",
    )
    _DTYPES = ("float32", "float64", "bfloat16")
    _LAYOUTS = ("contiguous", "transpose", "slice")
    _MUTATIONS = (
        "add_",
        "sub_",
        "mul_",
        "fill_",
        "zero_",
        "copy_",
        "index_fill_",
        "slice_assign",
    )
    _SHAPES = ((6, 8), (5, 9), (7, 10), (4, 12))

    def __init__(self, seed: int, *, mode: str = "forward"):
        if mode != "forward":
            raise ValueError("alias_mutation v1 supports forward mode only")
        self.seed = seed
        self.random = random.Random(seed)
        self.mode = mode

    def generate(self) -> AliasMutationCase:
        pattern = self._PATTERNS[self.seed % len(self._PATTERNS)]
        dtype = self._DTYPES[(self.seed // 24) % len(self._DTYPES)]
        layout = self._LAYOUTS[(self.seed // 64) % len(self._LAYOUTS)]
        base_shape = self._SHAPES[(self.seed // 5) % len(self._SHAPES)]
        # Keep the structured axes orthogonal: every pattern sees every mutation,
        # dtype, and layout over a short deterministic grid.
        mutation = self._MUTATIONS[(self.seed // len(self._PATTERNS)) % len(self._MUTATIONS)]

        operations: list[Operation] = []
        view_shapes: dict[str, tuple[int, ...]] = {"x": base_shape}
        view_chain: list[dict[str, Any]] = []

        def append(operation: Operation, shape: tuple[int, ...]) -> str:
            name = f"v{len(operations)}"
            operations.append(operation)
            view_shapes[name] = shape
            return name

        def view_slice(source: str, *, start: int | None, stop: int | None, step: int) -> str:
            shape = view_shapes[source]
            length = _slice_length(shape[-1], start, stop, step)
            result = append(
                Operation.create("slice", (source,), start=start, stop=stop, step=step),
                (*shape[:-1], length),
            )
            view_chain.append({"op": "slice", "start": start, "stop": stop, "step": step})
            return result

        output = ""
        post_mutation = None
        if pattern == "mutate_view_observe_base":
            view = append(Operation.create("transpose", ("x",), dim0=0, dim1=1), (base_shape[1], base_shape[0]))
            view_chain.append({"op": "transpose", "dims": (0, 1)})
            target = view
            output = "x"
            observables = (view,)
            alias_pairs = (("x", view, True),)
        elif pattern == "mutate_base_observe_view":
            view = view_slice("x", start=1, stop=-1, step=1)
            target = "x"
            output = view
            observables = ("x",)
            alias_pairs = (("x", view, True),)
        elif pattern == "sibling_views":
            even = view_slice("x", start=0, stop=None, step=2)
            odd = view_slice("x", start=1, stop=None, step=2)
            target = even
            output = "x"
            observables = (even, odd)
            alias_pairs = (("x", even, True), ("x", odd, True), (even, odd, True))
        elif pattern == "overlapping_views":
            left = view_slice("x", start=0, stop=-1, step=1)
            right = view_slice("x", start=1, stop=None, step=1)
            target = left
            output = "x"
            observables = (left, right)
            alias_pairs = (("x", left, True), ("x", right, True), (left, right, True))
        elif pattern == "chained_views":
            transposed = append(Operation.create("transpose", ("x",), dim0=0, dim1=1), (base_shape[1], base_shape[0]))
            view_chain.append({"op": "transpose", "dims": (0, 1)})
            chained = view_slice(transposed, start=1, stop=-1, step=1)
            target = chained
            output = "x"
            observables = (transposed, chained)
            alias_pairs = (("x", transposed, True), ("x", chained, True))
        elif pattern == "mutation_reduction":
            view = view_slice("x", start=0, stop=None, step=2)
            target = view
            post_mutation = "reduction"
            observables = ("x", view)
            alias_pairs = (("x", view, True),)
        elif pattern == "mutation_view_transform":
            view = view_slice("x", start=1, stop=None, step=2)
            target = view
            post_mutation = "transform"
            observables = ("x", view)
            alias_pairs = (("x", view, True),)
        elif pattern == "sequential_mutations":
            view = append(Operation.create("transpose", ("x",), dim0=0, dim1=1), (base_shape[1], base_shape[0]))
            view_chain.append({"op": "transpose", "dims": (0, 1)})
            target = view
            output = "x"
            observables = (view,)
            alias_pairs = (("x", view, True),)
        else:
            raise AssertionError(pattern)

        mutation_operation = self._mutation_operation(mutation, target, view_shapes[target])
        append(mutation_operation, view_shapes[target])
        if pattern == "sequential_mutations":
            second_mutation = self._mutation_operation("mul_", target, view_shapes[target])
            append(second_mutation, view_shapes[target])
        if post_mutation == "reduction":
            output = append(Operation.create("sum", ("x",)), ())
        elif post_mutation == "transform":
            output = append(
                Operation.create("transpose", (target,), dim0=0, dim1=1),
                (view_shapes[target][1], view_shapes[target][0]),
            )
            view_chain.append({"op": "transpose", "dims": (0, 1)})

        return_names = (output, *observables)
        target_shape = view_shapes[target]
        configs = [TensorConfig(base_shape, dtype=dtype, requires_grad=False, layout=layout)]
        if mutation == "copy_":
            configs.append(TensorConfig(target_shape, dtype=dtype, requires_grad=False, layout="contiguous"))
        metadata = (
            ("family", "alias_mutation"),
            ("pattern", pattern),
            ("alias_pattern", pattern),
            ("state_observation", True),
            ("dtype", dtype),
            ("mode", self.mode),
            ("base_shape", base_shape),
            ("layout", layout),
            ("view_chain", tuple(view_chain)),
            ("mutation", {"target": target, "op": mutation}),
            ("observable", return_names),
            ("alias_pairs", alias_pairs),
            ("overlapping", pattern == "overlapping_views"),
            ("input_layouts", tuple(config.layout for config in configs)),
        )
        program = Program(
            inputs=tuple(config.to_spec(name) for name, config in zip(("x", "y"), configs)),
            operations=tuple(operations),
            output=output,
            differentiable=False,
            metadata=metadata,
            observables=tuple(observables),
        )
        return AliasMutationCase(
            program=program,
            configs=tuple(configs),
            return_names=return_names,
            alias_pairs=alias_pairs,
        )

    @staticmethod
    def _mutation_operation(name: str, target: str, target_shape: tuple[int, ...]) -> Operation:
        if name in {"add_", "sub_", "mul_", "fill_"}:
            return Operation.create(name, (target,), value=0.5)
        if name == "zero_":
            return Operation.create(name, (target,))
        if name == "copy_":
            return Operation.create(name, (target, "y"))
        if name == "index_fill_":
            last = max(1, target_shape[-1])
            indices = (0, min(2, last - 1))
            return Operation.create(name, (target,), dim=-1, indices=indices, value=0.5)
        if name == "slice_assign":
            return Operation.create(name, (target,), start=0, stop=None, step=2, value=0.5)
        raise AssertionError(name)


def _slice_length(length: int, start: int | None, stop: int | None, step: int) -> int:
    return len(range(length)[slice(start, stop, step)])
