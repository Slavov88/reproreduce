from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    requires_grad: bool = True
    layout: str = "contiguous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class Operation:
    name: str
    inputs: tuple[str, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(cls, name: str, inputs: tuple[str, ...], **kwargs: Any) -> "Operation":
        return cls(name=name, inputs=inputs, kwargs=tuple(sorted(kwargs.items())))

    def kwargs_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "inputs": list(self.inputs), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class Program:
    inputs: tuple[TensorSpec, ...]
    operations: tuple[Operation, ...]
    output: str
    differentiable: bool = True
    metadata: tuple[tuple[str, Any], ...] = ()
    observables: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [item.to_dict() for item in self.inputs],
            "operations": [item.to_dict() for item in self.operations],
            "output": self.output,
            "differentiable": self.differentiable,
            "metadata": dict(self.metadata),
            "observables": list(self.observables),
        }

    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    def to_source(self) -> str:
        lines = ["import torch", "", ""]
        lines.append("def generated_program(" + ", ".join(item.name for item in self.inputs) + "):")
        if not self.operations:
            lines.append(f"    return {self.output}")
        else:
            for index, operation in enumerate(self.operations):
                result_name = self._result_name(index)
                if operation.name == "slice_assign":
                    lines.append(f"    {self._mutation_statement(operation)}")
                    lines.append(f"    {result_name} = {self._operation_target(operation)}")
                else:
                    lines.append(f"    {result_name} = {self._expression(operation)}")
            return_names = (self.output, *self.observables)
            if len(return_names) == 1:
                lines.append(f"    return {return_names[0]}")
            else:
                lines.append(f"    return ({', '.join(return_names)})")
        lines.extend(["", "", "if __name__ == '__main__':"])
        for item in self.inputs:
            dtype = f"torch.{item.dtype}"
            base_shape = item.shape
            if item.layout == "transpose" and len(item.shape) >= 2:
                base_shape = (item.shape[1], item.shape[0], *item.shape[2:])
            elif item.layout == "slice" and item.shape:
                base_shape = (*item.shape[:-1], max(1, 2 * item.shape[-1] - 1))
            shape = ", ".join(str(value) for value in base_shape)
            if len(base_shape) == 1:
                shape += ","
            lines.append(
                f"    {item.name} = torch.randn(({shape}), dtype={dtype}, "
                f"requires_grad={item.requires_grad})"
            )
            if item.layout == "transpose" and len(item.shape) >= 2:
                lines.append(f"    {item.name} = {item.name}.transpose(0, 1)")
            elif item.layout == "slice" and item.shape:
                lines.append(f"    {item.name} = {item.name}[..., ::2]")
        lines.append("    output = generated_program(" + ", ".join(item.name for item in self.inputs) + ")")
        if self.differentiable:
            lines.append("    if torch.is_tensor(output) and output.is_floating_point():")
            lines.append("        output.sum().backward()")
        lines.append("    print(output)")
        return "\n".join(lines) + "\n"

    def _result_name(self, index: int) -> str:
        return f"v{index}"

    def _operation_target(self, operation: Operation) -> str:
        name = operation.inputs[0]
        return self._result_name(int(name[1:])) if name.startswith("v") else name

    def _mutation_statement(self, operation: Operation) -> str:
        target = self._operation_target(operation)
        kwargs = operation.kwargs_dict()
        start = _slice_part(kwargs.get("start"))
        stop = _slice_part(kwargs.get("stop"))
        step = _slice_part(kwargs.get("step", 1))
        return f"{target}[..., {start}:{stop}:{step}] = {kwargs['value']!r}"

    def _expression(self, operation: Operation) -> str:
        args = [self._result_name(int(name[1:])) if name.startswith("v") else name for name in operation.inputs]
        kwargs = operation.kwargs_dict()
        name = operation.name
        if name in {"add", "sub", "mul", "div"}:
            right = args[1]
            if name == "div" and kwargs.get("safe_denominator"):
                right = f"(torch.abs({right}) + 0.5)"
            return f"{args[0]} { {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[name] } {right}"
        if name in {"maximum", "minimum"}:
            return f"torch.{name}({args[0]}, {args[1]})"
        if name == "where":
            return f"torch.where({args[0]} > 0, {args[0]}, {args[1]})"
        if name in {"sin", "cos", "exp", "log", "relu"}:
            return f"torch.{name}({args[0]})"
        if name in {"sum", "mean", "amax"}:
            if "dim" not in kwargs:
                return f"{args[0]}.{name}()"
            dim = kwargs["dim"]
            keepdim = kwargs.get("keepdim", False)
            return f"{args[0]}.{name}(dim={dim}, keepdim={keepdim!r})"
        if name == "reshape":
            shape = ", ".join(str(value) for value in kwargs["shape"])
            return f"{args[0]}.reshape({shape})"
        if name == "transpose":
            return f"{args[0]}.transpose({kwargs['dim0']}, {kwargs['dim1']})"
        if name == "permute":
            dims = ", ".join(str(value) for value in kwargs["dims"])
            return f"{args[0]}.permute({dims})"
        if name == "slice":
            start = _slice_part(kwargs.get("start"))
            stop = _slice_part(kwargs.get("stop"))
            step = _slice_part(kwargs.get("step", 2))
            return f"{args[0]}[..., {start}:{stop}:{step}]"
        if name == "narrow":
            return f"{args[0]}.narrow({kwargs['dim']}, {kwargs['start']}, {kwargs['length']})"
        if name == "select":
            return f"{args[0]}.select({kwargs['dim']}, {kwargs['index']})"
        if name == "unsqueeze":
            return f"{args[0]}.unsqueeze({kwargs['dim']})"
        if name == "squeeze":
            return f"{args[0]}.squeeze({kwargs['dim']})"
        if name == "flatten":
            return f"{args[0]}.flatten({kwargs.get('start_dim', 0)}, {kwargs.get('end_dim', -1)})"
        if name == "diagonal":
            return f"{args[0]}.diagonal({kwargs.get('offset', 0)}, {kwargs['dim1']}, {kwargs['dim2']})"
        if name in {"add_", "sub_", "mul_", "fill_"}:
            return f"{args[0]}.{name}({kwargs['value']!r})"
        if name == "zero_":
            return f"{args[0]}.zero_()"
        if name == "copy_":
            return f"{args[0]}.copy_({args[1]})"
        if name == "index_fill_":
            indices = tuple(kwargs["indices"])
            return (
                f"{args[0]}.index_fill_({kwargs['dim']}, "
                f"torch.tensor({indices!r}, dtype=torch.long, device={args[0]}.device), "
                f"{kwargs['value']!r})"
            )
        if name == "cat":
            return f"torch.cat([{', '.join(args)}], dim={kwargs.get('dim', 0)})"
        raise ValueError(f"unsupported operation: {name}")


def _slice_part(value: Any) -> str:
    return "" if value is None else str(value)
