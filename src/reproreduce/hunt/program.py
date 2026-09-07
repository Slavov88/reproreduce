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

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [item.to_dict() for item in self.inputs],
            "operations": [item.to_dict() for item in self.operations],
            "output": self.output,
            "differentiable": self.differentiable,
        }

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
                lines.append(f"    {self._result_name(index)} = {self._expression(operation)}")
            lines.append(f"    return {self.output}")
        lines.extend(["", "", "if __name__ == '__main__':"])
        for item in self.inputs:
            dtype = f"torch.{item.dtype}"
            shape = ", ".join(str(value) for value in item.shape)
            if len(item.shape) == 1:
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

    def _expression(self, operation: Operation) -> str:
        args = [self._result_name(int(name[1:])) if name.startswith("v") else name for name in operation.inputs]
        kwargs = operation.kwargs_dict()
        name = operation.name
        if name in {"add", "sub", "mul", "div"}:
            return f"{args[0]} { {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[name] } {args[1]}"
        if name in {"sin", "cos", "exp", "log", "relu"}:
            return f"torch.{name}({args[0]})"
        if name in {"sum", "mean"}:
            return f"{args[0]}.{name}()"
        if name == "reshape":
            shape = ", ".join(str(value) for value in kwargs["shape"])
            return f"{args[0]}.reshape({shape})"
        if name == "transpose":
            return f"{args[0]}.transpose({kwargs['dim0']}, {kwargs['dim1']})"
        if name == "permute":
            dims = ", ".join(str(value) for value in kwargs["dims"])
            return f"{args[0]}.permute({dims})"
        if name == "slice":
            return f"{args[0]}[..., ::{kwargs.get('step', 2)}]"
        if name == "cat":
            return f"torch.cat([{', '.join(args)}], dim={kwargs.get('dim', 0)})"
        raise ValueError(f"unsupported operation: {name}")
