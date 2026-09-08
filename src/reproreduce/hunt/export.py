from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from ..core.result import ReductionResult
from .config import TensorConfig
from .program import Program


def export_finding(
    result: ReductionResult,
    output: str | Path,
    *,
    finding: dict[str, Any] | None = None,
    program: Program | None = None,
    configs: tuple[TensorConfig, ...] | None = None,
    config_trace: Sequence[tuple[TensorConfig, ...]] | None = None,
    mode: str = "forward",
    backend: str = "inductor",
    input_seed: int = 0,
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    repro_source = result.reduced_source
    if program is not None:
        selected_configs = configs or tuple(
            TensorConfig(
                shape=spec.shape,
                dtype=spec.dtype,
                requires_grad=spec.requires_grad,
                layout=spec.layout,
            )
            for spec in program.inputs
        )
        if config_trace is not None:
            repro_source = _with_dynamic_harness(
                repro_source,
                program,
                config_trace,
                mode=mode,
                backend=backend,
                input_seed=input_seed,
            )
        elif program.observables:
            repro_source = _with_alias_harness(
                repro_source,
                program,
                selected_configs,
                mode=mode,
                backend=backend,
                input_seed=input_seed,
            )
        else:
            repro_source = _with_harness(
                repro_source,
                program,
                selected_configs,
                mode=mode,
                backend=backend,
                input_seed=input_seed,
            )
    (destination / "repro.py").write_text(repro_source, encoding="utf-8")

    metrics = result.metrics
    finding_data = finding or {}
    report = "\n".join(
        [
            "# ReproReduce finding",
            "",
            "## Reproduction",
            "",
            "Run the reduced program with `python repro.py`.",
            "",
            f"- Classification: {finding_data.get('classification', 'unrecorded')}",
            f"- Backend: {finding_data.get('backend', backend)}",
            f"- Mode: {finding_data.get('mode', mode)}",
            f"- Original LOC: {result.original_loc}",
            f"- Reduced LOC: {result.reduced_loc}",
            "",
            "## Reduction metrics",
            "",
            f"- Candidate runs: {metrics.get('candidate_runs', 0)}",
            f"- Cache hits: {metrics.get('cache_hits', 0)}",
            f"- Wall time: {metrics.get('total_reduction_wall_time', 0.0):.3f} seconds",
            "",
        ]
    )
    (destination / "report.md").write_text(report, encoding="utf-8")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        environment["pytorch"] = torch.__version__
        environment["cuda"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            environment["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")

    finding_data = {
        "original_loc": result.original_loc,
        "reduced_loc": result.reduced_loc,
        "metrics": result.metrics,
        "finding": finding_data,
    }
    (destination / "finding.json").write_text(json.dumps(finding_data, indent=2, default=str) + "\n", encoding="utf-8")
    return destination


def _with_harness(
    source: str,
    program: Program,
    configs: tuple[TensorConfig, ...],
    *,
    mode: str,
    backend: str,
    input_seed: int,
) -> str:
    lines = [source.rstrip(), "", "", "if __name__ == '__main__':"]
    lines.append("    import torch")
    lines.append("")
    for prefix, seed_offset in (("eager", 0), ("compiled", 0)):
        lines.append(f"    def make_{prefix}_inputs():")
        for index, (spec, config) in enumerate(zip(program.inputs, configs)):
            lines.extend(_input_lines(spec.name, config, input_seed + seed_offset + index))
        lines.append("        return " + ", ".join(spec.name for spec in program.inputs))
        lines.append("")
    lines.append("    eager_inputs = make_eager_inputs()")
    lines.append("    compiled_inputs = make_compiled_inputs()")
    lines.append("    eager_result = generated_program(*eager_inputs)")
    lines.append(f"    compiled_program = torch.compile(generated_program, backend={backend!r})")
    lines.append("    compiled_result = compiled_program(*compiled_inputs)")
    lines.append("    print('eager result:', eager_result)")
    lines.append("    print('compiled result:', compiled_result)")
    lines.append("    if torch.is_tensor(eager_result) and torch.is_tensor(compiled_result):")
    lines.append("        print('output max abs difference:', (eager_result.detach().to(torch.float64) - compiled_result.detach().to(torch.float64)).abs().max().item())")
    if mode == "gradient":
        lines.append("    eager_grads = torch.autograd.grad(eager_result.sum(), eager_inputs, allow_unused=True)")
        lines.append("    compiled_grads = torch.autograd.grad(compiled_result.sum(), compiled_inputs, allow_unused=True)")
        lines.append("    print('eager gradients:', eager_grads)")
        lines.append("    print('compiled gradients:', compiled_grads)")
        lines.append("    for eager_grad, compiled_grad in zip(eager_grads, compiled_grads):")
        lines.append("        if eager_grad is not None and compiled_grad is not None:")
        lines.append("            print('gradient max abs difference:', (eager_grad.to(torch.float64) - compiled_grad.to(torch.float64)).abs().max().item())")
    return "\n".join(lines) + "\n"


def _with_alias_harness(
    source: str,
    program: Program,
    configs: tuple[TensorConfig, ...],
    *,
    mode: str,
    backend: str,
    input_seed: int,
) -> str:
    lines = [source.rstrip(), "", "", "if __name__ == '__main__':", "    import torch", ""]
    for prefix in ("eager", "compiled"):
        lines.append(f"    def make_{prefix}_inputs():")
        for index, (spec, config) in enumerate(zip(program.inputs, configs)):
            lines.extend(_input_lines(spec.name, config, input_seed + index))
        lines.append("        return " + ", ".join(spec.name for spec in program.inputs))
        lines.append("")
    lines.append("    eager_inputs = make_eager_inputs()")
    lines.append("    compiled_inputs = make_compiled_inputs()")
    lines.append(f"    compiled_program = torch.compile(generated_program, backend={backend!r})")
    lines.append("    eager_result = generated_program(*eager_inputs)")
    lines.append("    compiled_result = compiled_program(*compiled_inputs)")
    lines.append("    eager_state = tuple(value.detach().clone() for value in eager_inputs)")
    lines.append("    compiled_state = tuple(value.detach().clone() for value in compiled_inputs)")
    lines.append("    print('eager return:', eager_result)")
    lines.append("    print('compiled return:', compiled_result)")
    lines.append("    print('eager input state:', eager_state)")
    lines.append("    print('compiled input state:', compiled_state)")
    lines.append("    if isinstance(eager_result, tuple) and isinstance(compiled_result, tuple):")
    lines.append("        for index, (eager_value, compiled_value) in enumerate(zip(eager_result, compiled_result)):")
    lines.append("            if torch.is_tensor(eager_value) and torch.is_tensor(compiled_value):")
    lines.append("                print('return item', index, 'max abs difference:', (eager_value.detach().to(torch.float64) - compiled_value.detach().to(torch.float64)).abs().max().item())")
    lines.append("    for index, (eager_value, compiled_value) in enumerate(zip(eager_state, compiled_state)):")
    lines.append("        print('input state', index, 'max abs difference:', (eager_value.to(torch.float64) - compiled_value.to(torch.float64)).abs().max().item())")
    return "\n".join(lines) + "\n"


def _with_dynamic_harness(
    source: str,
    program: Program,
    config_trace: Sequence[tuple[TensorConfig, ...]],
    *,
    mode: str,
    backend: str,
    input_seed: int,
) -> str:
    lines = [source.rstrip(), "", "", "if __name__ == '__main__':", "    import torch", ""]
    lines.append(f"    compiled_program = torch.compile(generated_program, backend={backend!r}, dynamic=True)")
    for index, configs in enumerate(config_trace):
        seed = input_seed + index * 1009
        lines.append(f"    # shape index {index}: {[config.shape for config in configs]!r}")
        eager_names = []
        compiled_names = []
        for prefix in ("eager", "compiled"):
            names = []
            for position, (spec, config) in enumerate(zip(program.inputs, configs)):
                name = f"{prefix}_{index}_{position}"
                names.append(name)
                lines.extend(line[4:] for line in _input_lines(name, config, seed + position))
            if prefix == "eager":
                eager_names = names
            else:
                compiled_names = names
        lines.append(f"    eager_result_{index} = generated_program({', '.join(eager_names)})")
        lines.append(f"    compiled_result_{index} = compiled_program({', '.join(compiled_names)})")
        lines.append(f"    print('shape {index} eager:', eager_result_{index})")
        lines.append(f"    print('shape {index} compiled:', compiled_result_{index})")
        lines.append(
            f"    if torch.is_tensor(eager_result_{index}) and torch.is_tensor(compiled_result_{index}):"
        )
        lines.append(
            f"        print('shape {index} max abs difference:', "
            f"(eager_result_{index}.detach().to(torch.float64) - compiled_result_{index}.detach().to(torch.float64)).abs().max().item())"
        )
        if mode == "gradient":
            lines.append(
                f"    print('shape {index} eager gradients:', "
                f"torch.autograd.grad(eager_result_{index}.sum(), ({', '.join(eager_names)}), allow_unused=True))"
            )
            lines.append(
                f"    print('shape {index} compiled gradients:', "
                f"torch.autograd.grad(compiled_result_{index}.sum(), ({', '.join(compiled_names)}), allow_unused=True))"
            )
    return "\n".join(lines) + "\n"


def _input_lines(name: str, config: TensorConfig, seed: int) -> list[str]:
    base_shape = config.shape
    if config.layout == "transpose":
        base_shape = (config.shape[1], config.shape[0], *config.shape[2:])
    elif config.layout == "slice":
        base_shape = (*config.shape[:-1], max(1, 2 * config.shape[-1] - 1))
    shape = ", ".join(str(value) for value in base_shape)
    if len(base_shape) == 1:
        shape += ","
    lines = [
        f"        torch.manual_seed({seed})",
        f"        {name} = torch.randn(({shape}), dtype=torch.{config.dtype}, requires_grad={config.requires_grad})",
    ]
    if config.layout == "transpose":
        lines.append(f"        {name} = {name}.transpose(0, 1)")
    elif config.layout == "slice":
        lines.append(f"        {name} = {name}[..., ::2]")
    return lines
