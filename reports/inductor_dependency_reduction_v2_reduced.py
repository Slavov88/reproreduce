from dataclasses import dataclass
import torch

@dataclass
class ExperimentConfig:
    rows: int = 2
    columns: int = 3
    index_value: tuple = ()

def build_input(config: ExperimentConfig) -> torch.Tensor:
    return torch.zeros((config.rows, config.columns), dtype=torch.float64)

def target_model(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    value.transpose(0, 1).index_fill_(1, index, 0.5)
    return value

def run_experiment(config: ExperimentConfig) -> torch.Tensor:
    value = build_input(config)
    index = torch.tensor(config.index_value, dtype=torch.long)
    eager_output = target_model(value.clone(), index)
    compiled = torch.compile(target_model, backend='inductor')
    compiled_input = eager_output.clone()
    return compiled(compiled_input, index)

def main() -> None:
    print(run_experiment(ExperimentConfig()))
main()
