import torch

def target_model(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    value.transpose(0, 1).index_fill_(-1, index, 0.5)
    return value

def run_experiment() -> torch.Tensor:
    value = torch.zeros((2, 3), dtype=torch.float64)
    index = torch.tensor((-1,), dtype=torch.long)
    eager_output = target_model(value.clone(), index)
    compiled = torch.compile(target_model, backend='inductor')
    compiled_input = eager_output.clone()
    return compiled(compiled_input, index)
print(run_experiment())
