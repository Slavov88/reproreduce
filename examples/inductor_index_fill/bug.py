import math
import random

import torch


def unused_helper(value):
    return math.sqrt(9.0) + value * 0.0


def model(x, index):
    unused = x + 1.0
    view = x.transpose(0, 1)
    view.index_fill_(-1, index, 0.5)
    check = view.contiguous()
    return x


x = torch.randn((2, 3), dtype=torch.float32)
index = torch.tensor((0, 1), dtype=torch.long)
noise = unused_helper(random.random())
eager_input = x.clone()
eager_output = model(eager_input, index)
compiled = torch.compile(model, backend="inductor")
compiled_input = eager_output.clone()
print(compiled(compiled_input, index))
