import argparse
import json
import torch


def target(x):
    y = x.clone()
    z = y.clone()
    z.retain_grad()
    return z, y


def run_case(backend):
    x = torch.zeros([], requires_grad=True)
    function = target if backend == "eager" else torch.compile(target, backend=backend)
    z, y = function(x)
    z.clone().backward()
    return z.grad, y.grad


def serialise(values):
    return [None if value is None else float(value.detach().cpu()) for value in values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    eager = run_case("eager")
    compiled = run_case("aot_eager")
    payload = {"eager": serialise(eager), "compiled": serialise(compiled)}
    if args.json:
        print(json.dumps(payload))
