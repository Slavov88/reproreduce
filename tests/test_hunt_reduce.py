import unittest

import torch

from reproreduce.hunt import (
    Operation,
    Program,
    ProgramExecutor,
    TensorConfig,
    TensorSpec,
    reduce_confirmed,
    reduce_execution,
)
from reproreduce.oracle import CompileDifferenceOracle


class HuntReductionTests(unittest.TestCase):
    def test_confirmed_finding_is_handed_to_reproreduce(self):
        program = Program(
            inputs=(TensorSpec("x", (2, 3)),),
            operations=(Operation.create("sin", ("x",)),),
            output="v0",
        )

        def source_evaluator(source):
            if "torch.sin" in source:
                return torch.zeros(1), torch.ones(1)
            return torch.zeros(1), torch.zeros(1)

        result = reduce_confirmed(
            program,
            CompileDifferenceOracle(source_evaluator=source_evaluator),
            timeout=5,
        )
        self.assertIn("torch.sin", result.reduced_source)
        self.assertTrue(result.reduced_run.returncode != 0)
        self.assertGreaterEqual(result.metrics["candidate_runs"], 1)

    def test_executor_builds_backend_aware_reduction_adapter(self):
        program = Program(
            inputs=(TensorSpec("x", (2, 3)),),
            operations=(Operation.create("sin", ("x",)),),
            output="v0",
        )
        configs = (TensorConfig((2, 3), requires_grad=False),)
        executor = ProgramExecutor(backend="synthetic", compiler=lambda fn: lambda x: fn(x) + 0.25)
        result = reduce_execution(program, configs, executor, mode="forward", timeout=5)
        self.assertIn("torch.sin", result.reduced_source)
        self.assertNotEqual(result.reduced_run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
