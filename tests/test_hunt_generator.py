import ast
import unittest

import torch

from reproreduce.hunt import TensorProgramGenerator


class HuntGeneratorTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        first = TensorProgramGenerator(42).generate()
        second = TensorProgramGenerator(42).generate()
        self.assertEqual(first.serialize(), second.serialize())

    def test_generated_programs_parse_and_execute_eagerly(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                program = TensorProgramGenerator(seed, max_operations=6).generate()
                source = program.to_source()
                ast.parse(source)
                namespace = {}
                exec(compile(source, "generated.py", "exec"), namespace)
                inputs = [
                    torch.randn(spec.shape, dtype=getattr(torch, spec.dtype), requires_grad=spec.requires_grad)
                    for spec in program.inputs
                ]
                output = namespace["generated_program"](*inputs)
                self.assertTrue(torch.is_tensor(output))

    def test_generated_programs_are_short(self):
        for seed in range(20):
            program = TensorProgramGenerator(seed).generate()
            self.assertLessEqual(len(program.operations), 6)
            self.assertGreaterEqual(len(program.operations), 2)


if __name__ == "__main__":
    unittest.main()
