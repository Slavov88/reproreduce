import unittest

import torch

from reproreduce.hunt import Operation, OutcomeClass, Program, ProgramExecutor, TensorSpec


class HuntExecutionTests(unittest.TestCase):
    def _program(self):
        return Program(
            inputs=(TensorSpec("x", (2, 3)),),
            operations=(Operation.create("mul", ("x", "x")), Operation.create("sum", ("v0",))),
            output="v1",
        )

    def test_identical_forward_is_pass(self):
        executor = ProgramExecutor(backend="eager", compiler=lambda fn: fn)
        result = executor.run(self._program(), mode="forward")
        self.assertEqual(result.classification, OutcomeClass.PASS)

    def test_forward_mismatch_is_classified(self):
        def compiler(fn):
            return lambda x: fn(x) + 0.25

        result = ProgramExecutor(compiler=compiler).run(self._program(), mode="forward")
        self.assertEqual(result.classification, OutcomeClass.FORWARD_MISMATCH)

    def test_gradient_mismatch_is_classified(self):
        def compiler(fn):
            return lambda x: fn(x) * 0

        result = ProgramExecutor(compiler=compiler).run(self._program(), mode="gradient")
        self.assertEqual(result.classification, OutcomeClass.GRADIENT_MISMATCH)

    def test_compiled_exception_is_not_called_a_bug(self):
        def compiler(_):
            raise RuntimeError("unsupported backend")

        result = ProgramExecutor(compiler=compiler).run(self._program(), mode="forward")
        self.assertEqual(result.classification, OutcomeClass.COMPILED_ERROR)


if __name__ == "__main__":
    unittest.main()
