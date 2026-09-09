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

    def test_compile_failure_is_separate_from_runtime_failure(self):
        def compiler_failure(_):
            raise RuntimeError("unsupported backend")

        compile_result = ProgramExecutor(compiler=compiler_failure).run(self._program(), mode="forward")
        self.assertEqual(compile_result.classification, OutcomeClass.COMPILE_FAILURE)

        def runtime_failure(_):
            def compiled(_x):
                raise RuntimeError("compiled execution failed")
            return compiled

        runtime_result = ProgramExecutor(compiler=runtime_failure).run(self._program(), mode="forward")
        self.assertEqual(runtime_result.classification, OutcomeClass.COMPILED_RUNTIME_FAILURE)


if __name__ == "__main__":
    unittest.main()
