import importlib.util
import unittest

from reproreduce import GradientDifferenceOracle


PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch is not installed")
class GradientDifferenceOracleTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.x = torch.tensor([2.0, 3.0], requires_grad=True)

    def _evaluate(self, function, compiled, *inputs):
        return GradientDifferenceOracle(compiler=lambda _: compiled).evaluate_function(function, *inputs)

    def test_matching_gradients(self):
        def function(x):
            return x * x

        result = self._evaluate(function, function, self.x)
        self.assertFalse(result.interesting)
        self.assertEqual(result.metadata["scalarization"], "tensor.sum")

    def test_small_gradient_difference_inside_tolerance(self):
        def function(x):
            return x * x

        def compiled(x):
            return x * x + 1e-7 * x

        result = self._evaluate(function, compiled, self.x)
        self.assertFalse(result.interesting)

    def test_meaningful_gradient_mismatch(self):
        def function(x):
            return x * x

        def compiled(x):
            return x * 0

        result = self._evaluate(function, compiled, self.x)
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["input"], "input[0]")
        self.assertEqual(result.metadata["kind"], "gradient_tensor_mismatch")

    def test_none_vs_tensor_gradient(self):
        def function(x):
            return x * x

        def compiled(x):
            return (x * x).detach()

        result = self._evaluate(function, compiled, self.x)
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["kind"], "none_mismatch")

    def test_shape_and_dtype_mismatch(self):
        oracle = GradientDifferenceOracle()
        shape = oracle.compare((self.torch.ones(2),), (self.torch.ones(1),))
        self.assertTrue(shape.interesting)
        self.assertEqual(shape.metadata["kind"], "gradient_shape_mismatch")
        dtype = oracle.compare(
            (self.torch.ones(2, dtype=self.torch.float32),),
            (self.torch.ones(2, dtype=self.torch.float64),),
        )
        self.assertTrue(dtype.interesting)
        self.assertEqual(dtype.metadata["kind"], "gradient_dtype_mismatch")

    def test_nan_and_inf_mismatch(self):
        oracle = GradientDifferenceOracle()
        nan = oracle.compare(
            (self.torch.tensor([float("nan")]),),
            (self.torch.tensor([0.0]),),
        )
        self.assertTrue(nan.interesting)
        self.assertTrue(nan.metadata["nan_mismatch"])
        inf = oracle.compare(
            (self.torch.tensor([float("inf")]),),
            (self.torch.tensor([float("-inf")]),),
        )
        self.assertTrue(inf.interesting)
        self.assertTrue(inf.metadata["inf_mismatch"])

    def test_candidate_only_exception(self):
        def function(x):
            return x * x

        def compiled(x):
            raise RuntimeError("compiled failure")

        result = self._evaluate(function, compiled, self.x)
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["kind"], "candidate_only_exception")

    def test_same_exception_is_not_a_gradient_discrepancy(self):
        def failing(x):
            raise RuntimeError("same")

        result = self._evaluate(failing, failing, self.x)
        self.assertFalse(result.interesting)

    def test_multiple_inputs_are_reported_by_position(self):
        def function(x, y):
            return x * y

        def compiled(x, y):
            return x * y + 0.5 * x

        y = self.torch.tensor([4.0, 5.0], requires_grad=True)
        result = self._evaluate(function, compiled, self.x, y)
        self.assertTrue(result.interesting)
        self.assertIn(result.metadata["input"], {"input[0]", "input[1]"})

    def test_real_torch_compile_eager_backend(self):
        def function(x):
            return x * x

        oracle = GradientDifferenceOracle(
            compiler=lambda fn: self.torch.compile(fn, backend="eager")
        )
        result = oracle.evaluate_function(function, self.x)
        self.assertFalse(result.interesting)


if __name__ == "__main__":
    unittest.main()
