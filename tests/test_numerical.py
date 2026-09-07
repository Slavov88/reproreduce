import importlib.util
import unittest

from reproreduce.oracle import CompileDifferenceOracle, ExecutionOutcome


PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch is not installed")
class CompileDifferenceOracleTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.oracle = CompileDifferenceOracle(atol=1e-5, rtol=1e-5)

    def test_identical_results_are_not_interesting(self):
        value = self.torch.ones(3)
        result = self.oracle.compare(value, value.clone())
        self.assertFalse(result.interesting)

    def test_small_difference_inside_tolerance_is_not_interesting(self):
        result = self.oracle.compare(self.torch.ones(3), self.torch.ones(3) + 1e-7)
        self.assertFalse(result.interesting)

    def test_meaningful_mismatch_is_interesting(self):
        result = self.oracle.compare(self.torch.ones(3), self.torch.ones(3) + 0.1)
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["mismatching_elements"], 3)
        self.assertGreater(result.metadata["max_abs_error"], 0.09)

    def test_shape_and_dtype_mismatches_are_interesting(self):
        shape = self.oracle.compare(self.torch.ones(2), self.torch.ones(3))
        self.assertEqual(shape.metadata["kind"], "shape_mismatch")
        dtype = self.oracle.compare(
            self.torch.ones(2, dtype=self.torch.float32),
            self.torch.ones(2, dtype=self.torch.float64),
        )
        self.assertEqual(dtype.metadata["kind"], "dtype_mismatch")

    def test_nan_and_inf_mismatches_are_interesting(self):
        nan_result = self.oracle.compare(
            self.torch.tensor([float("nan"), 1.0]),
            self.torch.tensor([0.0, 1.0]),
        )
        self.assertTrue(nan_result.interesting)
        self.assertTrue(nan_result.metadata["nan_mismatch"])
        inf_result = self.oracle.compare(
            self.torch.tensor([float("inf")]),
            self.torch.tensor([float("-inf")]),
        )
        self.assertTrue(inf_result.interesting)
        self.assertTrue(inf_result.metadata["inf_mismatch"])

    def test_candidate_only_exception_is_classified(self):
        def compiled(_):
            raise RuntimeError("compiled failure")

        result = CompileDifferenceOracle(compiler=lambda _: compiled).evaluate_function(
            lambda _: self.torch.ones(1), self.torch.ones(1)
        )
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["kind"], "candidate_only_exception")

    def test_reference_only_exception_is_classified(self):
        def reference(_):
            raise RuntimeError("eager failure")

        result = CompileDifferenceOracle(compiler=lambda _: lambda _: self.torch.ones(1)).evaluate_function(
            reference, self.torch.ones(1)
        )
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["kind"], "reference_only_exception")

    def test_matching_exceptions_are_not_interesting(self):
        def failing(_):
            raise RuntimeError("same")

        result = CompileDifferenceOracle(compiler=lambda _: failing).evaluate_function(
            failing, self.torch.ones(1)
        )
        self.assertFalse(result.interesting)

    def test_nested_tensor_outputs_report_paths(self):
        reference = {
            "logits": self.torch.ones(2),
            "aux": (self.torch.zeros(1), self.torch.ones(1)),
        }
        candidate = {
            "logits": self.torch.ones(2),
            "aux": (self.torch.zeros(1), self.torch.zeros(1)),
        }
        result = self.oracle.compare(reference, candidate)
        self.assertTrue(result.interesting)
        self.assertEqual(result.metadata["path"], "output['aux'][1]")

    def test_execution_outcomes_can_be_compared_directly(self):
        result = self.oracle.compare(
            ExecutionOutcome(value=self.torch.ones(1)),
            ExecutionOutcome(value=self.torch.zeros(1)),
        )
        self.assertTrue(result.interesting)


if __name__ == "__main__":
    unittest.main()
