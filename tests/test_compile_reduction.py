import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

from reproreduce import CompileDifferenceOracle, reduce


PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch is not installed")
class CompileReductionIntegrationTests(unittest.TestCase):
    def test_compile_discrepancy_drives_composed_reduction(self):
        import torch

        def source_evaluator(source: str):
            tree = ast.parse(source)
            shape = None
            dtype = None
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr in {"randn", "ones", "zeros"}
                ):
                    continue
                dimensions = [
                    argument.value
                    for argument in node.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, int)
                ]
                if dimensions:
                    shape = tuple(dimensions)
                dtype_keyword = next((item for item in node.keywords if item.arg == "dtype"), None)
                if dtype_keyword and isinstance(dtype_keyword.value, ast.Attribute):
                    dtype = dtype_keyword.value.attr

            discrepancy = (
                "Buggy()" in source
                and "model(x)" in source
                and shape is not None
                and shape[-1] == 7
                and dtype == "bfloat16"
            )
            reference = torch.zeros(1)
            candidate = torch.ones(1) * 0.25 if discrepancy else reference.clone()
            return reference, candidate

        source = """import torch
from torch import nn

class Innocent(nn.Module):
    def forward(self, x):
        return x

class Buggy(nn.Module):
    def forward(self, x):
        return x

model = nn.Sequential(
    Innocent(),
    Innocent(),
    Buggy(),
    Innocent(),
    Innocent(),
)
x = torch.randn(128, 128, 7, dtype=torch.bfloat16)
model(x)
"""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bug.py"
            script.write_text(source, encoding="utf-8")
            result = reduce(
                script,
                oracle=CompileDifferenceOracle(source_evaluator=source_evaluator),
                timeout=5,
            )

        self.assertEqual(result.reduced_source.count("Buggy()"), 1)
        self.assertNotIn("Innocent()", result.reduced_source)
        self.assertRegex(
            result.reduced_source,
            r"torch\.(randn|zeros|ones)\(1, 1, 7, dtype=torch.bfloat16\)",
        )
        transforms = {entry.get("transform") for entry in result.history}
        self.assertIn("RemoveModules", transforms)
        self.assertIn("TensorShapeChange", transforms)
        self.assertGreaterEqual(result.metrics["candidate_runs"], 1)


if __name__ == "__main__":
    unittest.main()
