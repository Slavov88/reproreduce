import importlib.util
import tempfile
import unittest
from pathlib import Path

from reproreduce import ExceptionOracle, reduce


PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch is not installed")
class PyTorchTensorReductionTests(unittest.TestCase):
    def _reduce(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bug.py"
            script.write_text(source, encoding="utf-8")
            return reduce(
                script,
                oracle=ExceptionOracle(
                    exception_type="RuntimeError", message_regex="REPROREDUCE_TARGET"
                ),
                # Torch process teardown can exceed 15 seconds on a cold Windows worker.
                timeout=30,
            )

    def test_composed_model_and_tensor_reduction(self):
        result = self._reduce(
            """import torch
from torch import nn

class Innocent(nn.Module):
    def forward(self, x):
        return x

class Buggy(nn.Module):
    def forward(self, x):
        if x.shape[-1] == 7 and x.dtype == torch.bfloat16:
            raise RuntimeError('REPROREDUCE_TARGET')
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
        )
        self.assertEqual(result.reduced_source.count("Buggy()"), 1)
        self.assertNotIn("Innocent()", result.reduced_source)
        self.assertRegex(result.reduced_source, r"torch\.(randn|zeros|ones)\(1, 1, 7, dtype=torch.bfloat16\)")
        self.assertIn("REPROREDUCE_TARGET", result.reduced_source)

    def test_iterated_module_list_reduces_to_indispensable_module(self):
        result = self._reduce(
            """import torch
from torch import nn

class Innocent(nn.Module):
    def forward(self, x):
        return x + 1

class Buggy(nn.Module):
    def forward(self, x):
        raise RuntimeError('REPROREDUCE_TARGET')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Innocent(),
            Innocent(),
            Buggy(),
            Innocent(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

Model()(torch.ones(1))
"""
        )
        self.assertEqual(result.reduced_source.count("Buggy()"), 1)
        self.assertNotIn("Innocent()", result.reduced_source)
        self.assertIn("nn.ModuleList", result.reduced_source)

    def test_sequential_reduces_to_indispensable_module(self):
        result = self._reduce(
            """import torch
from torch import nn

class Innocent(nn.Module):
    def forward(self, x):
        return x + 1

class Buggy(nn.Module):
    def forward(self, x):
        raise RuntimeError('REPROREDUCE_TARGET')

model = nn.Sequential(
    Innocent(),
    Innocent(),
    Buggy(),
    Innocent(),
    Innocent(),
)
model(torch.ones(1))
"""
        )
        self.assertEqual(result.reduced_source.count("Buggy()"), 1)
        self.assertNotIn("Innocent()", result.reduced_source)
        self.assertIn("RemoveModules", {entry.get("transform") for entry in result.history})

    def test_shape_dependent_failure(self):
        result = self._reduce(
            """import torch
junk = torch.randn(1000, 1000)
def reproduce():
    x = torch.randn(128, 128, 7)
    y = x + 1
    z = y * 2
    if x.shape[-1] == 7:
        raise RuntimeError('REPROREDUCE_TARGET')
reproduce()
"""
        )
        self.assertNotIn("junk", result.reduced_source)
        self.assertNotIn("y =", result.reduced_source)
        self.assertNotIn("z =", result.reduced_source)
        self.assertRegex(result.reduced_source, r"torch\.(randn|zeros|ones)\(1, 1, 7\)")
        self.assertIn("REPROREDUCE_TARGET", result.reduced_source)

    def test_dtype_dependent_failure(self):
        result = self._reduce(
            """import torch
x = torch.randn(32, 32, dtype=torch.bfloat16)
if x.dtype == torch.bfloat16:
    raise RuntimeError('REPROREDUCE_TARGET')
"""
        )
        self.assertIn("dtype=torch.bfloat16", result.reduced_source)
        self.assertNotIn("dtype=torch.float32", result.reduced_source)
        self.assertTrue(
            any(
                entry.get("transform") == "DtypeAttempt"
                and entry.get("attempted") == "float32"
                and entry.get("oracle") == "FAIL"
                for entry in result.history
            )
        )

    def test_layout_dependent_failure_rejects_contiguous(self):
        result = self._reduce(
            """import torch
x = torch.randn(64, 64).transpose(0, 1)
if not x.is_contiguous():
    raise RuntimeError('REPROREDUCE_TARGET')
"""
        )
        self.assertIn("transpose", result.reduced_source)
        self.assertNotIn(".contiguous()", result.reduced_source)
        self.assertIn("REPROREDUCE_TARGET", result.reduced_source)


if __name__ == "__main__":
    unittest.main()
