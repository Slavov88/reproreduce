import importlib.util
import unittest

from reproreduce.hunt import TensorConfig, TensorConfigGenerator


PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch is not installed")
class HuntConfigTests(unittest.TestCase):
    def test_seed_reproducibility_and_serialization(self):
        first = TensorConfigGenerator(7).generate()
        second = TensorConfigGenerator(7).generate()
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_generated_configurations_materialize(self):
        for seed in range(30):
            with self.subTest(seed=seed):
                config = TensorConfigGenerator(seed).generate()
                tensor = config.materialize()
                self.assertEqual(str(tensor.dtype).removeprefix("torch."), config.dtype)
                self.assertEqual(tensor.requires_grad, config.requires_grad)
                if config.layout == "contiguous":
                    self.assertTrue(tensor.is_contiguous())
                elif config.layout == "transpose" and all(value > 1 for value in config.shape[:2]):
                    self.assertFalse(tensor.is_contiguous())
                elif config.layout == "slice" and config.shape[-1] > 2:
                    self.assertFalse(tensor.is_contiguous())

    def test_invalid_layouts_are_rejected(self):
        with self.assertRaises(ValueError):
            TensorConfig(shape=(2,), layout="transpose").materialize()
        with self.assertRaises(ValueError):
            TensorConfig(shape=(2,), layout="unknown").materialize()


if __name__ == "__main__":
    unittest.main()
