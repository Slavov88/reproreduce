import ast
import unittest

from reproreduce.pytorch.modules import reduce_sequential_modules


class SequentialModuleReducerTests(unittest.TestCase):
    def test_one_indispensable_module(self):
        source = "model = nn.Sequential(Innocent(), Optional(), Buggy(), Noise())\n"
        reduced, history = reduce_sequential_modules(
            source,
            lambda candidate: "Buggy()" in candidate,
        )
        self.assertEqual(reduced.count("Buggy()"), 1)
        self.assertNotIn("Optional()", reduced)
        self.assertNotIn("Noise()", reduced)
        self.assertTrue(any(entry["transform"] == "RemoveModules" for entry in history))
        ast.parse(reduced)

    def test_multiple_indispensable_modules(self):
        source = "model = nn.Sequential(RequiredA(), Optional(), RequiredB(), Noise())\n"
        reduced, _ = reduce_sequential_modules(
            source,
            lambda candidate: all(name in candidate for name in ("RequiredA()", "RequiredB()")),
        )
        self.assertIn("RequiredA()", reduced)
        self.assertIn("RequiredB()", reduced)
        self.assertNotIn("Optional()", reduced)
        self.assertNotIn("Noise()", reduced)

    def test_every_module_required(self):
        source = "model = nn.Sequential(RequiredA(), RequiredB())\n"
        reduced, history = reduce_sequential_modules(
            source,
            lambda candidate: all(name in candidate for name in ("RequiredA()", "RequiredB()")),
        )
        self.assertEqual(reduced, "model = nn.Sequential(RequiredA(), RequiredB())\n")
        self.assertEqual(history, [])

    def test_starred_sequential_is_conservative(self):
        source = "model = nn.Sequential(*layers)\n"
        reduced, history = reduce_sequential_modules(source, lambda _: True)
        self.assertEqual(reduced, source)
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
