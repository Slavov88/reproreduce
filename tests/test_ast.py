import ast
import unittest

from reproreduce.reduce.ast import reduce_statement_lists


class RecursiveAstTests(unittest.TestCase):
    def test_supported_nested_statement_lists(self):
        fixtures = {
            "function": "def f():\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "async_function": "async def f():\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "if": "if True:\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "for": "for _ in range(1):\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "while": "while True:\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "with": "with open(__file__):\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "try": "try:\n    unused = 1\n    raise RuntimeError('TARGET')\nexcept RuntimeError:\n    handled = 1\n",
            "except_body": "try:\n    raise RuntimeError('OTHER')\nexcept RuntimeError:\n    unused = 1\n    raise RuntimeError('TARGET')\n",
            "try_else": "try:\n    raise RuntimeError('TARGET')\nexcept RuntimeError:\n    pass\nelse:\n    unused = 1\n",
            "try_finally": "try:\n    raise RuntimeError('TARGET')\nfinally:\n    unused = 1\n",
        }
        for name, source in fixtures.items():
            with self.subTest(name=name):
                reduced, _ = reduce_statement_lists(source, lambda candidate: "TARGET" in candidate)
                ast.parse(reduced)
                self.assertIn("TARGET", reduced)
                self.assertNotIn("unused = 1", reduced)

    def test_required_empty_blocks_are_repaired_with_pass(self):
        source = "def helper():\n    unused = 1\n    another_unused = 2\n\nraise RuntimeError('TARGET')\n"
        reduced, _ = reduce_statement_lists(
            source,
            lambda candidate: "def helper" in candidate and "TARGET" in candidate,
        )
        ast.parse(reduced)
        self.assertIn("pass", reduced)

    def test_nested_end_to_end_shape_is_valid(self):
        source = """def reproduce():
    unused = 1
    if True:
        for _ in range(1):
            keep = 2
            raise RuntimeError('TARGET')
        after = 3
reproduce()
"""
        reduced, history = reduce_statement_lists(source, lambda candidate: "TARGET" in candidate)
        ast.parse(reduced)
        scopes = {entry.get("scope") for entry in history if entry.get("transform") == "RemoveStatements"}
        self.assertIn("FunctionDef:reproduce.body", scopes)
        self.assertIn("If.body", scopes)
        self.assertIn("For.body", scopes)


if __name__ == "__main__":
    unittest.main()
