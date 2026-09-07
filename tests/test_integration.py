import tempfile
import unittest
from pathlib import Path

from reproreduce import ExceptionOracle, reduce


class IntegrationTests(unittest.TestCase):
    def test_reduces_bloated_exception_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "bug.py"
            script.write_text(
                "import math\nimport random\n\n"
                "unused_a = 123\nunused_b = [1, 2, 3]\n\n"
                "def irrelevant():\n    return math.sqrt(9)\n\n"
                "x = 7\nif x == 7:\n    raise RuntimeError('REPROREDUCE_TARGET')\n",
                encoding="utf-8",
            )
            result = reduce(
                script,
                oracle=ExceptionOracle(
                    exception_type="RuntimeError", message_regex="REPROREDUCE_TARGET"
                ),
                timeout=5,
            )
            self.assertLess(result.reduced_loc, result.original_loc)
            self.assertIn("REPROREDUCE_TARGET", result.reduced_source)
            compile(result.reduced_source, "repro.py", "exec")

    def test_rejects_unrelated_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bug.py"
            script.write_text(
                "raise RuntimeError('TARGET')\n"
                "raise NameError('UNRELATED')\n",
                encoding="utf-8",
            )
            result = reduce(
                script,
                oracle=ExceptionOracle(exception_type="RuntimeError", message_regex="TARGET"),
                timeout=5,
            )
            self.assertIn("RuntimeError", result.reduced_source)
            self.assertNotIn("UNRELATED", result.reduced_source)


if __name__ == "__main__":
    unittest.main()
