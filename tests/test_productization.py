import ast
import unittest
from pathlib import Path

import reproreduce
from reproreduce.cli.main import build_parser


class ProductizationTests(unittest.TestCase):
    def test_version_and_public_api_are_available(self):
        self.assertEqual(reproreduce.__version__, "0.2.0")
        self.assertTrue(callable(reproreduce.reduce))
        args = build_parser().parse_args(["summarize", "campaign.json"])
        self.assertEqual(args.command, "summarize")
        reduce_args = build_parser().parse_args(["reduce", "program.py", "--strategy", "dependency_v3"])
        self.assertEqual(reduce_args.strategy, "dependency_v3")

    def test_flagship_fixture_is_tracked_and_parseable(self):
        root = Path(__file__).parents[1]
        source = root.joinpath("examples", "inductor_index_fill", "bug.py")
        readme = root.joinpath("examples", "inductor_index_fill", "README.md")
        self.assertTrue(source.is_file())
        self.assertTrue(readme.is_file())
        ast.parse(source.read_text(encoding="utf-8"))
        text = source.read_text(encoding="utf-8")
        self.assertIn("index_fill_", text)
        self.assertIn("transpose", text)
        self.assertIn("#178952", readme.read_text(encoding="utf-8"))

    def test_architecture_and_release_docs_exist(self):
        root = Path(__file__).parents[1]
        self.assertIn("ReductionResult", root.joinpath("docs", "ARCHITECTURE.md").read_text(encoding="utf-8"))
        self.assertIn("reproreduce --version", root.joinpath("docs", "RELEASE_CHECKLIST.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
