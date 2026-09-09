import ast
import unittest
from pathlib import Path

from reproreduce.bench import SPECS, _determinism, _nonblank_loc


class LargeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parents[1]
        self.directory = self.root / "benchmarks" / "large_reduction_v1"

    def test_fixture_sizes_and_syntax(self):
        expected = {
            "large_exception.py": (250, 500),
            "large_inductor_index_fill.py": (150, 300),
            "nested_python.py": (200, 400),
            "generated_pytorch.py": (300, 800),
        }
        for filename, (minimum, maximum) in expected.items():
            with self.subTest(filename=filename):
                source = (self.directory / filename).read_text(encoding="utf-8")
                ast.parse(source)
                self.assertGreaterEqual(_nonblank_loc(source), minimum)
                self.assertLessEqual(_nonblank_loc(source), maximum)

    def test_specs_cover_each_tracked_fixture(self):
        names = {spec.filename for spec in SPECS}
        self.assertEqual(
            names,
            {
                "large_exception.py",
                "large_inductor_index_fill.py",
                "nested_python.py",
                "generated_pytorch.py",
            },
        )

    def test_determinism_report_compares_reduced_hashes(self):
        records = [
            {"name": "x", "reduced_nonblank_loc": 5, "reduced_source_sha256": "a", "candidate_runs": 3, "fingerprint_preserved": True},
            {"name": "x", "reduced_nonblank_loc": 5, "reduced_source_sha256": "a", "candidate_runs": 3, "fingerprint_preserved": True},
        ]
        result = _determinism(records)["x"]
        self.assertTrue(result["exact_reduced_source_match"])
        self.assertTrue(result["exact_candidate_count_match"])


if __name__ == "__main__":
    unittest.main()
