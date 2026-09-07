import tempfile
import unittest
from pathlib import Path

from reproreduce import ExceptionOracle, reduce
from reproreduce.core.cache import CandidateCache
from reproreduce.core.session import ReductionSession


class ReductionMetricsTests(unittest.TestCase):
    def test_reduction_reports_metrics_without_changing_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "bug.py"
            script.write_text(
                "unused = 1\nraise RuntimeError('TARGET')\n",
                encoding="utf-8",
            )
            result = reduce(
                script,
                oracle=ExceptionOracle("RuntimeError", "TARGET"),
                timeout=5,
            )

        self.assertGreaterEqual(result.metrics["candidate_runs"], 1)
        self.assertGreaterEqual(result.metrics["cache_misses"], 1)
        self.assertGreaterEqual(result.metrics["total_reduction_wall_time"], 0.0)
        self.assertEqual(result.reduced_run.returncode, result.original_run.returncode)
        self.assertIn("Candidate runs", result.summary())

    def test_cache_hits_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "bug.py"
            cache_path = root / "cache.sqlite3"
            script.write_text("raise RuntimeError('TARGET')\n", encoding="utf-8")
            session = ReductionSession(
                program=script,
                oracle=ExceptionOracle("RuntimeError", "TARGET"),
                timeout=5,
                cache_path=cache_path,
            )
            session._cache = CandidateCache(cache_path)
            try:
                first = session._evaluate(session.source)
                second = session._evaluate(session.source)
            finally:
                session._cache.close()

        self.assertEqual(first[0].stderr, second[0].stderr)
        self.assertEqual(session._candidate_runs, 1)
        self.assertEqual(session._cache_hits, 1)
        self.assertEqual(session._cache_misses, 1)


if __name__ == "__main__":
    unittest.main()
