import tempfile
import time
import unittest
from pathlib import Path

from reproreduce.core.run import RunResult
from reproreduce.cli.main import build_parser
from reproreduce.core.session import ReductionSession
from reproreduce.oracle.base import OracleResult
from reproreduce.oracle.exception import ExceptionOracle


class ParallelEvaluationTests(unittest.TestCase):
    def test_cli_exposes_jobs(self):
        args = build_parser().parse_args(["reduce", "bug.py", "--jobs", "4"])
        self.assertEqual(args.jobs, 4)

    def test_ordered_results_and_pending_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "bug.py"
            program.write_text("raise RuntimeError('target')\n", encoding="utf-8")
            session = ReductionSession(
                program=program,
                oracle=ExceptionOracle(exception_type="RuntimeError", message_regex="target"),
                timeout=5,
                cache_path=None,
                jobs=2,
            )
            session._baseline = OracleResult(
                True,
                "target",
                metadata={"exception_type": "RuntimeError", "message_signature": "target", "signal": None},
            )

            def fake_execute(source):
                delay = {"fail": 0.01, "accept": 0.2}.get(source, 2.0)
                time.sleep(delay)
                result = OracleResult(
                    source == "accept",
                    "target",
                    metadata={"exception_type": "RuntimeError", "message_signature": "target", "signal": None},
                )
                run = RunResult(
                    command=("fake", source),
                    returncode=1,
                    stdout="",
                    stderr="",
                    duration_seconds=delay,
                )
                return run, result, delay, 0.0

            session._execute_uncached = fake_execute
            outcomes = session._evaluate_source_batch(
                ["fail", "accept", "late-a", "late-b"],
                [{}, {}, {}, {}],
                jobs=2,
            )

            self.assertEqual(outcomes, [False, True, False, False])
            self.assertEqual(session._peak_concurrency, 2)
            self.assertEqual(session._candidates_submitted, 4)
            self.assertGreaterEqual(session._candidates_completed, 2)
            self.assertGreaterEqual(session._candidates_cancelled, 0)
            self.assertEqual(
                session._candidates_completed + session._candidates_cancelled,
                session._candidates_submitted,
            )
            self.assertEqual(
                session._speculative_executions,
                session._candidates_completed - 2,
            )
            self.assertEqual(session._useful_executions, 2)

    def test_jobs_one_uses_serial_execution_path(self):
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "bug.py"
            program.write_text("raise RuntimeError('target')\n", encoding="utf-8")
            session = ReductionSession(
                program=program,
                oracle=ExceptionOracle(exception_type="RuntimeError", message_regex="target"),
                timeout=5,
                cache_path=None,
                jobs=1,
            )
            session._baseline = OracleResult(
                True,
                "target",
                metadata={"exception_type": "RuntimeError", "message_signature": "target", "signal": None},
            )
            calls = []

            def fake_execute(source):
                calls.append(source)
                run = RunResult(("fake",), 1, "", "", 0.0)
                return run, OracleResult(
                    source == "accept",
                    "target",
                    metadata={"exception_type": "RuntimeError", "message_signature": "target", "signal": None},
                ), 0.0, 0.0

            session._execute_uncached = fake_execute
            self.assertEqual(
                session._evaluate_source_batch(["fail", "accept"], [{}, {}], jobs=1),
                [False, True],
            )
            self.assertEqual(calls, ["fail", "accept"])
            self.assertEqual(session._peak_concurrency, 0)


if __name__ == "__main__":
    unittest.main()
