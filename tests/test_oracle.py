import unittest

from reproreduce.core.run import RunResult
from reproreduce.oracle.exception import ExceptionOracle, exception_fingerprint


def run(stderr: str, *, returncode: int = 1, timed_out: bool = False) -> RunResult:
    return RunResult(
        command=("python", "candidate.py"),
        returncode=returncode,
        stdout="",
        stderr=stderr,
        duration_seconds=0.01,
        timed_out=timed_out,
    )


class ExceptionOracleTests(unittest.TestCase):
    def test_same_intended_exception_has_same_fingerprint(self):
        first = run('File "C:\\tmp\\candidate-a.py", line 4, in f\nRuntimeError: TARGET')
        second = run('File "/tmp/candidate-b.py", line 99, in f\nRuntimeError: TARGET')
        self.assertEqual(exception_fingerprint(first).message_signature, "TARGET")
        self.assertEqual(exception_fingerprint(first), exception_fingerprint(second))

    def test_different_exception_class_is_rejected(self):
        oracle = ExceptionOracle(exception_type="RuntimeError", message_regex="TARGET")
        baseline = oracle.evaluate(run("RuntimeError: TARGET"))
        candidate = oracle.evaluate(run("NameError: TARGET"))
        self.assertFalse(candidate.interesting)
        self.assertFalse(oracle.same_failure(baseline, candidate))

    def test_different_message_is_rejected(self):
        oracle = ExceptionOracle(exception_type="RuntimeError")
        baseline = oracle.evaluate(run("RuntimeError: TARGET"))
        candidate = oracle.evaluate(run("RuntimeError: OTHER"))
        self.assertFalse(oracle.same_failure(baseline, candidate))

    def test_timeout_is_not_an_exception_match(self):
        oracle = ExceptionOracle(exception_type="RuntimeError", message_regex="TARGET")
        result = oracle.evaluate(run("RuntimeError: TARGET", timed_out=True))
        self.assertFalse(result.interesting)
        self.assertEqual(result.metadata["reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
