import unittest

from reproreduce.core.session import _SessionCandidateTest
from reproreduce.reduce.ddmin import ddmin
from reproreduce.reduce.scheduler import invoke_test


class SearchSchedulerTests(unittest.TestCase):
    def test_ddmin_reports_attempt_state_without_changing_result(self):
        attempts = []
        reduced = ddmin(
            [0, 1, 2, 3],
            lambda candidate: 0 in candidate,
            on_attempt=lambda parent, candidate, granularity: attempts.append(
                (tuple(parent), tuple(candidate), granularity)
            ),
        )
        self.assertEqual(reduced, [0])
        self.assertTrue(attempts)
        self.assertGreaterEqual(attempts[0][2], 2)

    def test_scheduler_reuses_duplicate_outcome_before_session_evaluation(self):
        class FakeSession:
            def __init__(self):
                self._scheduler_requests = 0
                self._scheduler_sources = set()
                self.skips = 0
                self.evaluations = 0

            def _record_skipped_duplicate(self, source, context, accepted):
                self.skips += 1

            def _preserves_failure(self, source, *, context):
                self.evaluations += 1
                return source == "keep"

        session = FakeSession()
        test = _SessionCandidateTest(session, deduplicate=True)
        self.assertTrue(test("keep"))
        self.assertTrue(test("keep"))
        self.assertEqual(session.evaluations, 1)
        self.assertEqual(session.skips, 1)
        self.assertEqual(session._scheduler_requests, 2)

    def test_invoke_test_attaches_context_to_traceable_test(self):
        seen = []

        class Traceable:
            def set_context(self, metadata):
                self.metadata = metadata

            def __call__(self, source):
                seen.append((source, self.metadata))
                return True

        self.assertTrue(invoke_test(Traceable(), "candidate", metadata={"scope": "body"}))
        self.assertEqual(seen, [("candidate", {"scope": "body"})])


if __name__ == "__main__":
    unittest.main()
