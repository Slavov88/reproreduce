import unittest

from reproreduce.hunt import (
    ConfirmationPolicy,
    ExecutionResult,
    OutcomeClass,
    run_campaign,
)
from reproreduce.oracle import OracleResult


class FakeExecutor:
    def __init__(self, classification):
        self.classification = classification
        self.calls = 0

    def run(self, program, configs, *, mode):
        self.calls += 1
        return ExecutionResult(
            classification=self.classification,
            oracle_result=OracleResult(
                True,
                "synthetic",
                metadata={"kind": "tensor_mismatch"},
            ),
            program_id=program.identity,
            backend="synthetic",
            mode=mode,
        )


class HuntCampaignTests(unittest.TestCase):
    def test_end_to_end_generate_detect_confirm_deduplicate(self):
        executor = FakeExecutor(OutcomeClass.FORWARD_MISMATCH)
        stats, findings = run_campaign(
            cases=3,
            seed=12,
            mode="forward",
            confirmation=ConfirmationPolicy(attempts=2, min_successes=2),
            executor=executor,
        )
        self.assertEqual(stats.cases_generated, 3)
        self.assertEqual(stats.forward_mismatches, 3)
        self.assertEqual(stats.unique_findings, 3)
        self.assertEqual(len(findings.findings), 3)
        self.assertEqual(executor.calls, 9)

    def test_candidate_only_errors_are_counted_separately(self):
        stats, findings = run_campaign(
            cases=2,
            executor=FakeExecutor(OutcomeClass.COMPILED_ERROR),
        )
        self.assertEqual(stats.candidate_only_errors, 2)
        self.assertEqual(stats.unique_findings, 0)
        self.assertFalse(findings.findings)


if __name__ == "__main__":
    unittest.main()
