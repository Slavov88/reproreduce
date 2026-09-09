import unittest

from reproreduce.hunt import (
    ConfirmationPolicy,
    ExecutionResult,
    FindingDeduplicator,
    FindingFingerprint,
    Operation,
    OutcomeClass,
    Program,
    TensorConfig,
    TensorSpec,
    confirm,
)
from reproreduce.oracle import OracleResult


class HuntConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.program = Program(inputs=(TensorSpec("x", (2, 3)),), operations=(), output="x")
        self.config = (TensorConfig((2, 3)),)
        self.result = ExecutionResult(
            classification=OutcomeClass.GRADIENT_MISMATCH,
            oracle_result=OracleResult(
                True,
                "gradient:gradient_tensor_mismatch",
                metadata={"kind": "gradient_tensor_mismatch"},
            ),
            program_id=self.program.identity,
            backend="aot_eager",
            mode="gradient",
        )

    def test_confirmation_requires_configured_successes(self):
        calls = iter([
            self.result,
            self.result,
            ExecutionResult(OutcomeClass.COMPILED_ERROR, None, "x", "aot_eager", "gradient"),
        ])
        confirmation = confirm(
            lambda: next(calls),
            expected=OutcomeClass.GRADIENT_MISMATCH,
            policy=ConfirmationPolicy(attempts=3, min_successes=2),
        )
        self.assertTrue(confirmation.reproducible)
        self.assertEqual(confirmation.successes, 2)

    def test_confirmation_requires_same_fingerprint_when_requested(self):
        different = ExecutionResult(
            OutcomeClass.GRADIENT_MISMATCH,
            OracleResult(True, "gradient:other", metadata={"kind": "gradient_other"}),
            "x",
            "aot_eager",
            "gradient",
        )
        calls = iter([self.result, different, different])
        confirmation = confirm(
            lambda: next(calls),
            expected=OutcomeClass.GRADIENT_MISMATCH,
            expected_fingerprint=self.result.oracle_result.fingerprint,
            policy=ConfirmationPolicy(attempts=3, min_successes=2),
        )
        self.assertFalse(confirmation.reproducible)

    def test_different_outcomes_are_not_confirmed(self):
        confirmation = confirm(
            lambda: ExecutionResult(OutcomeClass.COMPILED_ERROR, None, "x", "aot_eager", "gradient"),
            expected=OutcomeClass.GRADIENT_MISMATCH,
            policy=ConfirmationPolicy(attempts=3, min_successes=2),
        )
        self.assertFalse(confirmation.reproducible)

    def test_deduplicator_collapses_same_fingerprint(self):
        finding = FindingFingerprint.from_case(self.result, self.program, self.config)
        duplicate = FindingFingerprint.from_case(self.result, self.program, self.config)
        deduplicator = FindingDeduplicator()
        self.assertTrue(deduplicator.add(finding))
        self.assertFalse(deduplicator.add(duplicate))
        self.assertEqual(len(deduplicator.findings), 1)

    def test_operator_sequence_changes_identity(self):
        first = FindingFingerprint.from_case(self.result, self.program, self.config)
        second_program = Program(
            inputs=self.program.inputs,
            operations=(Operation.create("sin", ("x",)),),
            output="v0",
        )
        second = FindingFingerprint.from_case(self.result, second_program, self.config)
        self.assertNotEqual(first.identity, second.identity)


if __name__ == "__main__":
    unittest.main()
