import ast
import unittest

import torch

from reproreduce.hunt import (
    OutcomeClass,
    ProgramExecutor,
    StructuredBroadcastGenerator,
    coverage_summary,
)


class StructuredBroadcastGeneratorTests(unittest.TestCase):
    def test_deterministic_metadata_and_program_identity(self):
        first = StructuredBroadcastGenerator(17, mode="gradient").generate()
        second = StructuredBroadcastGenerator(17, mode="gradient").generate()
        self.assertEqual(first.program.identity, second.program.identity)
        self.assertEqual(first.program.metadata_dict(), second.program.metadata_dict())
        ast.parse(first.program.to_source())

    def test_broadcast_shapes_are_compatible_and_metadata_is_serializable(self):
        for seed in range(72):
            with self.subTest(seed=seed):
                case = StructuredBroadcastGenerator(seed, mode="forward").generate()
                metadata = case.program.metadata_dict()
                self.assertEqual(metadata["broadcast_pattern"], StructuredBroadcastGenerator._PATTERNS[seed % 6])
                self.assertEqual(
                    metadata["broadcast_shape"],
                    tuple(metadata["broadcast_shape"]),
                )
                self.assertIn("output_shape", metadata)
                self.assertIn("input_layouts", metadata)
                self.assertIn("dtype", metadata)
                self.assertEqual(len(case.program.inputs), len(case.configs))
                self.assertEqual(case.program.to_dict()["metadata"]["dtype"], metadata["dtype"])

    def test_generated_cases_execute_eagerly_and_have_expected_output_shapes(self):
        for seed in range(72):
            with self.subTest(seed=seed):
                case = StructuredBroadcastGenerator(seed, mode="forward").generate()
                namespace = {}
                source = case.program.to_source()
                exec(compile(source, "broadcast.py", "exec"), namespace)
                inputs = tuple(config.materialize(seed=seed + index) for index, config in enumerate(case.configs))
                output = namespace["generated_program"](*inputs)
                self.assertEqual(tuple(output.shape), tuple(case.program.metadata_dict()["output_shape"]))
                for config, value in zip(case.configs, inputs):
                    self.assertEqual(tuple(value.shape), config.shape)

    def test_gradient_cases_are_differentiable(self):
        executor = ProgramExecutor(backend="eager")
        for seed in range(36):
            with self.subTest(seed=seed):
                case = StructuredBroadcastGenerator(seed, mode="gradient").generate()
                result = executor.run(
                    case.program,
                    case.configs,
                    mode="gradient",
                    input_seed=seed,
                )
                self.assertEqual(result.classification, OutcomeClass.PASS)
                self.assertEqual(case.program.metadata_dict()["mode"], "gradient")

    def test_coverage_summary_counts_structured_cells(self):
        cases = []
        for seed in range(24):
            case = StructuredBroadcastGenerator(seed, mode="forward").generate()
            cases.append({"coverage": case.program.metadata_dict()})
        summary = coverage_summary(cases)
        self.assertEqual(sum(summary["broadcast_pattern"].values()), 24)
        self.assertEqual(sum(summary["dtype"].values()), 24)
        self.assertGreaterEqual(len(summary["broadcast_pattern"]), 6)
        self.assertGreaterEqual(len(summary["layout"]), 2)

    def test_noncontiguous_layout_cells_are_real(self):
        seen = 0
        for seed in range(72):
            case = StructuredBroadcastGenerator(seed, mode="forward").generate()
            values = tuple(config.materialize(seed=seed + index) for index, config in enumerate(case.configs))
            for config, value in zip(case.configs, values):
                if config.layout != "contiguous" and not value.is_contiguous():
                    seen += 1
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
