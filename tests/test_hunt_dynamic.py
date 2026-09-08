import ast
import json
import tempfile
import unittest
from pathlib import Path

from reproreduce.hunt import (
    DynamicShapeGenerator,
    OutcomeClass,
    ProgramExecutor,
    coverage_summary,
    run_campaign,
)


class DynamicShapeTests(unittest.TestCase):
    def test_generation_is_deterministic_and_serializable(self):
        first = DynamicShapeGenerator(3003, mode="gradient").generate()
        second = DynamicShapeGenerator(3003, mode="gradient").generate()
        self.assertEqual(first.program.identity, second.program.identity)
        self.assertEqual(first.to_dict(), second.to_dict())
        ast.parse(first.program.to_source())
        metadata = first.program.metadata_dict()
        self.assertTrue(metadata["dynamic_compile"])
        self.assertEqual(metadata["family"], "dynamic")
        self.assertEqual(len(metadata["shape_trace"]), 4)
        self.assertEqual(metadata["shape_trace"], first.shape_trace)

    def test_traces_have_multiple_compatible_shapes_and_no_zero_dimensions(self):
        for seed in range(3000, 3016):
            with self.subTest(seed=seed):
                case = DynamicShapeGenerator(seed).generate()
                self.assertEqual(case.trace_length, 4)
                self.assertGreater(len({step[0] for step in case.shape_trace}), 1)
                for step in case.shape_trace:
                    for shape in step:
                        self.assertTrue(shape)
                        self.assertTrue(all(dimension > 0 for dimension in shape))

    def test_eager_output_shapes_match_trace_metadata(self):
        import torch

        for seed in range(3000, 3008):
            with self.subTest(seed=seed):
                case = DynamicShapeGenerator(seed).generate()
                namespace = {}
                exec(compile(case.program.to_source(), "dynamic.py", "exec"), namespace)
                outputs = []
                for index, configs in enumerate(case.config_trace):
                    inputs = tuple(
                        config.materialize(seed=seed + index * 1009 + position)
                        for position, config in enumerate(configs)
                    )
                    output = namespace["generated_program"](*inputs)
                    self.assertTrue(torch.is_tensor(output))
                    outputs.append(tuple(output.shape))
                self.assertEqual(outputs, list(case.program.metadata_dict()["output_shape_trace"]))

    def test_dynamic_callable_is_compiled_once_and_reused(self):
        calls = []

        def compiler(function):
            calls.append(function)
            return function

        case = DynamicShapeGenerator(3004).generate()
        result = ProgramExecutor(backend="test", compiler=compiler).run_dynamic(
            case.program,
            case.config_trace,
            input_seed=3004,
        )
        self.assertEqual(result.classification, OutcomeClass.PASS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.shape_executions, 4)
        self.assertEqual(len(result.shape_results), 4)

    def test_static_trace_is_a_separate_per_shape_control(self):
        calls = []

        def compiler(function):
            calls.append(function)
            return function

        case = DynamicShapeGenerator(3003).generate()
        executor = ProgramExecutor(backend="test", compiler=compiler)
        results = executor.run_static_trace(case.program, case.config_trace, input_seed=3003)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.classification == OutcomeClass.PASS for result in results))
        self.assertEqual(len(calls), 4)

    def test_gradient_trace_is_differentiable(self):
        for seed in range(4000, 4008):
            with self.subTest(seed=seed):
                case = DynamicShapeGenerator(seed, mode="gradient").generate()
                result = ProgramExecutor(backend="test", compiler=lambda function: function).run_dynamic(
                    case.program,
                    case.config_trace,
                    mode="gradient",
                    input_seed=seed,
                )
                self.assertEqual(result.classification, OutcomeClass.PASS)

    def test_campaign_checkpoint_and_coverage_include_dynamic_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dynamic.json"
            stats, _ = run_campaign(
                cases=4,
                seed=3000,
                family="dynamic",
                executor=ProgramExecutor(backend="test", compiler=lambda function: function),
                checkpoint=checkpoint,
            )
            self.assertEqual(stats.cases_generated, 4)
            self.assertEqual(stats.shape_executions, 16)
            self.assertTrue(checkpoint.exists())
            report = json.loads(checkpoint.read_text())
            self.assertEqual(len(report["cases"]), 4)
            self.assertEqual(report["shape_executions"], 16)
            self.assertEqual(report["experiment"]["dynamic_compile"], True)
            coverage = coverage_summary(report["cases"])
            self.assertEqual(sum(coverage["symbolic_pattern"].values()), 4)
            self.assertEqual(sum(coverage["varying_dim_count"].values()), 4)


if __name__ == "__main__":
    unittest.main()
