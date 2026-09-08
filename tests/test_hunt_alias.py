import ast
import json
import tempfile
import unittest
from pathlib import Path

import torch

from reproreduce.hunt import (
    AliasMutationGenerator,
    OutcomeClass,
    ProgramExecutor,
    coverage_summary,
    run_campaign,
)


class AliasMutationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_serializable(self):
        first = AliasMutationGenerator(5000).generate()
        second = AliasMutationGenerator(5000).generate()
        self.assertEqual(first.program.identity, second.program.identity)
        self.assertEqual(first.program.to_dict(), second.program.to_dict())
        self.assertEqual(first.program.metadata_dict()["family"], "alias_mutation")
        ast.parse(first.program.to_source())

    def test_all_structured_cases_are_eager_valid_and_aliasing_is_real(self):
        executor = ProgramExecutor(backend="test", compiler=lambda function: function)
        patterns = set()
        mutations = set()
        for seed in range(5000, 5064):
            with self.subTest(seed=seed):
                case = AliasMutationGenerator(seed).generate()
                result = executor.run_alias(
                    case.program,
                    case.configs,
                    return_names=case.return_names,
                    alias_pairs=case.alias_pairs,
                    input_seed=seed,
                )
                self.assertEqual(result.classification, OutcomeClass.PASS)
                self.assertTrue(all(check["valid"] for check in result.alias_metadata["eager_aliases"]))
                patterns.add(case.program.metadata_dict()["alias_pattern"])
                mutations.add(case.program.metadata_dict()["mutation"]["op"])
        self.assertEqual(len(patterns), 8)
        self.assertEqual(len(mutations), 8)

    def test_mutating_view_changes_base_storage(self):
        case = AliasMutationGenerator(5000).generate()
        namespace = {}
        exec(compile(case.program.to_source(), "alias.py", "exec"), namespace)
        value = case.configs[0].materialize(seed=5000)
        before = value.detach().clone()
        output = namespace["generated_program"](value)
        self.assertTrue(torch.equal(value, output[0]))
        self.assertFalse(torch.equal(before, value))
        self.assertTrue(torch._C._is_alias_of(output[0], output[1]))

    def test_campaign_checkpoint_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "alias.json"
            stats, _ = run_campaign(
                cases=16,
                seed=5000,
                family="alias_mutation",
                executor=ProgramExecutor(backend="test", compiler=lambda function: function),
                checkpoint=checkpoint,
            )
            self.assertEqual(stats.cases_generated, 16)
            self.assertEqual(stats.invalid_cases, 0)
            report = json.loads(checkpoint.read_text())
            self.assertEqual(report["experiment"]["state_observation"], True)
            self.assertEqual(sum(report["coverage"]["alias_pattern"].values()), 16)
            self.assertEqual(sum(report["coverage"]["mutation"].values()), 16)
            self.assertIn("overlapping", report["coverage"])
            self.assertEqual(coverage_summary(report["cases"]), report["coverage"])

    def test_gradient_mode_is_explicitly_rejected_for_v1(self):
        with self.assertRaises(ValueError):
            run_campaign(
                cases=1,
                seed=6000,
                family="alias_mutation",
                mode="gradient",
                executor=ProgramExecutor(backend="test", compiler=lambda function: function),
            )


if __name__ == "__main__":
    unittest.main()
