import tempfile
import unittest
from pathlib import Path

import torch

from reproreduce.hunt import (
    DynamicShapeGenerator,
    Operation,
    Program,
    TensorConfig,
    TensorSpec,
    export_finding,
    reduce_confirmed,
)
from reproreduce.oracle import CompileDifferenceOracle


class HuntExportTests(unittest.TestCase):
    def test_exports_dynamic_trace_harness(self):
        case = DynamicShapeGenerator(3000, mode="gradient").generate()
        program = case.program

        def source_evaluator(source):
            return (torch.zeros(1), torch.ones(1)) if "torch.sin" in source else (torch.zeros(1), torch.zeros(1))

        result = reduce_confirmed(
            program,
            CompileDifferenceOracle(source_evaluator=source_evaluator),
            timeout=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = export_finding(
                result,
                Path(directory) / "dynamic-finding",
                program=program,
                config_trace=case.config_trace,
                mode="gradient",
                backend="eager",
                input_seed=3000,
            )
            repro_source = (output / "repro.py").read_text(encoding="utf-8")
            compile(repro_source, "dynamic-repro.py", "exec")
            self.assertIn("dynamic=True", repro_source)
            self.assertIn("shape index 0", repro_source)

    def test_exports_issue_ready_finding_files(self):
        program = Program(
            inputs=(TensorSpec("x", (2, 3)),),
            operations=(Operation.create("sin", ("x",)),),
            output="v0",
        )

        def source_evaluator(source):
            return (torch.zeros(1), torch.ones(1)) if "torch.sin" in source else (torch.zeros(1), torch.zeros(1))

        result = reduce_confirmed(
            program,
            CompileDifferenceOracle(source_evaluator=source_evaluator),
            timeout=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = export_finding(
                result,
                Path(directory) / "finding",
                finding={"class": "forward_mismatch"},
                program=program,
                configs=(TensorConfig((2, 3), requires_grad=False),),
                backend="eager",
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"repro.py", "report.md", "environment.json", "finding.json"},
            )
            repro_source = (output / "repro.py").read_text(encoding="utf-8")
            compile(repro_source, "repro.py", "exec")
            self.assertIn("eager result", repro_source)
            self.assertIn("forward_mismatch", (output / "finding.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
