import json
import tempfile
import unittest
from pathlib import Path

from reproreduce.cli.main import build_parser
from reproreduce.hunt import (
    cluster_failure_records,
    failure_metadata,
    failure_stage,
    load_failure_clusters,
    normalize_failure_message,
)


class FailureDiagnosticsTests(unittest.TestCase):
    def test_normalization_removes_paths_addresses_and_node_suffixes(self):
        first = "AssertionError: n=copy_, n.args[0]=permute_1, placeholders={arg1_1, arg0_1}, path=/tmp/run-1, ptr=0xabc"
        second = "AssertionError: n=copy_, n.args[0]=permute, placeholders={arg0_1, arg1_1}, path=/tmp/run-2, ptr=0xdef"
        self.assertEqual(normalize_failure_message(first), normalize_failure_message(second))

    def test_stage_classification_distinguishes_timeout_and_lowering(self):
        self.assertEqual(
            failure_stage("BackendCompilerFailed", "AssertionError: n=copy_, graph=graph()", phase="runtime"),
            "backend_wrapped_unknown",
        )
        self.assertEqual(
            failure_stage("BackendCompilerFailed", "HuntTimeout: case exceeded 30.0 seconds", phase="runtime"),
            "harness_timeout",
        )
        self.assertEqual(
            failure_stage(
                "BackendCompilerFailed",
                "AssertionError: n=copy_",
                phase="runtime",
                traceback_text="torch/_functorch/_aot_autograd/functional_utils.py assert_functional_graph",
            ),
            "aot_autograd",
        )

    def test_clusters_keep_distinct_stages_separate(self):
        records = [
            {
                "seed": 1,
                "failure_phase": "runtime",
                "metadata": {
                    "exception_type": "BackendCompilerFailed",
                    "exception_message": "AssertionError: n=copy_, n.args[0]=permute_1",
                },
                "coverage": {"mutation": {"op": "index_fill_"}, "dtype": "float32", "layout": "slice"},
            },
            {
                "seed": 2,
                "failure_phase": "runtime",
                "metadata": {
                    "exception_type": "BackendCompilerFailed",
                    "exception_message": "AssertionError: n=copy_, n.args[0]=permute",
                },
                "coverage": {"mutation": {"op": "index_fill_"}, "dtype": "float64", "layout": "transpose"},
            },
            {
                "seed": 3,
                "failure_phase": "runtime",
                "metadata": {
                    "exception_type": "BackendCompilerFailed",
                    "exception_message": "HuntTimeout: case exceeded 30.0 seconds",
                },
                "coverage": {"mutation": {"op": "sub_"}, "dtype": "float64", "layout": "contiguous"},
            },
        ]
        clusters = cluster_failure_records(records)
        self.assertEqual(len(clusters), 2)
        counts = sorted(cluster["count"] for cluster in clusters.values())
        self.assertEqual(counts, [1, 2])

    def test_summary_command_is_parseable(self):
        args = build_parser().parse_args(["summarize", "campaign.json"])
        self.assertEqual(args.command, "summarize")
        self.assertEqual(args.campaign, "campaign.json")

    def test_serialization_and_campaign_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps({"cases": [{
                "seed": 4,
                "classification": "COMPILED_RUNTIME_FAILURE",
                "failure_phase": "runtime",
                "metadata": {"exception_type": "X", "exception_message": "RuntimeError: bad"},
                "coverage": {"mutation": {"op": "fill_"}, "dtype": "float32", "layout": "contiguous"},
            }]}), encoding="utf-8")
            clusters = load_failure_clusters(path)
            self.assertEqual(sum(item["count"] for item in clusters.values()), 1)
            self.assertIn("failure_fingerprint", failure_metadata("X", "RuntimeError: bad", phase="runtime"))


if __name__ == "__main__":
    unittest.main()
