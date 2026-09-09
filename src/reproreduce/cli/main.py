from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version

from ..api import reduce
from ..oracle.exception import ExceptionOracle


def _package_version() -> str:
    try:
        return version("reproreduce")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reproreduce",
        description="Reduce failing Python/PyTorch programs and inspect compiler failures.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    commands = parser.add_subparsers(dest="command", required=True)
    reduce_parser = commands.add_parser(
        "reduce",
        help="reduce a failing Python script while preserving its oracle",
        description="Reduce a self-contained failing Python script and export a reproducer.",
    )
    reduce_parser.add_argument("program")
    reduce_parser.add_argument("--oracle", choices=["exception"], default="exception")
    reduce_parser.add_argument("--exception-type")
    reduce_parser.add_argument("--message")
    reduce_parser.add_argument("--timeout", type=float, default=30.0)
    reduce_parser.add_argument("--output", default="repro")
    reduce_parser.add_argument("--cache")
    hunt_parser = commands.add_parser(
        "hunt",
        help="search generated PyTorch programs",
        description="Run a structured eager-versus-compiled PyTorch campaign.",
    )
    hunt_parser.add_argument("--backend", choices=["aot_eager", "eager", "inductor"], default="aot_eager")
    hunt_parser.add_argument("--mode", choices=["forward", "gradient"], default="forward")
    hunt_parser.add_argument(
        "--family",
        choices=["random", "broadcast", "dynamic", "alias_mutation"],
        default="random",
    )
    hunt_parser.add_argument("--cases", type=int, default=100)
    hunt_parser.add_argument("--seed", type=int, default=0)
    hunt_parser.add_argument("--confirm-runs", type=int, default=5)
    hunt_parser.add_argument(
        "--case-timeout",
        type=float,
        default=0.0,
        help="per-case budget in seconds; 0 disables the budget",
    )
    hunt_parser.add_argument("--output")
    hunt_parser.add_argument("--coverage-output")
    summarize_parser = commands.add_parser(
        "summarize",
        help="cluster hunt failure records",
        description="Cluster compiler/runtime failures from a saved campaign JSON file.",
    )
    summarize_parser.add_argument("campaign")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reduce":
        oracle = ExceptionOracle(
            exception_type=args.exception_type,
            message_regex=args.message,
        )
        try:
            result = reduce(
                args.program,
                oracle=oracle,
                timeout=args.timeout,
                cache=args.cache,
            )
            output = result.export(args.output)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"reproreduce: error: {error}", file=sys.stderr)
            return 1
        print(result.summary())
        print(f"Output: {output}")
        return 0
    if args.command == "summarize":
        try:
            from ..hunt.failures import load_failure_clusters

            clusters = load_failure_clusters(args.campaign)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"reproreduce: error: {error}", file=sys.stderr)
            return 1
        print(json.dumps({"cluster_count": len(clusters), "clusters": clusters}, indent=2, sort_keys=True))
        return 0
    if args.command == "hunt":
        try:
            from ..hunt import ConfirmationPolicy
            from ..hunt.campaign import run_campaign, write_campaign_report, write_coverage_summary

            stats, _ = run_campaign(
                cases=args.cases,
                seed=args.seed,
                backend=args.backend,
                mode=args.mode,
                confirmation=ConfirmationPolicy(
                    attempts=args.confirm_runs,
                    min_successes=args.confirm_runs,
                ),
                case_timeout=args.case_timeout,
                checkpoint=args.output,
                family=args.family,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"reproreduce: error: {error}", file=sys.stderr)
            return 1
        print(json.dumps(stats.summary_dict(), indent=2))
        if args.output:
            print(f"Output: {write_campaign_report(stats, args.output)}")
            coverage_output = args.coverage_output
            if coverage_output is None:
                from pathlib import Path

                coverage_output = str(Path(args.output).with_suffix(".coverage.md"))
            print(f"Coverage: {write_coverage_summary(stats, coverage_output)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
