from __future__ import annotations

import argparse
import json
import sys

from ..api import reduce
from ..oracle.exception import ExceptionOracle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reproreduce")
    commands = parser.add_subparsers(dest="command", required=True)
    reduce_parser = commands.add_parser("reduce", help="reduce a failing Python script")
    reduce_parser.add_argument("program")
    reduce_parser.add_argument("--oracle", choices=["exception"], default="exception")
    reduce_parser.add_argument("--exception-type")
    reduce_parser.add_argument("--message")
    reduce_parser.add_argument("--timeout", type=float, default=30.0)
    reduce_parser.add_argument("--output", default="repro")
    reduce_parser.add_argument("--cache")
    hunt_parser = commands.add_parser("hunt", help="search generated PyTorch programs")
    hunt_parser.add_argument("--backend", choices=["aot_eager", "eager", "inductor"], default="aot_eager")
    hunt_parser.add_argument("--mode", choices=["forward", "gradient"], default="forward")
    hunt_parser.add_argument("--cases", type=int, default=100)
    hunt_parser.add_argument("--seed", type=int, default=0)
    hunt_parser.add_argument("--confirm-runs", type=int, default=5)
    hunt_parser.add_argument("--case-timeout", type=float, default=120.0)
    hunt_parser.add_argument("--output")
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
    if args.command == "hunt":
        try:
            from ..hunt import ConfirmationPolicy
            from ..hunt.campaign import run_campaign, write_campaign_report

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
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"reproreduce: error: {error}", file=sys.stderr)
            return 1
        print(json.dumps(stats.summary_dict(), indent=2))
        if args.output:
            print(f"Output: {write_campaign_report(stats, args.output)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
