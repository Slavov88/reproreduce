from __future__ import annotations

import argparse
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
