"""CLI entrypoint for the Frontier Intelligence Pipeline.

Examples
--------
Full run:

    python -m src.main run

Research papers only:

    python -m src.main run --only papers --papers 200

Fast papers dry-run without LLM:

    python -m src.main run --only papers --papers 50 --no-llm

Disable individual verticals during a full run:

    python -m src.main run --no-startups --no-news --no-jobs
"""

from __future__ import annotations

import argparse
import asyncio
import time

from rich.console import Console

from src.core.http_client import AsyncFetcher
from src.core.logging import configure_logging, get_logger
from src.output.sheets_writer import OutputWriter
from src.pipeline.runner import Pipeline

console = Console()
log = get_logger("main")


VERTICALS = (
    "startups",
    "papers",
    "news",
    "jobs",
)


def _build_tasks(
    args: argparse.Namespace,
    pipe: Pipeline,
    fetcher: AsyncFetcher,
) -> list:
    """Build the requested ingestion tasks deterministically."""
    if args.only:
        if args.only == "startups":
            return [
                pipe.run_startups(
                    args.startups,
                    fetcher,
                )
            ]

        if args.only == "papers":
            return [
                pipe.run_papers(
                    args.papers,
                    fetcher,
                )
            ]

        if args.only == "news":
            return [
                pipe.run_news(
                    args.news,
                    fetcher,
                )
            ]

        if args.only == "jobs":
            return [
                pipe.run_jobs(
                    args.jobs,
                    fetcher,
                )
            ]

        raise ValueError(
            f"Unsupported vertical: {args.only}"
        )

    tasks = []

    if not args.no_papers and args.papers > 0:
        tasks.append(
            pipe.run_papers(
                args.papers,
                fetcher,
            )
        )

    if not args.no_startups and args.startups > 0:
        tasks.append(
            pipe.run_startups(
                args.startups,
                fetcher,
            )
        )

    if not args.no_news and args.news > 0:
        tasks.append(
            pipe.run_news(
                args.news,
                fetcher,
            )
        )

    if not args.no_jobs and args.jobs > 0:
        tasks.append(
            pipe.run_jobs(
                args.jobs,
                fetcher,
            )
        )

    return tasks


async def _run(args: argparse.Namespace) -> None:
    configure_logging()

    pipe = Pipeline(
        use_llm=not args.no_llm
    )

    started = time.monotonic()

    async with AsyncFetcher() as fetcher:
        tasks = _build_tasks(
            args,
            pipe,
            fetcher,
        )

        if not tasks:
            raise SystemExit(
                "No ingestion verticals selected. "
                "Enable at least one vertical or use --only."
            )

        # All requested verticals share the same AsyncFetcher, giving them a
        # common HTTP connection pool, global rate limiter, retry policy and
        # concurrency controls.
        await asyncio.gather(
            *tasks
        )

    OutputWriter().write(
        pipe.output,
        pipe.resolver.log,
    )

    elapsed = time.monotonic() - started

    console.rule(
        "[bold green]Pipeline complete"
    )

    console.print(
        {
            "startups": len(
                pipe.output["startups"]
            ),
            "products": len(
                pipe.output["products"]
            ),
            "papers": len(
                pipe.output["papers"]
            ),
            "jobs": len(
                pipe.output["jobs"]
            ),
            "news": len(
                pipe.output["news"]
            ),
            "entity_mappings": len(
                pipe.resolver.log
            ),
            **pipe.stats,
            "elapsed_s": round(
                elapsed,
                1,
            ),
        }
    )


def _positive_int(value: str) -> int:
    """Argparse validator for non-negative record limits."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "must be greater than or equal to 0"
        )

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="frontier-intel",
        description=__doc__,
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    run = sub.add_parser(
        "run",
        help="run the ingestion pipeline",
    )

    run.add_argument(
        "--startups",
        type=_positive_int,
        default=1000,
        help="maximum startup records to request",
    )

    run.add_argument(
        "--products",
        type=_positive_int,
        default=1000,
        help=(
            "reserved product target; products currently originate "
            "from the startup source"
        ),
    )

    run.add_argument(
        "--papers",
        type=_positive_int,
        default=1000,
        help="maximum research papers to request",
    )

    run.add_argument(
        "--news",
        type=_positive_int,
        default=500,
        help="maximum news records to request",
    )

    run.add_argument(
        "--jobs",
        type=_positive_int,
        default=500,
        help="maximum job records to request",
    )

    run.add_argument(
        "--only",
        choices=VERTICALS,
        help=(
            "run exactly one vertical; when supplied, all other "
            "vertical flags are ignored"
        ),
    )

    run.add_argument(
        "--no-startups",
        action="store_true",
        help="disable startup ingestion",
    )

    run.add_argument(
        "--no-papers",
        action="store_true",
        help="disable research-paper ingestion",
    )

    run.add_argument(
        "--no-news",
        action="store_true",
        help="disable news ingestion",
    )

    run.add_argument(
        "--no-jobs",
        action="store_true",
        help="disable job ingestion",
    )

    run.add_argument(
        "--no-llm",
        action="store_true",
        help="skip LLM structuring/enrichment",
    )

    args = parser.parse_args()

    if args.cmd == "run":
        asyncio.run(
            _run(args)
        )


if __name__ == "__main__":
    main()