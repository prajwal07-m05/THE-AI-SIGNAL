"""CLI entrypoint for the Frontier Intelligence Pipeline.

Examples
--------
    # Full run at trial targets (1k each of startups/products/papers + fresh signals)
    python -m src.main run --startups 1000 --products 1000 --papers 1000

    # Just the research-papers vertical, 200 records
    python -m src.main run --papers 200 --no-startups --no-news --no-jobs

    # Skip LLM (structured sources already give clean data) for a fast dry-run
    python -m src.main run --papers 50 --no-llm
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


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    pipe = Pipeline(use_llm=not args.no_llm)
    started = time.monotonic()

    async with AsyncFetcher() as fetcher:
        tasks = []
        if not args.no_papers and args.papers:
            tasks.append(pipe.run_papers(args.papers, fetcher))
        if not args.no_startups and args.startups:
            tasks.append(pipe.run_startups(args.startups, fetcher))
        if not args.no_news:
            tasks.append(pipe.run_news(args.news, fetcher))
        if not args.no_jobs:
            tasks.append(pipe.run_jobs(args.jobs, fetcher))
        # Verticals run concurrently; each internally bounds its own concurrency.
        await asyncio.gather(*tasks)

    OutputWriter().write(pipe.output, pipe.resolver.log)
    elapsed = time.monotonic() - started

    console.rule("[bold green]Pipeline complete")
    console.print(
        {
            "startups": len(pipe.output["startups"]),
            "products": len(pipe.output["products"]),
            "papers": len(pipe.output["papers"]),
            "jobs": len(pipe.output["jobs"]),
            "news": len(pipe.output["news"]),
            "entity_mappings": len(pipe.resolver.log),
            **pipe.stats,
            "elapsed_s": round(elapsed, 1),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="frontier-intel", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the ingestion pipeline")
    r.add_argument("--startups", type=int, default=1000)
    r.add_argument("--products", type=int, default=1000)  # products come from startups source
    r.add_argument("--papers", type=int, default=1000)
    r.add_argument("--news", type=int, default=500)
    r.add_argument("--jobs", type=int, default=500)
    r.add_argument("--no-startups", action="store_true")
    r.add_argument("--no-papers", action="store_true")
    r.add_argument("--no-news", action="store_true")
    r.add_argument("--no-jobs", action="store_true")
    r.add_argument("--no-llm", action="store_true", help="skip LLM structuring")

    args = p.parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
