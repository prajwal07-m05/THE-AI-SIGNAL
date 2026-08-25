"""Tests for the CLI entrypoint and task dispatch."""

from __future__ import annotations

import argparse
import asyncio

import pytest

from src import main as cli


class FakePipeline:
    def __init__(self):
        self.calls: list[tuple[str, int, object]] = []

    async def run_startups(self, limit, fetcher):
        self.calls.append(("startups", limit, fetcher))

    async def run_papers(self, limit, fetcher):
        self.calls.append(("papers", limit, fetcher))

    async def run_news(self, limit, fetcher):
        self.calls.append(("news", limit, fetcher))

    async def run_jobs(self, limit, fetcher):
        self.calls.append(("jobs", limit, fetcher))


def _args(**overrides):
    values = {
        "only": None,
        "startups": 1000,
        "products": 1000,
        "papers": 1000,
        "news": 500,
        "jobs": 500,
        "no_startups": False,
        "no_papers": False,
        "no_news": False,
        "no_jobs": False,
        "no_llm": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


async def _run_tasks(tasks):
    await asyncio.gather(*tasks)


@pytest.mark.parametrize(
    ("only", "expected"),
    [
        ("startups", "startups"),
        ("papers", "papers"),
        ("news", "news"),
        ("jobs", "jobs"),
    ],
)
def test_only_selects_exactly_one_vertical(
    only: str,
    expected: str,
):
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            only=only,
            startups=11,
            papers=22,
            news=33,
            jobs=44,
        ),
        pipe,
        fetcher,
    )

    assert len(tasks) == 1

    asyncio.run(_run_tasks(tasks))

    expected_limits = {
        "startups": 11,
        "papers": 22,
        "news": 33,
        "jobs": 44,
    }

    assert [
        (name, limit)
        for name, limit, _ in pipe.calls
    ] == [
        (expected, expected_limits[expected])
    ]


def test_full_run_selects_all_enabled_verticals():
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            startups=10,
            papers=20,
            news=30,
            jobs=40,
        ),
        pipe,
        fetcher,
    )

    assert len(tasks) == 4

    asyncio.run(_run_tasks(tasks))

    assert [
        (name, limit)
        for name, limit, _ in pipe.calls
    ] == [
        ("papers", 20),
        ("startups", 10),
        ("news", 30),
        ("jobs", 40),
    ]


def test_disabled_verticals_are_not_scheduled():
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            startups=10,
            papers=20,
            news=30,
            jobs=40,
            no_startups=True,
            no_news=True,
        ),
        pipe,
        fetcher,
    )

    assert len(tasks) == 2

    asyncio.run(_run_tasks(tasks))

    assert [
        (name, limit)
        for name, limit, _ in pipe.calls
    ] == [
        ("papers", 20),
        ("jobs", 40),
    ]


def test_zero_limit_prevents_vertical_from_being_scheduled():
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            startups=0,
            papers=0,
            news=10,
            jobs=0,
        ),
        pipe,
        fetcher,
    )

    assert len(tasks) == 1

    asyncio.run(_run_tasks(tasks))

    assert [
        (name, limit)
        for name, limit, _ in pipe.calls
    ] == [
        ("news", 10),
    ]


def test_only_ignores_disable_flags():
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            only="papers",
            papers=25,
            no_papers=True,
            no_startups=True,
            no_news=True,
            no_jobs=True,
        ),
        pipe,
        fetcher,
    )

    assert len(tasks) == 1

    asyncio.run(_run_tasks(tasks))

    assert [
        (name, limit)
        for name, limit, _ in pipe.calls
    ] == [
        ("papers", 25),
    ]


def test_only_uses_selected_vertical_limit():
    pipe = FakePipeline()
    fetcher = object()

    tasks = cli._build_tasks(
        _args(
            only="jobs",
            startups=999,
            papers=999,
            news=999,
            jobs=37,
        ),
        pipe,
        fetcher,
    )

    asyncio.run(_run_tasks(tasks))

    assert pipe.calls == [
        ("jobs", 37, fetcher),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "1",
        "10",
        "999999",
    ],
)
def test_positive_int_accepts_non_negative_values(value: str):
    assert cli._positive_int(value) == int(value)


@pytest.mark.parametrize(
    "value",
    [
        "-1",
        "-100",
    ],
)
def test_positive_int_rejects_negative_values(value: str):
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="greater than or equal to 0",
    ):
        cli._positive_int(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "1.5",
        "10x",
    ],
)
def test_positive_int_rejects_non_integer_values(value: str):
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="must be an integer",
    ):
        cli._positive_int(value)


def test_parser_requires_subcommand():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("run")

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_parser_accepts_run_options():
    parser = argparse.ArgumentParser(
        prog="frontier-intel",
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    run = sub.add_parser("run")

    run.add_argument(
        "--papers",
        type=cli._positive_int,
        default=1000,
    )

    run.add_argument(
        "--only",
        choices=cli.VERTICALS,
    )

    run.add_argument(
        "--no-llm",
        action="store_true",
    )

    args = parser.parse_args(
        [
            "run",
            "--papers",
            "25",
            "--only",
            "papers",
            "--no-llm",
        ]
    )

    assert args.cmd == "run"
    assert args.papers == 25
    assert args.only == "papers"
    assert args.no_llm is True


def test_verticals_are_explicit_and_stable():
    assert cli.VERTICALS == (
        "startups",
        "papers",
        "news",
        "jobs",
    )