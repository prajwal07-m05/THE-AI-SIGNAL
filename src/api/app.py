"""Web dashboard for the Frontier Intelligence Pipeline."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(
    title="Frontier Intelligence Pipeline",
    description="Dashboard for Frontier Intelligence pipeline outputs.",
    version="1.0.0",
)

DEMO_DIR = Path("./dashboard_data")
OUT_DIR = Path("./out")

# Use the committed demo dataset when available (deployment),
# otherwise fall back to the runtime output directory.
DATA_DIR = DEMO_DIR if DEMO_DIR.exists() else OUT_DIR

CSV_FILES = {
    "startups": DATA_DIR / "startups.csv",
    "products": DATA_DIR / "products.csv",
    "papers": DATA_DIR / "research_papers.csv",
    "jobs": DATA_DIR / "jobs.csv",
    "news": DATA_DIR / "news.csv",
    "entity_mapping": DATA_DIR / "entity_mapping_log.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file safely."""
    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as fh:
            return list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _count(path: Path) -> int:
    """Return the number of data rows in a CSV file."""
    return len(_read_csv(path))


def _latest_records(
    path: Path,
    limit: int = 10,
) -> list[dict[str, str]]:
    """Return the latest rows from a CSV file."""
    records = _read_csv(path)

    if not records:
        return []

    return records[-limit:][::-1]


def _dataset() -> dict[str, Any]:
    """Build dashboard statistics."""
    counts = {
        key: _count(path)
        for key, path in CSV_FILES.items()
    }

    total = sum(
        counts[key]
        for key in (
            "startups",
            "products",
            "papers",
            "jobs",
            "news",
        )
    )

    return {
        "counts": counts,
        "total": total,
    }


def _escape(value: str) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(value))


def _render_table(
    records: list[dict[str, str]],
) -> str:
    """Render recent records as an HTML table."""
    if not records:
        return '<div class="empty">No records available.</div>'

    preferred = [
        "content.title",
        "content.entityName",
        "content.startupName",
        "content.company",
        "content.authors",
        "content.paper_url",
        "content.published_date",
    ]

    headers = [
        key
        for key in preferred
        if any(key in record for record in records)
    ]

    if not headers:
        headers = list(records[0].keys())[:5]

    header_html = "".join(
        f"<th>{_escape(header)}</th>"
        for header in headers
    )

    rows_html = []

    for record in records:
        cells = []

        for header in headers:
            value = record.get(header, "")

            if len(value) > 180:
                value = value[:177] + "..."

            cells.append(
                f"<td>{_escape(value)}</td>"
            )

        rows_html.append(
            "<tr>" + "".join(cells) + "</tr>"
        )

    return (
        "<table>"
        "<thead><tr>"
        f"{header_html}"
        "</tr></thead>"
        "<tbody>"
        f"{''.join(rows_html)}"
        "</tbody>"
        "</table>"
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Render health-check endpoint."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "frontier-intelligence-pipeline",
        }
    )


@app.get("/api/stats")
async def stats() -> JSONResponse:
    """Return dashboard statistics."""
    return JSONResponse(_dataset())


@app.get("/api/records/{vertical}")
async def records(
    vertical: str,
    limit: int = 20,
) -> JSONResponse:
    """Return recent records for one vertical."""
    if vertical not in {
        "startups",
        "products",
        "papers",
        "jobs",
        "news",
    }:
        return JSONResponse(
            {
                "error": "Unknown vertical",
                "allowed": [
                    "startups",
                    "products",
                    "papers",
                    "jobs",
                    "news",
                ],
            },
            status_code=404,
        )

    limit = max(1, min(limit, 100))

    return JSONResponse(
        {
            "vertical": vertical,
            "count": _count(CSV_FILES[vertical]),
            "records": _latest_records(
                CSV_FILES[vertical],
                limit,
            ),
        }
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Render the main dashboard."""
    data = _dataset()
    counts = data["counts"]

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>Frontier Intelligence</title>

    <style>
        :root {{
            --bg: #09090b;
            --panel: #111113;
            --panel-hover: #18181b;
            --border: #27272a;
            --text: #fafafa;
            --muted: #a1a1aa;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }}

        .container {{
            width: min(1180px, calc(100% - 40px));
            margin: 0 auto;
            padding: 42px 0 70px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 20px;
            margin-bottom: 34px;
        }}

        .eyebrow {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .14em;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 9px;
        }}

        h1 {{
            margin: 0;
            font-size: clamp(30px, 5vw, 48px);
            letter-spacing: -0.04em;
        }}

        .subtitle {{
            margin-top: 10px;
            color: var(--muted);
            max-width: 680px;
            line-height: 1.6;
        }}

        .status {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 999px;
            padding: 9px 14px;
            font-size: 13px;
            white-space: nowrap;
        }}

        .status-dot {{
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #a1a1aa;
            margin-right: 8px;
        }}

        .total {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 12px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 28px;
        }}

        .card {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 16px;
            padding: 20px;
            transition: background .15s ease;
        }}

        .card:hover {{
            background: var(--panel-hover);
        }}

        .label {{
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 12px;
        }}

        .value {{
            font-size: 30px;
            font-weight: 750;
            letter-spacing: -0.03em;
        }}

        .total .value {{
            font-size: 42px;
        }}

        .section {{
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 18px;
            overflow: hidden;
        }}

        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 20px;
            border-bottom: 1px solid var(--border);
        }}

        .section-title {{
            font-size: 16px;
            font-weight: 700;
        }}

        .section-meta {{
            color: var(--muted);
            font-size: 13px;
        }}

        .records {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            min-width: 720px;
        }}

        th,
        td {{
            text-align: left;
            padding: 13px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
            vertical-align: top;
        }}

        th {{
            color: var(--muted);
            font-weight: 600;
            background: #0d0d0f;
        }}

        td {{
            color: #e4e4e7;
        }}

        tr:last-child td {{
            border-bottom: 0;
        }}

        .empty {{
            padding: 32px 20px;
            color: var(--muted);
            text-align: center;
        }}

        .footer {{
            margin-top: 24px;
            color: var(--muted);
            font-size: 12px;
            text-align: center;
        }}

        @media (max-width: 850px) {{
            .grid {{
                grid-template-columns: repeat(2, 1fr);
            }}

            .header {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}

        @media (max-width: 520px) {{
            .container {{
                width: min(100% - 24px, 1180px);
                padding-top: 26px;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>

<body>
    <main class="container">
        <header class="header">
            <div>
                <div class="eyebrow">AI Signal</div>

                <h1>Frontier Intelligence</h1>

                <div class="subtitle">
                    Multi-vertical intelligence pipeline for
                    startups, products, research papers, jobs,
                    and AI news.
                </div>
            </div>

            <div class="status">
                <span class="status-dot"></span>
                Pipeline dashboard online
            </div>
        </header>

        <section class="total">
            <div class="label">
                Total intelligence records
            </div>

            <div class="value">
                {data["total"]:,}
            </div>
        </section>

        <section class="grid">
            <div class="card">
                <div class="label">Startups</div>
                <div class="value">{counts["startups"]:,}</div>
            </div>

            <div class="card">
                <div class="label">Products</div>
                <div class="value">{counts["products"]:,}</div>
            </div>

            <div class="card">
                <div class="label">Research Papers</div>
                <div class="value">{counts["papers"]:,}</div>
            </div>

            <div class="card">
                <div class="label">Jobs</div>
                <div class="value">{counts["jobs"]:,}</div>
            </div>

            <div class="card">
                <div class="label">AI News</div>
                <div class="value">{counts["news"]:,}</div>
            </div>
        </section>

        <section class="section">
            <div class="section-header">
                <div class="section-title">
                    Recent research papers
                </div>

                <div class="section-meta">
                    {counts["papers"]:,} records
                </div>
            </div>

            <div class="records">
                {_render_table(
                    _latest_records(
                        CSV_FILES["papers"],
                        10,
                    )
                )}
            </div>
        </section>

        <div class="footer">
            Frontier Intelligence Pipeline · AI Signal
        </div>
    </main>
</body>
</html>
"""
